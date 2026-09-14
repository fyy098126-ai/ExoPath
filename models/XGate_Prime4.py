"""
XGate_Prime4  —  Pure Low-Rank Gated Causal Forecaster (No Attention)

Core Design Philosophy in Prime4:
----------------------------------
1. Essence of XLinear (The Global Token):
   XLinear's genuine power comes from the Global Token and pure Sigmoid self-gating blocks 
   (FeatureGate), NOT from softmax attention. Prime4 fully inherits this essence:
   - Maintains the authentic learnable `glob_token` in the Temporal Gating Module (TGM).
   - Uses pure Sigmoid-gated FeatureGates (MLPs with Sigmoid weights) to route all features.
   - NO softmax, NO query-key dot-product attention anywhere in the seasonal or trend paths.

2. Low-Rank Gating Block (LRGB) to Conquer the Curse of Dimensionality:
   In original XLinear, the Variate Gating Module (VGM) gates the channel dimension. For high-channel 
   datasets (e.g. Traffic C=862), the cross-channel gating linear layers scale as O(C^2) (e.g. 1724x1724 ≈ 3M params).
   Prime4 factorizes the channel gating into a low-rank bottleneck:
     gate = Sigmoid( W_up( ReLU( W_down( x ) ) ) )
     where W_down: C_in -> R, W_up: R -> C_out, and R is the low-rank bottleneck (rank << C).
   This reduces Variate Gating parameters by up to 98% (down to ~40K parameters for Traffic!), 
   enabling pure channel-gating at scale without parameter explosion.

3. SLR-Mixer (Symmetric Low-Rank Trend Mixer):
   Symmetric cross-channel trend mixing is also factorized into low-rank down-up projections 
   C -> R -> C_out. This captures global co-trending co-drifts using pure lightweight linear mixing 
   (conceptually inheriting CrossLinear's spatial mixing strengths in a low-rank gated fashion).

4. Detailed Ablation Matrix:
   - full           : Complete pure gated Prime4 model
   - wo_revin       : No RevIN distribution normalization
   - wo_decomp      : No EMA decomposition (seasonal path gets raw signal, trend skipped)
   - wo_trend       : Skipping trend prediction head (seasonal path only)
   - wo_trend_mix   : Skipping SLR-Mixer (channel-independent trend)
   - wo_exo_trend   : Trend path only uses target channel (no cross-channel trend)
   - wo_tgm         : Skipping TGM (use raw z_endo)
   - wo_query_raw   : VGM uses raw z_endo instead of temporal-enriched glob_gated
   - wo_router      : Skipping VGM cross-channel gating (no causal exog context)
   - wo_soft_sparse : Equivalent to full (no-op fallback since attention is removed)
   - wo_group_mix   : Equivalent to full (no-op fallback since attention is removed)
   - wo_global      : Bypassing learned VGM routing (gating weights set to 0.5)
   - wo_origin      : Removing pure-self temporal path in seasonal prediction head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. RevIN 
# =========================================================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm', target_dim=None):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        else:
            mean = self.mean[:, :, -target_dim:] if target_dim else self.mean
            stdev = self.stdev[:, :, -target_dim:] if target_dim else self.stdev
            if self.affine:
                weight = self.affine_weight[-target_dim:] if target_dim else self.affine_weight
                bias = self.affine_bias[-target_dim:] if target_dim else self.affine_bias
                x = (x - bias) / weight
            return x * stdev + mean


# =========================================================================
# 2. DynamicEMADecomp  
# =========================================================================
class DynamicEMADecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.scales = [kernel_size // 2, kernel_size, kernel_size * 2]
        self.scale_weights = nn.Parameter(torch.zeros(len(self.scales)))

    def forward(self, x):
        B, L, C = x.shape
        norm_w = F.softmax(self.scale_weights, dim=0)
        trends = []
        for k in self.scales:
            k = k if k % 2 != 0 else k + 1
            p = k // 2
            x_pad = F.pad(x.permute(0, 2, 1), (p, p), mode='replicate')
            trends.append(F.avg_pool1d(x_pad, kernel_size=k, stride=1).permute(0, 2, 1))
        final_trend = sum(t * norm_w[i] for i, t in enumerate(trends))
        return final_trend, x - final_trend




# =========================================================================
# 3. FeatureGate  (sigmoid self-gating: x * sigmoid(MLP(x)))
# =========================================================================
class FeatureGate(nn.Module):
    def __init__(self, d_in, hf, dropout=0.1):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(d_in, hf), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hf, d_in), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.weight(x)


# =========================================================================
# 4. Low-Rank Feature Gate (LR-Gate)
#    Gates the channel dimension through a low-rank bottleneck C_in -> R -> C_out.
#    Replaces expensive O(C_in * C_out) gating with highly efficient O((C_in + C_out) * R).
# =========================================================================
class LowRankFeatureGate(nn.Module):
    def __init__(self, c_in, c_out, rank=16, dropout=0.1):
        super().__init__()
        self.rank = min(rank, c_in, c_out)
        self.down = nn.Linear(c_in, self.rank, bias=False)
        self.up = nn.Linear(self.rank, c_out, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, d_model, C_in] (channel dimension is the last dimension)
        gate_logits = self.up(self.dropout(F.relu(self.down(x))))  # [B, d_model, C_out]
        return torch.sigmoid(gate_logits)


# =========================================================================
# 5. SLR-Mixer (Symmetric Low-Rank Trend Mixer)
#    Pure linear symmetric channel mixing using low-rank factorization C -> R -> C_out.
# =========================================================================
class SLRTrendMixer(nn.Module):
    def __init__(self, c_in, c_out, d_model, rank=16, dropout=0.1, cross_gate_init=-4.0):
        super().__init__()
        self.c_out = c_out
        self.rank = min(rank, c_in, c_out)

        # Low-rank factorized linear mapping
        self.down = nn.Linear(c_in, self.rank, bias=False)
        self.up = nn.Linear(self.rank, c_out, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)

        self.cross_gate = nn.Parameter(torch.full((1, c_out, 1), cross_gate_init))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_trend_all):
        # z_trend_all: [B, C_in, d_model]
        endo = z_trend_all[:, -self.c_out:, :]  # [B, C_out, d_model]
        
        # Symmetrically mix channels in low-rank bottleneck space
        x = z_trend_all.transpose(1, 2)  # [B, d_model, C_in]
        cross = self.up(self.down(x)).transpose(1, 2)  # [B, C_out, d_model]

        gate = torch.sigmoid(self.cross_gate)
        return endo + gate * self.dropout(cross - endo)


# =========================================================================
# 6. Model (XGate_Prime4 Pure Gated Main Class)
# =========================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ablation_mode = getattr(configs, 'ablation_mode', 'full')
        self.C = configs.enc_in
        self.features = getattr(configs, 'features', 'M')
        self.out_dim = 1 if self.features == 'MS' else self.C
        self.exo_dim = self.C - self.out_dim if self.features == 'MS' else self.C

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        t_ff = getattr(configs, 't_ff', 0) or (self.d_model * 4)
        
        # Adaptive Bottleneck Rank: scales automatically with dataset channels
        self.rank = getattr(configs, 'num_groups', 0) or (
            16 if self.C >= 321 else 8 if self.C >= 21 else 4
        )
        drop = getattr(configs, 'dropout', 0.1)

        # ── RevIN + Decomp ─────────────────────────────────────────────────
        self.revin = RevIN(self.C)
        self.decomp = DynamicEMADecomp(kernel_size=getattr(configs, 'ema_kernel', 25))

        # ── Trend path (SLR Symmetric Mixer) ───────────────────────────────
        self.trend_proj = nn.Linear(self.seq_len, self.d_model)
        
        # Low-rank Symmetric Trend Mixer (SLR-Mixer)
        cross_gate_init = getattr(configs, 'cross_gate_init', -4.0)
        self.trend_mixer = SLRTrendMixer(self.C, self.out_dim, self.d_model, rank=self.rank, dropout=drop, cross_gate_init=cross_gate_init)
        self.trend_norm = nn.LayerNorm(self.d_model)
        self.trend_mlp = nn.Sequential(
            nn.Linear(self.d_model, t_ff), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(t_ff, self.d_model),
        )
        self.trend_head = nn.Linear(self.d_model, self.pred_len)
        
        trend_mix_alpha_init = getattr(configs, 'trend_mix_alpha_init', -2.0)
        self.trend_mix_alpha = nn.Parameter(torch.tensor(trend_mix_alpha_init))  # sigmoid(-2) ≈ 0.12
        self.trend_mix_scale = getattr(configs, 'trend_mix_scale', 1.0)

        # ── Seasonal path (TGM + VGM Pure Gating) ─────────────────────────
        self.seasonal_proj = nn.Linear(self.seq_len, self.d_model)
        
        # 1. Temporal Gating Module (TGM): Authentic XLinear Global Token + FeatureGate
        self.glob_token = nn.Parameter(torch.ones(1, self.out_dim, self.d_model))
        self.en_attention = FeatureGate(2 * self.d_model, t_ff, drop)
        
        # 2. Variate Gating Module (VGM): Low-Rank Gating conquer O(C^2)
        self.c_in_vgm = (self.C + self.out_dim) if self.features == 'M' else (self.exo_dim + self.out_dim)
        self.vgm_gate = LowRankFeatureGate(self.c_in_vgm, self.out_dim, rank=self.rank, dropout=drop)
        
        # Prediction head
        self.seasonal_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(2 * self.d_model, self.pred_len),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        mode = self.ablation_mode
        b = x_enc.shape[0]

        # ── RevIN ──────────────────────────────────────────────────────────
        x = self.revin(x_enc, 'norm') if mode != 'wo_revin' else x_enc

        # ── Decomp ─────────────────────────────────────────────────────────
        if mode == 'wo_decomp':
            seasonal_all = x
        else:
            trend_all, seasonal_all = self.decomp(x)

        # ── Trend path ─────────────────────────────────────────────────────
        if mode not in ('wo_trend', 'wo_decomp'):
            z_trend = self.trend_proj(trend_all.permute(0, 2, 1))  # [B, C, dm]
            
            if mode == 'wo_exo_trend':
                # Ablation: Trend only uses target channel (no cross-channel trend)
                z_trend = z_trend[:, -self.out_dim:, :]
            elif mode != 'wo_trend_mix':
                # Symmetric Low-Rank Trend Mixing (SLR-Mixer)
                z_trend = self.trend_mixer(z_trend)  # [B, C_out, dm]
            else:
                z_trend = z_trend[:, -self.out_dim:, :]
                
            z_trend = self.trend_mlp(self.trend_norm(z_trend)) + z_trend
            trend_pred = self.trend_head(z_trend).transpose(1, 2)  # [B, T, C_out]

        # ── Seasonal path ──────────────────────────────────────────────────
        z_seas = self.seasonal_proj(seasonal_all.permute(0, 2, 1))  # [B, C, dm]
        z_endo = z_seas[:, -self.out_dim:, :]
        z_exo = z_seas[:, :-self.out_dim, :] if self.features == 'MS' else z_seas

        # --- Temporal Gating Module (TGM) ---
        if mode == 'wo_tgm':
            origin_gated = z_endo
            glob_gated = z_endo
        else:
            glob = self.glob_token.repeat(b, 1, 1)
            en_emb = torch.cat([z_endo, glob], dim=-1)  # [B, C_out, 2 * d_model]
            en_gated = self.en_attention(en_emb)
            origin_gated = en_gated[:, :, :self.d_model]
            glob_gated = en_gated[:, :, self.d_model:]

        # --- Variate Gating Module (VGM) ---
        if mode != 'wo_router' and self.c_in_vgm > self.out_dim:
            query_vec = z_endo if mode == 'wo_query_raw' else glob_gated
            
            # Concatenate exogenous and temporal global token along the channel dimension
            ex_emb = torch.cat([z_exo, query_vec], dim=1)  # [B, c_in_vgm, d_model]
            
            if mode == 'wo_global':
                # Uniform mean gating weight (0.5)
                gate = torch.full((b, self.d_model, self.out_dim), 0.5, device=x_enc.device)
            else:
                # Transpose to [B, d_model, c_in_vgm] and pass through Low-Rank Gating
                gate = self.vgm_gate(ex_emb.transpose(1, 2))  # [B, d_model, out_dim]
                
            glob_final = glob_gated * gate.transpose(1, 2)  # [B, C_out, d_model]
        else:
            glob_final = glob_gated

        # Dual-path seasonal prediction head
        seas_in = (torch.cat([glob_final, glob_final], dim=-1)
                   if mode == 'wo_origin'
                   else torch.cat([origin_gated, glob_final], dim=-1))
        seasonal_pred = self.seasonal_head(seas_in).transpose(1, 2)  # [B, T, C_out]

        # ── Combine ────────────────────────────────────────────────────────
        if mode in ('wo_trend', 'wo_decomp'):
            res_out = seasonal_pred
        else:
            t_w = self.trend_mix_scale * torch.sigmoid(self.trend_mix_alpha)
            res_out = t_w * trend_pred + seasonal_pred

        # ── RevIN inverse ──────────────────────────────────────────────────
        if mode != 'wo_revin':
            res_out = self.revin(res_out, 'denorm',
                                 target_dim=self.out_dim if self.features == 'MS' else None)
        return res_out
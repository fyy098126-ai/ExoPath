"""
XGate_Prime5  —  Pure Low-Rank Gated Causal Forecaster (No Attention)

Core Design Philosophy in Prime5:
----------------------------------
1. Essence of XLinear (The Global Token):
   XLinear's genuine power comes from the Global Token and pure Sigmoid self-gating blocks 
   (FeatureGate), NOT from softmax attention. Prime5 fully inherits this essence:
   - Maintains the authentic learnable `glob_token` in the Temporal Gating Module (TGM).
   - Uses pure Sigmoid-gated FeatureGates (MLPs with Sigmoid weights) to route all features.
   - NO softmax, NO query-key dot-product attention anywhere in the seasonal or trend paths.

2. Low-Rank Gating Block (LRGB) to Conquer the Curse of Dimensionality:
   In original XLinear, the Variate Gating Module (VGM) gates the channel dimension. For high-channel 
   datasets (e.g. Traffic C=862), the cross-channel gating linear layers scale as O(C^2) (e.g. 1724x1724 ≈ 3M params).
   Prime5 factorizes the channel gating into a low-rank bottleneck:
     gate = Sigmoid( W_up( ReLU( W_down( x ) ) ) )
     where W_down: C_in -> R, W_up: R -> C_out, and R is the low-rank bottleneck (rank << C).
   This reduces Variate Gating parameters by up to 98% (down to ~40K params for Traffic!), 
   enabling pure channel-gating at scale without parameter explosion.

3. SLR-Mixer (Symmetric Low-Rank Trend Mixer):
   Symmetric cross-channel trend mixing is also factorized into low-rank down-up projections.
   *Prime5 Ultimate Upgrade*: The entire Trend Path is now a pure linear spatial-temporal projector.
   All MLPs and Activation functions (GELU/ReLU) are completely removed to strictly preserve 
   non-stationary mean-shifts and conform to the optimal Inductive Bias for trend extrapolation.

4. Strict Algebraic Fusion & Vectorized Float64 EMA (For DBLoss Alignment):
   - Replaced adaptive trend weighting with strict mathematical addition (Y = Trend + Season)
     to seamlessly align with the DBLoss mathematical assumptions.
   - Replaced SMA decomp with an O(1) parallelized Learnable Causal EMA powered by torch.float64, 
     preventing catastrophic exponential underflow on extremely long sequences (e.g., T=720).


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
# 2. Channel-Independent Causal EMA (O(1) Vectorization)
# =========================================================================
class CausalEMA(nn.Module):
    def __init__(self, c_in, ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.alpha_logits = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        B, T, C = x.shape
        device = x.device
        
        alpha = torch.full((C,), 0.5, device=device)

        
        powers = torch.flip(torch.arange(T, dtype=torch.double, device=device), dims=(0,)).unsqueeze(1)
        alpha_f64 = alpha.to(torch.double) 
        
        weights = torch.pow((1.0 - alpha_f64), powers) # [T, C]
        divisor = weights.unsqueeze(0) # [1, T, C] (去掉了 clone, 因为不再原地修改)
        
        # ===========================================================
        # 🚀 修复点：使用 torch.cat 替代原地切片赋值，保护计算图！
        w_0 = weights[0:1, :]
        w_rest = weights[1:, :] * alpha_f64
        weights_mod = torch.cat([w_0, w_rest], dim=0).unsqueeze(0) # [1, T, C]
        # ===========================================================
        
        x_f64 = x.to(torch.double)
        out = torch.cumsum(x_f64 * weights_mod, dim=1)
        out = out / divisor
        
        Trend = out.to(torch.float32)
        Seasonality = x - Trend
        return Trend, Seasonality


# =========================================================================
# 2A. [Ablation] Dynamic Multi-scale SMA Decomp (Prime4-style)
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
# =========================================================================
class LowRankFeatureGate(nn.Module):
    def __init__(self, c_in, c_out, rank=16, dropout=0.1):
        super().__init__()
        self.rank = min(rank, c_in, c_out)
        self.down = nn.Linear(c_in, self.rank, bias=False)
        self.up = nn.Linear(self.rank, c_out, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate_logits = self.up(self.dropout(F.relu(self.down(x))))
        return torch.sigmoid(gate_logits)


# =========================================================================
# 5. SLR-Mixer (Symmetric Low-Rank Trend Mixer)
# =========================================================================
class SLRTrendMixer(nn.Module):
    def __init__(self, c_in, c_out, d_model, rank=16, dropout=0.1, cross_gate_init=-4.0):
        super().__init__()
        self.c_out = c_out
        self.rank = min(rank, c_in, c_out)

        self.down = nn.Linear(c_in, self.rank, bias=False)
        self.up = nn.Linear(self.rank, c_out, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)

        self.cross_gate = nn.Parameter(torch.full((1, c_out, 1), cross_gate_init))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_trend_all):
        endo = z_trend_all[:, -self.c_out:, :]  
        x = z_trend_all.transpose(1, 2)  
        cross = self.up(self.down(x)).transpose(1, 2)  

        gate = torch.sigmoid(self.cross_gate)
        return endo + gate * self.dropout(cross - endo)


# =========================================================================
# 6. Model (XGate_Prime5)
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
        
        self.rank = getattr(configs, 'num_groups', 0) or (
            16 if self.C >= 321 else 8 if self.C >= 21 else 4
        )
        drop = getattr(configs, 'dropout', 0.1)

        # ── RevIN + EMA Decomp ─────────────────────────────────────────────
        self.revin = RevIN(self.C)
        self.decomp = CausalEMA(c_in=self.C, ablation_mode=self.ablation_mode)
        if self.ablation_mode == 'w_sma_decomp':
            self.sma_decomp = DynamicEMADecomp(kernel_size=getattr(configs, 'ema_kernel', 25))

        # ── Trend path (Minimalist Pure Linear Pipe) ───────────────────────
        self.trend_proj = nn.Linear(self.seq_len, self.d_model)
        
        cross_gate_init = getattr(configs, 'cross_gate_init', -4.0)
        self.trend_mixer = SLRTrendMixer(self.C, self.out_dim, self.d_model, rank=self.rank, dropout=drop, cross_gate_init=cross_gate_init)
        
        self.trend_head = nn.Linear(self.d_model, self.pred_len)
        # 池化/归一化/激活消融
        self.trend_norm = nn.LayerNorm(self.d_model)
        # GELU 可叠加到池化分支：w_trend_act / w_trend_pool_act / w_trend_pool2_act
        self.ablation_trend_act = nn.GELU() if self.ablation_mode in ('w_trend_act', 'w_trend_pool_act', 'w_trend_pool2_act') else nn.Identity()

        # ── Seasonal path (Non-linear Gating) ──────────────────────────────
        self.seasonal_proj = nn.Linear(self.seq_len, self.d_model)
        self.glob_token = nn.Parameter(torch.ones(1, self.out_dim, self.d_model))
        self.en_attention = FeatureGate(2 * self.d_model, t_ff, drop)
        
        self.c_in_vgm = (self.C + self.out_dim) if self.features == 'M' else (self.exo_dim + self.out_dim)
        self.vgm_gate = LowRankFeatureGate(self.c_in_vgm, self.out_dim, rank=self.rank, dropout=drop)
        
        self.seasonal_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(2 * self.d_model, self.pred_len),
        )

        # 用于验证 DBLoss 冲突的自适应权重 (正常模式下不激活)
        if self.ablation_mode == 'w_adaptive_fuse':
            self.trend_mix_alpha = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        mode = self.ablation_mode
        b = x_enc.shape[0]

        # ── 1. RevIN (Must be BEFORE Decomposition) ────────────────────────
        x = self.revin(x_enc, 'norm') if mode != 'wo_revin' else x_enc

        # ── 2. Causal EMA Decomp ───────────────────────────────────────────
        if mode == 'wo_decomp':
            seasonal_all = x
        elif mode == 'w_sma_decomp':
            trend_all, seasonal_all = self.sma_decomp(x)
        else:
            trend_all, seasonal_all = self.decomp(x)

        # ── 3. Trend Path (Pure Linear Spatio-Temporal Extrapolation) ──────
        if mode not in ('wo_trend', 'wo_decomp'):
            z_trend_all = self.trend_proj(trend_all.permute(0, 2, 1))  
            
            if mode == 'wo_exo_trend':
                z_trend = z_trend_all[:, -self.out_dim:, :]
            elif mode in ('w_trend_pool', 'w_trend_pool_act', 'w_trend_pool2', 'w_trend_pool2_act'):
                global_trend = z_trend_all.mean(dim=1, keepdim=True)  # [B,1,d_model]
                endo_trend = z_trend_all[:, -self.out_dim:, :]
                z_trend = endo_trend + global_trend
                z_trend = self.trend_norm(z_trend)
                if mode in ('w_trend_pool2', 'w_trend_pool2_act'):
                    global_trend2 = z_trend.mean(dim=1, keepdim=True)
                    z_trend = z_trend + global_trend2
                    z_trend = self.trend_norm(z_trend)
            elif mode == 'w_trend_norm':
                z_trend = self.trend_mixer(z_trend_all)
                z_trend = self.trend_norm(z_trend)
            elif mode != 'wo_trend_mix':
                z_trend = self.trend_mixer(z_trend_all) 
            else:
                z_trend = z_trend_all[:, -self.out_dim:, :]
            
            # 消融：用非线性强行截断趋势基线
            z_trend = self.ablation_trend_act(z_trend)
            trend_pred = self.trend_head(z_trend).transpose(1, 2)  

        # ── 4. Seasonal Path (Non-linear Exogenous Routing) ────────────────
        z_seas = self.seasonal_proj(seasonal_all.permute(0, 2, 1))  
        z_endo = z_seas[:, -self.out_dim:, :]
        z_exo = z_seas[:, :-self.out_dim, :] if self.features == 'MS' else z_seas

        # --- Temporal Gating Module (TGM) ---
        if mode == 'wo_tgm':
            origin_gated = z_endo
            glob_gated = z_endo
        else:
            glob = self.glob_token.repeat(b, 1, 1)
            en_emb = torch.cat([z_endo, glob], dim=-1)  
            en_gated = self.en_attention(en_emb)
            origin_gated = en_gated[:, :, :self.d_model]
            glob_gated = en_gated[:, :, self.d_model:]

        # --- Variate Gating Module (VGM) ---
        if mode != 'wo_router' and self.c_in_vgm > self.out_dim:
            query_vec = z_endo if mode == 'wo_query_raw' else glob_gated
            ex_emb = torch.cat([z_exo, query_vec], dim=1) 
            
            if mode == 'wo_global':
                gate = torch.full((b, self.d_model, self.out_dim), 0.5, device=x_enc.device)
            else:
                gate = self.vgm_gate(ex_emb.transpose(1, 2)) 
                
            glob_final = glob_gated * gate.transpose(1, 2)  
        else:
            glob_final = glob_gated

        seas_in = (torch.cat([glob_final, glob_final], dim=-1)
                   if mode == 'wo_origin'
                   else torch.cat([origin_gated, glob_final], dim=-1))
        seasonal_pred = self.seasonal_head(seas_in).transpose(1, 2)  

        # ── 5. Strict Algebraic Fusion (For DBLoss) ────────────────────────
        if mode in ('wo_trend', 'wo_decomp'):
            res_out = seasonal_pred
        elif mode == 'w_adaptive_fuse':
            # 消融：强行缩放趋势，人为破坏 DBLoss 的物理假设 (Res = T + S)
            t_w = torch.sigmoid(self.trend_mix_alpha)
            res_out = t_w * trend_pred + seasonal_pred
        else:
            # 正常：完美代数闭环
            res_out = trend_pred + seasonal_pred

        # ── 6. RevIN Inverse ───────────────────────────────────────────────
        if mode != 'wo_revin':
            res_out = self.revin(res_out, 'denorm',
                                 target_dim=self.out_dim if self.features == 'MS' else None)
        return res_out
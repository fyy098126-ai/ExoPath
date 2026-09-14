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
# 2A. [Ablation: Prime5 旧版 SMA 分解] 用于证明 Exo-Guided EMA 的优越性
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
# 2B. [Prime7 新核心] Exo-Guided CC-EMA (外生指导条件分解)
# =========================================================================
class ExoGuidedCausalEMA(nn.Module):
    def __init__(self, c_in, out_dim, features='M'):
        super().__init__()
        self.out_dim = out_dim
        self.exo_dim = c_in - out_dim if features == 'MS' else 0
        
        self.base_alpha_logits = nn.Parameter(torch.zeros(out_dim))
        
        if self.exo_dim > 0:
            self.vol_sensitivity = nn.Parameter(torch.ones(1) * 2.0) 

    def forward(self, x):
        B, T, C = x.shape
        device = x.device

        # 动态捕捉外生事件波动，并使用 detach 作为静态先验保护 DBLoss
        if self.exo_dim > 0:
            x_exo = x[:, :, :-self.out_dim]
            exo_diff = torch.abs(x_exo[:, 1:, :] - x_exo[:, :-1, :])
            
            # 🚀【修复维度 Bug】：计算出 [B] 维度的 mean，然后 unsqueeze 为 [B, 1]
            exo_vol = exo_diff.mean(dim=[1, 2]) 
            exo_vol_prior = exo_vol.unsqueeze(1).detach() # -> [B, 1]
            
            # 此时 [1, out_dim] - [B, 1] -> [B, out_dim] (完美的 2D 矩阵！)
            alpha_logits = self.base_alpha_logits.unsqueeze(0) - self.vol_sensitivity * exo_vol_prior
        else:
            alpha_logits = self.base_alpha_logits.unsqueeze(0).repeat(B, 1)
            
        alpha = torch.sigmoid(alpha_logits).to(torch.double) # [B, out_dim]

        powers = torch.flip(torch.arange(T, dtype=torch.double, device=device), dims=(0,)).unsqueeze(1)
        weights = torch.pow((1.0 - alpha).unsqueeze(1), powers.unsqueeze(0)) 
        divisor = weights.clone()
        
        w_0 = weights[:, 0:1, :]
        w_rest = weights[:, 1:, :] * alpha.unsqueeze(1)
        weights_mod = torch.cat([w_0, w_rest], dim=1) 

        x_endo = x[:, :, -self.out_dim:].to(torch.double)
        out_endo = torch.cumsum(x_endo * weights_mod, dim=1) / divisor
        trend_endo = out_endo.to(torch.float32)

        # 外生通道保持静态 0.5 基础分解
        if self.exo_dim > 0:
            alpha_exo = torch.full((1, self.exo_dim), 0.5, dtype=torch.double, device=device)
            w_exo = torch.pow((1.0 - alpha_exo).unsqueeze(1), powers.unsqueeze(0))
            div_exo = w_exo.clone()
            w_0_exo = w_exo[:, 0:1, :]
            w_rest_exo = w_exo[:, 1:, :] * alpha_exo.unsqueeze(1)
            w_mod_exo = torch.cat([w_0_exo, w_rest_exo], dim=1)
            
            x_exo_d = x_exo.to(torch.double)
            out_exo = torch.cumsum(x_exo_d * w_mod_exo, dim=1) / div_exo
            trend_exo = out_exo.to(torch.float32)
            
            trend_all = torch.cat([trend_exo, trend_endo], dim=-1)
        else:
            trend_all = trend_endo

        seasonality = x - trend_all
        return trend_all, seasonality

# =========================================================================
# 3. LRER (Low-Rank Exogenous Router) - 低秩稀疏路由
# =========================================================================
class LRER(nn.Module):
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
# 4. SLTM (Symmetric Latent Trend Mixer) - 纯线性潜空间趋势外推
# =========================================================================
class SLTM(nn.Module):
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
# 5. XGate_Prime7 主模型 (完美集成 Prime5 消融逻辑)
# =========================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 获取消融模式 (兼容你所有写好的 .sh 脚本)
        self.ablation_mode = getattr(configs, 'ablation_mode', 'full')
        
        self.C = configs.enc_in
        self.features = getattr(configs, 'features', 'M')
        self.out_dim = 1 if self.features == 'MS' else self.C
        self.exo_dim = self.C - self.out_dim if self.features == 'MS' else 0

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        t_ff = getattr(configs, 't_ff', 0) or (self.d_model * 4)
        
        self.rank = getattr(configs, 'num_groups', 0) or (
            16 if self.C >= 321 else 8 if self.C >= 21 else 4
        )
        drop = getattr(configs, 'dropout', 0.1)

        # ── 1. RevIN + Decomp ─────────────────────────────────────────────
        self.revin = RevIN(self.C)
        self.decomp = ExoGuidedCausalEMA(c_in=self.C, out_dim=self.out_dim, features=self.features)
        
        # [Ablation: 旧版 SMA 兜底]
        if self.ablation_mode == 'w_sma_decomp':
            self.sma_decomp = DynamicEMADecomp(kernel_size=getattr(configs, 'ema_kernel', 25))

        # ── 2. PLE 趋势路径 ────────────────────────────────────────────────
        self.trend_proj = nn.Linear(self.seq_len, self.d_model)
        cross_gate_init = getattr(configs, 'cross_gate_init', -4.0)
        self.trend_mixer = SLTM(self.C, self.out_dim, self.d_model, rank=self.rank, dropout=drop, cross_gate_init=cross_gate_init)
        
        self.trend_norm = nn.LayerNorm(self.d_model)
        self.trend_head = nn.Linear(self.d_model, self.pred_len)
        
        # [Ablation: 用于证明纯线性外推重要性的非线性激活截断]
        self.ablation_trend_act = nn.GELU() if self.ablation_mode in ('w_trend_act', 'w_trend_pool_act', 'w_trend_pool2_act') else nn.Identity()

        # ── 3. 季节性路径 (EBT + LRER) ─────────────────────────────────────
        self.seasonal_proj = nn.Linear(self.seq_len, self.d_model)
        
        # EBT (Endogenous Blueprint) 取代了 TGM
        self.ebt_token = nn.Parameter(torch.ones(1, self.out_dim, self.d_model))
        self.endo_gate = nn.Sequential(
            nn.Linear(2 * self.d_model, t_ff), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(t_ff, 2 * self.d_model), nn.Sigmoid()
        )
        
        # LRER 取代了 VGM
        self.c_in_router = (self.C + self.out_dim) if self.features == 'M' else (self.exo_dim + self.out_dim)
        self.exo_router = LRER(self.c_in_router, self.out_dim, rank=self.rank, dropout=drop)
        
        self.seasonal_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(2 * self.d_model, self.pred_len),
        )

        # [Ablation: 用于证明 DBLoss 严格代数闭环重要性的自适应参数]
        if self.ablation_mode == 'w_adaptive_fuse':
            self.trend_mix_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        mode = self.ablation_mode
        b = x_enc.shape[0]

        # ── 1. RevIN ──────────────────────────────────────────────────────
        x = self.revin(x_enc, 'norm') if mode != 'wo_revin' else x_enc

        # ── 2. Decomposition ──────────────────────────────────────────────
        if mode == 'wo_decomp':
            seasonal_all = x
        elif mode == 'w_sma_decomp':  # [Ablation: Prime5 兼容] 证明 Exo-Guided 的优越
            trend_all, seasonal_all = self.sma_decomp(x)
        else:
            trend_all, seasonal_all = self.decomp(x)

        # ── 3. Trend Path ─────────────────────────────────────────────────
        if mode not in ('wo_trend', 'wo_decomp'):
            z_trend_all = self.trend_proj(trend_all.permute(0, 2, 1))  
            
            # [Ablation: 兼容所有的 w_trend 变体]
            if mode == 'wo_exo_trend':
                z_trend = z_trend_all[:, -self.out_dim:, :]
            elif mode in ('w_trend_pool', 'w_trend_pool_act', 'w_trend_pool2', 'w_trend_pool2_act'):
                global_trend = z_trend_all.mean(dim=1, keepdim=True)
                endo_trend = z_trend_all[:, -self.out_dim:, :]
                z_trend = endo_trend + global_trend
                z_trend = self.trend_norm(z_trend)
                if mode in ('w_trend_pool2', 'w_trend_pool2_act'):
                    z_trend = z_trend + z_trend.mean(dim=1, keepdim=True)
                    z_trend = self.trend_norm(z_trend)
            elif mode == 'w_trend_norm':
                z_trend = self.trend_mixer(z_trend_all)
                z_trend = self.trend_norm(z_trend)
            elif mode != 'wo_trend_mix':
                z_trend = self.trend_mixer(z_trend_all)
            else:
                z_trend = z_trend_all[:, -self.out_dim:, :]
                
            # [Ablation: 强行加入 GELU，用于论证纯线性的必要性]
            z_trend = self.ablation_trend_act(z_trend)
            trend_pred = self.trend_head(z_trend).transpose(1, 2)  

        # ── 4. Seasonal Path ──────────────────────────────────────────────
        z_seas = self.seasonal_proj(seasonal_all.permute(0, 2, 1))  
        z_endo = z_seas[:, -self.out_dim:, :]
        z_exo = z_seas[:, :-self.out_dim, :] if self.features == 'MS' else z_seas

        # --- EBT (内生蓝图提取) ---
        # [Ablation: Prime5 兼容，将旧版 wo_tgm 映射到禁用新版 EBT]
        if mode == 'wo_tgm': 
            origin_gated = z_endo
            ebt_gated = z_endo
        else:
            glob = self.ebt_token.repeat(b, 1, 1)
            en_emb = torch.cat([z_endo, glob], dim=-1)  
            en_gated = en_emb * self.endo_gate(en_emb)
            origin_gated = en_gated[:, :, :self.d_model]
            ebt_gated = en_gated[:, :, self.d_model:]

        # --- LRER (低秩稀疏外生路由) ---
        if mode != 'wo_router' and self.c_in_router > self.out_dim:
            # [Ablation: 旧版 wo_query_raw，用原始内生信号替代蓝图去路由]
            query_vec = z_endo if mode == 'wo_query_raw' else ebt_gated
            ex_emb = torch.cat([z_exo, query_vec], dim=1) 
            
            # [Ablation: 强行均匀融合 (Uniform Fusion)]
            if mode == 'wo_global':
                gate = torch.full((b, self.d_model, self.out_dim), 0.5, device=x_enc.device)
            else:
                gate = self.exo_router(ex_emb.transpose(1, 2)) 
                
            exo_driven_final = ebt_gated * gate.transpose(1, 2)  
        else:
            exo_driven_final = ebt_gated

        # [Ablation: 兼容 wo_origin]
        seas_in = (torch.cat([exo_driven_final, exo_driven_final], dim=-1)
                   if mode == 'wo_origin'
                   else torch.cat([origin_gated, exo_driven_final], dim=-1))
        
        seasonal_pred = self.seasonal_head(seas_in).transpose(1, 2)  

        # ── 5. Fusion ─────────────────────────────────────────────────────
        if mode in ('wo_trend', 'wo_decomp'):
            res_out = seasonal_pred
        elif mode == 'w_adaptive_fuse':
            # [Ablation: 人为破坏 DBLoss 代数闭环]
            t_w = torch.sigmoid(self.trend_mix_alpha)
            res_out = t_w * trend_pred + seasonal_pred
        else:
            # 正常：Y = T + S，与 DBLoss 物理假定同构
            res_out = trend_pred + seasonal_pred

        # ── 6. RevIN Inverse ──────────────────────────────────────────────
        if mode != 'wo_revin':
            res_out = self.revin(res_out, 'denorm', target_dim=self.out_dim if self.features == 'MS' else None)
        return res_out
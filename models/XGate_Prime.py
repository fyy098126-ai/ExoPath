import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. 基础组件: RevIN (解决分布偏移，保留)
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
# 2. 动态 SMA 分解
# =========================================================================
class DynamicSMADecomp(nn.Module):
    def __init__(self, kernel_size=25, ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.scales = [kernel_size // 2, kernel_size, kernel_size * 2]
        self.scale_weights = nn.Parameter(torch.zeros(len(self.scales)))

    def forward(self, x):
        if self.ablation_mode == 'wo_decomp':
            return torch.zeros_like(x), x
            
        B, L, C = x.shape
        trends = []
        norm_weights = F.softmax(self.scale_weights, dim=0)
        
        for k in self.scales:
            k = k if k % 2 != 0 else k + 1 
            padding = k // 2
            x_pad = F.pad(x.permute(0, 2, 1), (padding, padding), mode='replicate')
            trend_k = F.avg_pool1d(x_pad, kernel_size=k, stride=1)
            trends.append(trend_k.permute(0, 2, 1))
            
        final_trend = sum([t * norm_weights[i] for i, t in enumerate(trends)])
        return final_trend, x - final_trend

# =========================================================================
# 3. 趋势项 Cross-Embedding (CrossLinear 思想)
# =========================================================================
class CrossLinearEmbedding(nn.Module):
    def __init__(self, ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode

    def forward(self, trend_endo, trend_exo, features_mode):
        if trend_exo is None or self.ablation_mode == 'wo_cross_emb':
            return trend_endo
            
        endo_norm = F.normalize(trend_endo, p=2, dim=-1)
        exo_norm = F.normalize(trend_exo, p=2, dim=-1)
        cross_corr = torch.bmm(endo_norm, exo_norm.transpose(1, 2))
        
        # M 模式下，防自指对角线掩码
        if features_mode == 'M' and trend_endo.shape[1] == trend_exo.shape[1]:
            mask = 1.0 - torch.eye(trend_endo.shape[1], device=trend_endo.device).unsqueeze(0)
            cross_corr = cross_corr * mask
            
        corr_weights = F.softmax(cross_corr, dim=-1)
        exo_driven_trend = torch.bmm(corr_weights, trend_exo)
        
        return trend_endo + exo_driven_trend

# =========================================================================
# 4. 核心组件：Gating Block (完全基于 XLinear 源码)
# =========================================================================
class Gating_Block(nn.Module):
    def __init__(self, d_in, hf, dropout=0.1):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(d_in, hf),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hf, d_in),
            nn.Sigmoid() # 软门控
        )
        
    def forward(self, x):
        return x * self.weight(x)

# =========================================================================
# 5. 季节项 XLinear Router (内生提取 + 全局路由)
# =========================================================================
class Seasonal_XLinear_Router(nn.Module):
    def __init__(self, c_target, c_exo, d_eff, t_ff, c_ff, dropout=0.1):
        super().__init__()
        self.c_target = c_target
        self.c_exo = c_exo
        self.d_eff = d_eff
        
        # 🎯 初始化 Global Token (依据 XLinear，使用 ones)
        self.glob_token = nn.Parameter(torch.ones([1, c_target, d_eff]))
        
        # TGM (内生时域门控)：处理展平特征，2倍 d_eff 拼接
        self.en_attention = Gating_Block(2 * d_eff, t_ff, dropout)
        
        # VGM (跨变量路由)：沿着通道维度做门控
        self.ex_attention = Gating_Block(c_exo + c_target, c_ff, dropout)

    def forward(self, z_endo, z_exo, ablation_mode):
        b = z_endo.shape[0]
        
        # 1. TGM (内生门控提取)
        glob = self.glob_token.repeat(b, 1, 1) # [b, c_t, d_eff]
        en_emb = torch.cat([z_endo, glob], dim=-1) # [b, c_t, 2*d_eff]
        
        # 过门控，内生特征不仅被提取去噪，也赋予了 Global Token 当前上下文
        en_gated = self.en_attention(en_emb) # [b, c_t, 2*d_eff]
        
        origin_gated = en_gated[:, :, :self.d_eff] # 🎯 被门控提取过的纯净内生特征
        glob_gated = en_gated[:, :, self.d_eff:]   # 更新后的 Global Token
        
        # 2. VGM (外生因果发现与过滤)
        if z_exo is not None and ablation_mode != 'wo_router':
            # 拼接通道: [b, c_exo + c_t, d_eff]
            ex_emb = torch.cat([z_exo, glob_gated], dim=1) 
            
            # 🎯 Permute：在通道维度上进行门控，找出哪些外生变量真正驱动了内生变量
            ex_gated = self.ex_attention(ex_emb.permute(0, 2, 1)) # [b, d_eff, c_exo + c_t]
            
            # 取出吸收了外生突变信息的最终 Global Token
            glob_final = ex_gated[:, :, self.c_exo:].permute(0, 2, 1) # [b, c_t, d_eff]
        else:
            glob_final = glob_gated

        if ablation_mode == 'wo_global':
            glob_final = torch.zeros_like(glob_final)
            
        return origin_gated, glob_final

# =========================================================================
# 6. Top Level Model (极致轻量与纯净的大一统架构)
# =========================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.ablation_mode = getattr(configs, 'ablation_mode', 'full')
        self.C = configs.enc_in
        self.features = getattr(configs, 'features', 'M')
        
        self.out_dim = 1 if self.features == 'MS' else self.C
        self.exo_dim = self.C - self.out_dim if self.features == 'MS' else self.C
        
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.d_model = configs.d_model
        
        # Patch 展平后的总时域感受野长度
        self.d_eff = self.patch_num * self.d_model
        
        # --- 物理基础 ---
        self.revin = RevIN(self.C)
        self.decomp = DynamicEMADecomp(kernel_size=getattr(configs, 'ema_kernel', 25), ablation_mode=self.ablation_mode)
        
        # ===============================================
        # 🎯 Trend 轨道 (交叉嵌入 -> Patch -> MLP 提取)
        # ===============================================
        self.cross_emb = CrossLinearEmbedding(self.ablation_mode)
        self.trend_projection = nn.Sequential(
            nn.Linear(self.patch_len, self.d_model),
            nn.Dropout(configs.dropout)
        )
        # Trend 的 MLP 特征提取器
        self.trend_mlp = nn.Sequential(
            nn.Linear(self.d_eff, self.d_eff * 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.d_eff * 2, self.d_eff)
        )
        self.trend_head = nn.Linear(self.d_eff, configs.pred_len)

        # ===============================================
        # 🎯 Seasonal 轨道 (Patch -> XLinear 路由 -> 联合预测)
        # ===============================================
        self.seasonal_projection = nn.Sequential(
            nn.Linear(self.patch_len, self.d_model),
            nn.Dropout(configs.dropout)
        )
        
        # 完全抛弃 CausalPatchMLPBlock，用轻量的 XLinearRouter 代替
        t_ff = getattr(configs, 't_ff', self.d_eff * 4) # 默认内部膨胀系数为2 (2*d_eff -> 4*d_eff)
        c_ff = getattr(configs, 'c_ff', (self.exo_dim + self.out_dim) * 2) 
        
        self.router = Seasonal_XLinear_Router(
            c_target=self.out_dim, 
            c_exo=self.exo_dim, 
            d_eff=self.d_eff,
            t_ff=t_ff,
            c_ff=c_ff,
            dropout=configs.dropout
        )
        
        # 🎯 最终预测头：内生门控特征 + Global Token 拼接 (2 * d_eff)
        self.seasonal_head = nn.Sequential(
            nn.Dropout(configs.dropout),
            nn.Linear(2 * self.d_eff, configs.pred_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, C = x_enc.shape
        
        if self.ablation_mode != 'wo_revin':
            x_norm = self.revin(x_enc, mode='norm')
        else:
            x_norm = x_enc
        
        trend_all, seasonal_all = self.decomp(x_norm)
        
        trend_endo = trend_all[:, :, -self.out_dim:].transpose(1, 2)
        trend_exo = trend_all[:, :, :-self.out_dim].transpose(1, 2) if self.features == 'MS' else trend_all.transpose(1, 2)
        
        # ===============================================
        # 1. 趋势轨道前向传播
        # ===============================================
        # 跨域力学融合
        fused_trend_seq = self.cross_emb(trend_endo, trend_exo, self.features)
        
        pad_len = self.patch_len + self.stride * (self.patch_num - 1) - L
        trend_pad = F.pad(fused_trend_seq, (0, pad_len), mode='replicate') if pad_len > 0 else fused_trend_seq
        
        # Patch & Project & Flatten
        z_trend_patches = self.trend_projection(trend_pad.unfold(2, self.patch_len, self.stride)) # [B, C_t, P, D]
        z_trend_flat = z_trend_patches.reshape(B, self.out_dim, self.d_eff)
        
        # 🎯 MLP 特征提取
        z_trend_ext = self.trend_mlp(z_trend_flat)
        trend_pred = self.trend_head(z_trend_ext).transpose(1, 2) # [B, Pred_len, C_out]

        # ===============================================
        # 2. 季节轨道前向传播
        # ===============================================
        seasonal_pad = F.pad(seasonal_all.permute(0, 2, 1), (0, pad_len), mode='replicate').permute(0, 2, 1) if pad_len > 0 else seasonal_all
        
        # Patch & Project & Flatten (不再穿过深层 CausalPatchMLPBlock)
        z_seasonal_patches = self.seasonal_projection(seasonal_pad.permute(0, 2, 1).unfold(2, self.patch_len, self.stride))
        z_seasonal_flat = z_seasonal_patches.reshape(B, self.C, self.d_eff)
        
        z_endo = z_seasonal_flat[:, -self.out_dim:, :]
        z_exo = z_seasonal_flat[:, :-self.out_dim, :] if self.features == 'MS' else z_seasonal_flat
        
        # 🎯 XLinear 极简路由：获取提取后的内生特征 & 更新后的全局枢纽
        origin_gated, glob_final = self.router(z_endo, z_exo, self.ablation_mode)
        
        # 🎯 联合预测：拼接门控内生特征和 Global Token
        # [B, C_out, d_eff] + [B, C_out, d_eff] -> [B, C_out, 2 * d_eff]
        en_final = torch.cat([origin_gated, glob_final], dim=-1) 
        
        seasonal_pred = self.seasonal_head(en_final).transpose(1, 2)
        
        # ===============================================
        # 3. 终极合流与反归一化
        # ===============================================
        res_out = seasonal_pred + trend_pred
        if self.ablation_mode != 'wo_revin':
            return self.revin(res_out, mode='denorm', target_dim=self.out_dim)
        return res_out
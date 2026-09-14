import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================================
# 1. 基础组件：RevIN (处理非平稳性)
# =========================================================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight + self.affine_bias
        else:
            if self.affine: x = (x - self.affine_bias) / self.affine_weight
            x = x * self.stdev + self.mean
        return x

# =========================================================================
# 2. 核心模块：Advanced Fusion Module (Ablation-Ready)
# =========================================================================
class Advanced_Fusion(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.use_vgm = configs.use_vgm
        self.use_led = configs.use_led
        self.use_multiscale = configs.use_multiscale
        d_model = configs.d_model
        patch_num = int(np.ceil(configs.seq_len / configs.patch_len))

        # VGM 空间过滤
        if self.use_vgm:
            self.vgm = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
                nn.Sigmoid()
            )

        # 滞后捕捉 (基于 CrossLinear 的 1D 卷积思想)
        if self.use_multiscale:
            self.lag_convs = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2) for k in [1, 3, 5]
            ])
            self.conv_fusion = nn.Linear(d_model * 3, d_model)
        else:
            self.lag_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            self.conv_fusion = nn.Linear(d_model, d_model)

        # 环境字典 LED
        if self.use_led:
            self.env_dict = nn.Parameter(torch.randn(configs.num_states, d_model))
            self.query_proj = nn.Linear(d_model, d_model)
            self.temporal_align = nn.AdaptiveAvgPool1d(patch_num)

        # 动态调制生成器
        self.alpha = nn.Parameter(torch.ones(1) * 0.8)
        self.gamma_gen = nn.Linear(d_model, d_model)
        self.beta_gen  = nn.Linear(d_model, d_model)

    def forward(self, z_endo, x_exog_feat):
        # z_endo: [B_N, P, D] | x_exog_feat: [B, T, D]
        B_N, P, D = z_endo.shape
        B, T, _ = x_exog_feat.shape
        num_vars = B_N // B

        # 1. 空间去噪
        if self.use_vgm:
            gate = self.vgm(x_exog_feat.mean(dim=1, keepdim=True))
            x_exog_feat = x_exog_feat * gate

        # 2. 滞后特征提取与维度对齐
        x_in = x_exog_feat.transpose(1, 2) # [B, D, T]
        if self.use_multiscale:
            lag_feats = [conv(x_in) for conv in self.lag_convs] # List of [B, D, T]
            combined = torch.cat(lag_feats, dim=1) # [B, 3*D, T]
            lagged_exog = self.conv_fusion(combined.transpose(1, 2)) # [B, T, D]
        else:
            lagged_exog = self.conv_fusion(self.lag_conv(x_in).transpose(1, 2)) # [B, T, D]

        # 3. 字典查询与时间轴对齐
        if self.use_led:
            exog_p = self.temporal_align(lagged_exog.transpose(1, 2)).transpose(1, 2) # [B, P, D]
            queries = self.query_proj(exog_p)
            attn = F.softmax(torch.matmul(queries, self.env_dict.T) / np.sqrt(D), dim=-1)
            exog_rep = torch.matmul(attn, self.env_dict) # [B, P, D]
        else:
            exog_rep = F.adaptive_avg_pool1d(lagged_exog.transpose(1, 2), P).transpose(1, 2)

        # 4. 特征注入
        exog_rep = exog_rep.repeat_interleave(num_vars, dim=0)
        gamma = self.gamma_gen(exog_rep)
        beta = self.beta_gen(exog_rep)
        
        z_fused = (z_endo * gamma) + beta
        # 残差逻辑: z = α * endo + (1-α) * fused
        return self.alpha * z_endo + (1 - self.alpha) * z_fused

# =========================================================================
# 3. 完整模型主体 (PA_LED)
# =========================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.features = configs.features
        self.patch_len = configs.patch_len
        self.patch_num = int(np.ceil(configs.seq_len / configs.patch_len))
        
        # 数据处理器
        self.revin = RevIN(configs.enc_in) if getattr(configs, 'use_revin', True) else None
        
        # 维度映射
        exog_dim = configs.enc_in - 1 if configs.features == 'MS' else configs.enc_in
        self.exog_proj = nn.Linear(exog_dim, configs.d_model)
        self.value_embedding = nn.Linear(self.patch_len, configs.d_model)
        
        # 挂载高级融合模块
        self.fusion = Advanced_Fusion(configs)
        
        # Backbone: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model, nhead=configs.n_heads,
            dim_feedforward=configs.d_ff, dropout=configs.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)
        
        # 预测头
        self.c_out = configs.c_out if configs.features == 'M' else 1
        self.head = nn.Linear(configs.d_model * self.patch_num, configs.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B = x_enc.shape[0]
        if self.revin: x_enc = self.revin(x_enc, mode='norm')
            
        # 模式兼容逻辑 (M / MS)
        if self.features == 'MS':
            x_target = x_enc[:, :, -1:] # 最后一列是 endogenous
            x_exog = x_enc[:, :, :-1]   # 其余是 exogenous
        else:
            x_target = x_exog = x_enc   # M 模式下全量交互
            
        # Patching & Embedding
        z = x_target.transpose(1, 2).unfold(-1, self.patch_len, self.patch_len)
        C_t = z.shape[1]
        z_endo = self.value_embedding(z.reshape(-1, self.patch_num, self.patch_len))
        
        # 外生特征与融合
        x_exog_feat = self.exog_proj(x_exog)
        z_fused = self.fusion(z_endo, x_exog_feat)
        
        # 骨干网络计算
        z_out = self.encoder(z_fused)
        
        # 展平输出
        z_out = z_out.reshape(B, C_t, -1)
        res = self.head(z_out).reshape(B, C_t, -1).transpose(1, 2)
        
        if self.revin: res = self.revin(res, mode='denorm')
        return res
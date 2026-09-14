import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class VariableGatedProjection_V3(nn.Module):
    """
    PA_LED_V3 核心改进：三维统计指纹门控 + 残差放大器
    """
    def __init__(self, enc_in, d_model, configs):
        super().__init__()
        # 输入维度为 enc_in * 3 (Mean, Std, Max)
        mid_dim = max(d_model // 2, enc_in * 2)
        
        self.var_gate = nn.Sequential(
            nn.Linear(enc_in * 3, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, enc_in)
        )
        # 初始化为 0，Sigmoid(0)=0.5，配合残差初始状态温和
        self.var_gate[-1].bias.data.fill_(0.0) 
        
        self.proj = nn.Sequential(
            nn.Linear(enc_in, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, d_model)
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_exog):
        B, T, N = x_exog.shape
        
        # 1. 提取三维特征指纹 (绕过 RevIN 的 Mean=0 陷阱)
        v_mean = x_exog.mean(dim=1)
        v_std = x_exog.std(dim=1)
        v_max, _ = x_exog.max(dim=1)
        v_stats = torch.cat([v_mean, v_std, v_max], dim=-1) # [B, N*3]
        
        # 2. 计算门控权重 (引入温度系数 0.5 强制拉开差距)
        gate_logits = self.var_gate(v_stats)
        v_weights = torch.sigmoid(gate_logits / 0.5) 
        
        # 3. 残差重加权策略：x = x * (1 + weights) 
        # 既保留了原始信息，又实现了显著的特征放大
        x_exog = x_exog * (1 + v_weights.unsqueeze(1))
            
        out = self.proj(x_exog) 
        return self.dropout(out), v_weights

class Advanced_Fusion_V3(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        d_model = configs.d_model
        patch_num = int(np.ceil(configs.seq_len / configs.patch_len))
        
        # --- 原有模块初始化 ---
        self.lag_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2) for k in [1, 3, 5]
        ])
        self.conv_fusion = nn.Linear(d_model * 3, d_model)
        self.env_dict = nn.Parameter(torch.randn(configs.num_states, d_model))
        self.query_proj = nn.Linear(d_model, d_model)
        self.temporal_align = nn.AdaptiveAvgPool1d(patch_num)
        self.gamma_gen = nn.Linear(d_model, d_model)
        self.beta_gen  = nn.Linear(d_model, d_model)

        # --- 【核心修改】：自适应先验初始化 ---
        # 从 configs 获取物理比例 (例如 ETTh1 传 0.8, Weather 传 0.4)
        prob = getattr(configs, 'initial_alpha', 0.5) 
        # 将比例映射回 Sigmoid 前的数值区间 (Logit Transform)
        # 公式: logit(p) = ln(p / (1-p))
        prob = np.clip(prob, 1e-4, 1 - 1e-4) # 边界保护
        initial_logit = np.log(prob / (1 - prob))
        
        # 初始化参数，这样第一个 epoch 训练前 sigmoid(alpha) 刚好等于 prob
        self.alpha = nn.Parameter(torch.ones(1) * initial_logit)
        print(f"🚀 [PA_LED_V3] Alpha initialized with physical prior: {prob} (logit: {initial_logit:.4f})")

    def forward(self, z_base, x_exog_feat):
        B_N, P, D = z_base.shape
        num_vars = B_N // x_exog_feat.shape[0]

        # 1. 滞后特征与时间对齐
        x_in = x_exog_feat.transpose(1, 2)
        lagged = self.conv_fusion(torch.cat([c(x_in) for c in self.lag_convs], dim=1).transpose(1, 2))
        exog_p = self.temporal_align(lagged.transpose(1, 2)).transpose(1, 2)
        
        # 2. 字典查询
        queries = self.query_proj(exog_p)
        attn = F.softmax(torch.matmul(queries, self.env_dict.T) / np.sqrt(D), dim=-1)
        exog_patch = torch.matmul(attn, self.env_dict)

        # 3. 动态调制注入
        exog_rep = exog_patch.repeat_interleave(num_vars, dim=0)
        z_fused = (z_base * torch.tanh(self.gamma_gen(exog_rep))) + self.beta_gen(exog_rep)

        # 4. 【关键】：使用约束后的 Alpha 进行加权融合
        a = torch.sigmoid(self.alpha)
        return a * z_base + (1 - a) * z_fused

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.patch_len = configs.patch_len
        self.patch_num = int(np.ceil(configs.seq_len / configs.patch_len))

        self.revin = RevIN(configs.enc_in)
        self.exog_proj = VariableGatedProjection_V3(configs.enc_in, configs.d_model, configs)
        self.value_embedding = nn.Linear(self.patch_len, configs.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model, nhead=configs.n_heads,
            dim_feedforward=configs.d_ff, dropout=configs.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)
        self.fusion = Advanced_Fusion_V3(configs)
        
        self.c_out = configs.enc_in # M模式
        self.head = nn.Linear(configs.d_model * self.patch_num, configs.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B = x_enc.shape[0]
        x_enc = self.revin(x_enc, mode='norm')
        
        # 1. 内生路径
        z = x_enc.transpose(1, 2).unfold(-1, self.patch_len, self.patch_len)
        z_endo = self.value_embedding(z.reshape(-1, self.patch_num, self.patch_len))
        
        # 2. 外生路径 (V3门控)
        x_exog_feat, self.v_weights = self.exog_proj(x_enc)

        # 3. 后注入融合
        z_feat = self.encoder(z_endo)
        z_out = self.fusion(z_feat, x_exog_feat)
        
        res = self.head(z_out.reshape(B, self.c_out, -1)).reshape(B, self.c_out, -1).transpose(1, 2)
        res = self.revin(res, mode='denorm')
        return res
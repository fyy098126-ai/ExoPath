import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np



# =========================================================================

# 1. RevIN: 归一化组件 (解决分布漂移问题)

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

# 2. VariableGatedProjection: 物理变量权重选择 (对标 XLinear 门控思想)

# =========================================================================

class VariableGatedProjection(nn.Module):

    def __init__(self, enc_in, d_model, configs):

        super().__init__()

        self.use_var_gate = getattr(configs, 'use_var_gate', 1)

        # 即使对于 ETT 小数据集，mid_dim 也能提供非线性表达能力

        mid_dim = max(d_model, enc_in * 2)

       

        if self.use_var_gate:

            self.var_gate = nn.Sequential(

                nn.Linear(enc_in, mid_dim),

                nn.GELU(),

                nn.Linear(mid_dim, enc_in),

                nn.Sigmoid()

            )

            self.var_gate[-2].bias.data.fill_(2.0) # 初始保持高通

       

        self.proj = nn.Sequential(

            nn.Linear(enc_in, mid_dim),

            nn.GELU(),

            nn.Linear(mid_dim, d_model)

        )

        self.dropout = nn.Dropout(configs.dropout)



    def forward(self, x_exog):

        B, T, N = x_exog.shape

        if self.use_var_gate:

            v_stats = x_exog.mean(dim=1)

            v_weights = self.var_gate(v_stats)

            x_exog = x_exog * v_weights.unsqueeze(1) # 变量级点乘

        else:

            v_weights = torch.ones(B, N).to(x_exog.device)

           

        out = self.proj(x_exog)

        return self.dropout(out), v_weights



# =========================================================================

# 3. Advanced_Fusion: 后注入融合核心 (LED 字典 + FiLM 调制)

# =========================================================================

class Advanced_Fusion(nn.Module):

    def __init__(self, configs):

        super().__init__()

        self.configs = configs

        d_model = configs.d_model

        patch_num = int(np.ceil(configs.seq_len / configs.patch_len))

       

        # 模块消融开关

        self.use_led = getattr(configs, 'use_led', 1)

        self.use_multiscale = getattr(configs, 'use_multiscale', 1)



        # 1D 卷积提取时间滞后惯性

        if self.use_multiscale:

            self.lag_convs = nn.ModuleList([

                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2) for k in [1, 3, 5]

            ])

            self.conv_fusion = nn.Linear(d_model * 3, d_model)

        else:

            self.lag_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

            self.conv_fusion = nn.Linear(d_model, d_model)



        # 环境状态字典 LED

        if self.use_led:

            self.env_dict = nn.Parameter(torch.randn(configs.num_states, d_model))

            self.query_proj = nn.Linear(d_model, d_model)

       

        self.temporal_align = nn.AdaptiveAvgPool1d(patch_num)

        self.alpha = nn.Parameter(torch.ones(1) * 0.8)

        self.gamma_gen = nn.Linear(d_model, d_model)

        self.beta_gen  = nn.Linear(d_model, d_model)



    def forward(self, z_base, x_exog_feat):

        B_N, P, D = z_base.shape

        num_vars = B_N // x_exog_feat.shape[0]



        # 滞后特征提取

        x_in = x_exog_feat.transpose(1, 2)

        if self.use_multiscale:

            lagged = self.conv_fusion(torch.cat([c(x_in) for c in self.lag_convs], dim=1).transpose(1, 2))

        else:

            lagged = self.conv_fusion(self.lag_conv(x_in).transpose(1, 2))



        # 字典补丁查询

        exog_p = self.temporal_align(lagged.transpose(1, 2)).transpose(1, 2)

        if self.use_led:

            queries = self.query_proj(exog_p)

            #修改：温度缩放 (Temperature Scaling)

            # tau = 2.0 # 增加探索性

            # attn = F.softmax(torch.matmul(queries, self.env_dict.T) / (np.sqrt(D) * tau), dim=-1)

            attn = F.softmax(torch.matmul(queries, self.env_dict.T) / np.sqrt(D), dim=-1)

            exog_patch = torch.matmul(attn, self.env_dict)

        else:

            exog_patch = exog_p



        # 动态调制注入

        exog_rep = exog_patch.repeat_interleave(num_vars, dim=0)

        z_fused = (z_base * self.gamma_gen(exog_rep)) + self.beta_gen(exog_rep)

        return self.alpha * z_base + (1 - self.alpha) * z_fused



# =========================================================================

# 4. PA_LED2.0 主模型

# =========================================================================

class Model(nn.Module):

    def __init__(self, configs):

        super(Model, self).__init__()

        self.configs = configs

        self.patch_len = configs.patch_len

        self.patch_num = int(np.ceil(configs.seq_len / configs.patch_len))

        self.fusion_pos = getattr(configs, 'fusion_pos', 'late')



        self.revin = RevIN(configs.enc_in) if getattr(configs, 'use_revin', 1) else None

       

        # 物理级变量门控层

        exog_dim = configs.enc_in - 1 if configs.features == 'MS' else configs.enc_in

        self.exog_proj = VariableGatedProjection(exog_dim, configs.d_model, configs)

        self.value_embedding = nn.Linear(self.patch_len, configs.d_model)

       

        # Transformer 骨干网络

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=configs.d_model, nhead=configs.n_heads,

            dim_feedforward=configs.d_ff, dropout=configs.dropout, batch_first=True

        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)

        self.fusion = Advanced_Fusion(configs)

       

        self.c_out = configs.c_out if configs.features == 'M' else 1

        self.head = nn.Linear(configs.d_model * self.patch_num, configs.pred_len * self.c_out)



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        B = x_enc.shape[0]

        if self.revin: x_enc = self.revin(x_enc, mode='norm')

           

        if self.configs.features == 'MS':

            x_target, x_exog = x_enc[:, :, -1:], x_enc[:, :, :-1]

        else:

            x_target = x_exog = x_enc  

           

        # 1. 内生嵌入 (Patching)

        z = x_target.transpose(1, 2).unfold(-1, self.patch_len, self.patch_len)

        C_t = z.shape[1]

        z_endo = self.value_embedding(z.reshape(-1, self.patch_num, self.patch_len))

       

        # 2. 外生门控特征处理

        x_exog_feat, self.v_weights = self.exog_proj(x_exog)



        # 3. 核心流控 (Late Injection 逻辑)

        if self.fusion_pos == 'early':

            z_in = self.fusion(z_endo, x_exog_feat)

            z_out = self.encoder(z_in)

        else:

            z_feat = self.encoder(z_endo) # 先过 Transformer 筑基

            z_out = self.fusion(z_feat, x_exog_feat) # 后注入环境补丁

       

        res = self.head(z_out.reshape(B, C_t, -1)).reshape(B, C_t, -1).transpose(1, 2)

        if self.revin: res = self.revin(res, mode='denorm')

        return res
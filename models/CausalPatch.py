import torch

import torch.nn as nn

import torch.nn.functional as F



# ----------------- 1. RevIN (平稳化基座) -----------------

class RevIN(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):

        super().__init__()

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

            if x.shape[-1] != self.num_features:

                m, s = self.mean[:, :, -1:], self.stdev[:, :, -1:]

                x = x * s + m if not self.affine else ((x - self.affine_bias[-1:]) / self.affine_weight[-1:]) * s + m

            else:

                x = x * self.stdev + self.mean if not self.affine else ((x - self.affine_bias) / self.affine_weight) * self.stdev + self.mean

        return x



# ----------------- 2. 模块一: HTA (硬物理时滞锁定) -----------------

class HardTemporalAlignment(nn.Module):

    def __init__(self, seq_dim, Nx, max_lag):

        super().__init__()

        self.max_lag = max_lag

        # 为每个外生变量学习一个时滞偏好打分器

        self.lag_scorer = nn.Linear(seq_dim, 1)



    def forward(self, exog):

        """ exog: [B, Nx, T] """

        B, Nx, T = exog.shape

        actual_max_lag = min(self.max_lag, T - 1)

       

        # 构建时滞候选池: [B, Nx, Lags, T]

        shifted_exogs = [exog]

        for lag in range(1, actual_max_lag + 1):

            shifted = F.pad(exog[:, :, :-lag], (lag, 0), mode='replicate')

            shifted_exogs.append(shifted)

        E_candidates = torch.stack(shifted_exogs, dim=2)

       

        # 评估每个时滞的显著性

        scores = self.lag_scorer(E_candidates).squeeze(-1) # [B, Nx, Lags]

       

        # 🔥 Gumbel-Softmax (Hard=True) 🔥

        # 前向传播时是绝对的 One-hot (只选1个最强时滞)，反向传播时保留梯度

        if self.training:

            lag_weights = F.gumbel_softmax(scores, tau=1.0, hard=True, dim=-1)

        else:

            lag_idx = torch.argmax(scores, dim=-1, keepdim=True)

            lag_weights = torch.zeros_like(scores).scatter_(-1, lag_idx, 1.0)

           

        # 记录选中的时滞索引，用于论文可解释性分析

        self.selected_lags = torch.argmax(lag_weights, dim=-1) # [B, Nx]

       

        # 提取硬对齐后的特征 [B, Nx, T]

        aligned_exog = torch.einsum('bxl, bxlt -> bxt', lag_weights, E_candidates)

        return aligned_exog



# ----------------- 3. 模块二: AEM (非对称事件触发调制) -----------------

class AsymmetricEventModulator(nn.Module):

    def __init__(self, d_model, Nx, Ne):

        super().__init__()

        # 1. 独立事件能量评估器

        self.energy_scorer = nn.Sequential(

            nn.Linear(d_model, d_model // 4), nn.GELU(),

            nn.Linear(d_model // 4, 1)

        )

       

        # 2. 强力防御机制：可学习的激活阈值 (初始化为1.5，默认拦截绝大多数正常波动)

        self.learned_threshold = nn.Parameter(torch.ones(Nx) * 1.5)

       

        # 3. 跨变量映射 (XLinear风格)

        self.cross_linear = nn.Linear(Nx, Ne)

       

        # 4. 生成调制参数 (FiLM: gamma, beta)

        self.film_gen = nn.Sequential(

            nn.Linear(d_model, d_model), nn.GELU(),

            nn.Linear(d_model, d_model * 2)

        )

       

        # 🔥 绝对防御：零初始化 🔥

        # 确保开局输出的 gamma=0, beta=0，模型完美等价于纯 MLP

        nn.init.zeros_(self.film_gen[2].weight)

        nn.init.zeros_(self.film_gen[2].bias)



    def forward(self, z_exog):

        B, Nx, P, D = z_exog.shape

       

        # 计算局部事件能量

        energy = self.energy_scorer(z_exog.reshape(-1, D)).reshape(B, Nx, P)

       

        # 硬阈值截断 -> 挡住无用噪声

        excess_energy = F.relu(energy - self.learned_threshold.view(1, Nx, 1))

        self.gate = torch.tanh(excess_energy).unsqueeze(-1) # 记录下来用于论文作图

       

        # 稀疏过滤与单向跨变量映射

        filtered_exog = z_exog * self.gate

        mapped_exog = self.cross_linear(filtered_exog.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

       

        # 生成调制参数

        mod_params = self.film_gen(mapped_exog) # [B, Ne, P, 2*D]

        gamma, beta = torch.chunk(mod_params, 2, dim=-1)

        return gamma, beta



# ----------------- 4. Patch-MLP 骨干 (致敬 Paper 69 & 0) -----------------

class MLPBlock(nn.Module):

    def __init__(self, patch_num, d_model, dropout=0.1):

        super().__init__()

        self.time_mix = nn.Sequential(

            nn.Linear(patch_num, patch_num * 2), nn.GELU(),

            nn.Dropout(dropout), nn.Linear(patch_num * 2, patch_num)

        )

        self.feat_mix = nn.Sequential(

            nn.Linear(d_model, d_model * 2), nn.GELU(),

            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)

        )

        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)



    def forward(self, x):

        x = x + self.time_mix(x.transpose(1, 2)).transpose(1, 2)

        x = self.norm1(x)

        x = x + self.feat_mix(x)

        x = self.norm2(x)

        return x



# ----------------- 5. 顶层模型集成 -----------------

class Model(nn.Module):

    def __init__(self, configs):

        super().__init__()

        self.configs = configs

        self.patch_len = configs.patch_len

        self.patch_num = configs.seq_len // configs.patch_len

        self.d_model = configs.d_model

       

        # 消融模式: 0(Base), 1(仅HTA), 2(仅AEM), 3(Full)

        self.ablation_mode = getattr(configs, 'ablation_mode', 3)

       

        self.Ne, self.Nx = (1, configs.enc_in - 1) if configs.features == 'MS' else (configs.enc_in, configs.enc_in)

           

        self.revin = RevIN(configs.enc_in)

        self.val_emb = nn.Linear(self.patch_len, configs.d_model)

       

        # 实例化核心模块

        if self.ablation_mode in [1, 3]:

            self.hta = HardTemporalAlignment(configs.seq_len, self.Nx, getattr(configs, 'max_lag', 24))

        if self.ablation_mode in [2, 3]:

            self.aem = AsymmetricEventModulator(configs.d_model, self.Nx, self.Ne)

           

        self.encoder = nn.ModuleList([MLPBlock(self.patch_num, configs.d_model, configs.dropout) for _ in range(configs.e_layers)])

        self.head = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        B = x_enc.shape[0]

        x_norm = self.revin(x_enc, mode='norm')

       

        raw_endo, raw_exog = (x_norm[:,:,-1:], x_norm[:,:,:-1]) if self.configs.features == 'MS' else (x_norm, x_norm)

        raw_endo, raw_exog = raw_endo.transpose(1, 2), raw_exog.transpose(1, 2)



        # ----------------- 消融逻辑串联 -----------------

        # 步骤 1: HTA 硬时滞对齐 (如果禁用，则默认 lag=0)

        if hasattr(self, 'hta'):

            aligned_exog = self.hta(raw_exog)

        else:

            aligned_exog = raw_exog



        # 语义切块

        z_endo = self.val_emb(raw_endo.unfold(-1, self.patch_len, self.patch_len))

        z_exog = self.val_emb(aligned_exog.unfold(-1, self.patch_len, self.patch_len))



        # 步骤 2: AEM 非对称特征调制

        if hasattr(self, 'aem'):

            gamma, beta = self.aem(z_exog)

            # 零初始化的 FiLM 调制：完全杜绝负迁移！

            x = z_endo * (1.0 + gamma) + beta

        else:

            x = z_endo # 纯 Base



        # 骨干推理

        x = x.reshape(-1, self.patch_num, self.d_model)

        for layer in self.encoder:

            x = layer(x)

       

        res = self.head(x.reshape(B, self.Ne, -1)).transpose(1, 2)

        return self.revin(res, mode='denorm')
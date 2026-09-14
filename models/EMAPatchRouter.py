import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. 基础平稳化组件 (RevIN)
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
            x = x * stdev + mean
            return x

# =========================================================================
# 2. 增强版指数平滑分解
# =========================================================================
class EMADecomp(nn.Module):
    def __init__(self, kernel_size=25, alpha=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        init_logit = torch.log(torch.tensor(alpha / (1 - alpha)))
        self.alpha_logit = nn.Parameter(init_logit)
        self.register_buffer('arith_range', torch.arange(kernel_size, dtype=torch.float32))

    def forward(self, x):
        B, L, C = x.shape
        x_trans = x.permute(0, 2, 1)
        x_pad = F.pad(x_trans, (self.kernel_size - 1, 0), mode='replicate')

        alpha = torch.sigmoid(self.alpha_logit)
        weights = alpha * ((1 - alpha) ** self.arith_range)
        weights = weights / (weights.sum() + 1e-9)
        weights = torch.flip(weights, dims=[0]).view(1, 1, self.kernel_size)

        trend = F.conv1d(
            x_pad.view(B * C, 1, L + self.kernel_size - 1),
            weights,
            groups=1
        ).view(B, C, L).permute(0, 2, 1)

        return x - trend, trend

# =========================================================================
# 3. 终极组件：目标条件 MLP 路由 (Receiver-Centric Unified MLP Router)
# =========================================================================
class UnifiedMLPRouter(nn.Module):
    def __init__(self, out_channels, exo_dim, d_model, max_lag=24, bottleneck_dim=16, router_init=1.0):
        super().__init__()
        self.max_lag = max_lag
        self.exo_dim = exo_dim
        self.out_channels = out_channels

        # 物理衰减先验 (跨度约 4 步)
        self.register_buffer('lag_indices', torch.arange(max_lag + 1, dtype=torch.float32))
        self.decay_logit = nn.Parameter(torch.tensor(math.log(4.0)))

        # 核心创新：条件路由生成器 (替代 Attention，O(C) 复杂度)
        self.target_context = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU()
        )
        
        # 1. 直接生成目标专属的时滞矩阵 [O, Exo_Vars * Lags]
        self.lag_router = nn.Linear(bottleneck_dim, exo_dim * (max_lag + 1))
        
        # 2. 生成变量过滤门控，斩断无关变量的干扰
        self.var_gate = nn.Linear(bottleneck_dim, exo_dim)

        # FiLM 调制参数生成 (极度安全的残差注入)
        self.film_gen = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, d_model * 2)
        )
        nn.init.zeros_(self.film_gen[2].weight)
        nn.init.zeros_(self.film_gen[2].bias)


        # 独立通道的残差缩放因子 (修复 Broadcasting 维度对齐)
        # update 的 shape 是 [B, P, out_channels, D_model]
        # 必须设为 [1, 1, out_channels, 1] 才能严格对齐目标变量的维度
        self.router_alpha = nn.Parameter(torch.ones(1, 1, out_channels, 1) * router_init)

    def forward(self, z_endo, raw_exo_seasonal, val_emb_layer, patch_len, stride):
        B, P, O, D = z_endo.shape
        L = raw_exo_seasonal.shape[2] # raw_exo_seasonal 维度为 [B, E, L]

        # ----------------------------------------------------
        # 步骤 1: 目标视角的上下文压缩 (Target Compression)
        # ----------------------------------------------------
        H_endo = z_endo.mean(dim=1)           # [B, O, d_model]
        context = self.target_context(H_endo) # [B, O, bottleneck]

        # ----------------------------------------------------
        # 步骤 2: MLP 直接提货 (Lag Weight Generation)
        # ----------------------------------------------------
        # 目标直接投影出属于自己的 3D 路由栅格 [Batch, Target, Exo_var, Lags]
        lag_matrix = self.lag_router(context).view(B, O, self.exo_dim, self.max_lag + 1)
        
        decay_rate = torch.exp(self.decay_logit)
        causal_prior = torch.exp(-self.lag_indices / decay_rate).view(1, 1, 1, -1)
        lag_weights = F.softmax(lag_matrix + causal_prior.log(), dim=-1) # [B, O, E, Lags]
        self.lag_distribution = lag_weights.detach()

        # ----------------------------------------------------
        # 步骤 3: 原始时间轴切片与动态对齐 (Einsum 极速对齐)
        # ----------------------------------------------------
        exo_features = raw_exo_seasonal.transpose(1, 2) # [B, L, E]
        shifted_exos = []
        for lag in range(self.max_lag + 1):
            if lag == 0:
                shifted = exo_features
            else:
                shifted = F.pad(exo_features[:, :-lag, :], (0, 0, lag, 0), mode='replicate')
            shifted_exos.append(shifted)
        
        E_cands = torch.stack(shifted_exos, dim=-1) # [B, L, E, Lags]

        # 乘加聚合：利用各个 Target 的独立偏好，去组装它眼中的外部世界
        # boel (Batch, Target, Exo, Lag) * btel (Batch, Time, Exo, Lag) -> boet (Batch, Target, Exo, Time)
        aligned_exo = torch.einsum('boel, btel -> boet', lag_weights, E_cands)

        # ----------------------------------------------------
        # 步骤 4: 变量级硬隔离过滤 (Variable Gating)
        # ----------------------------------------------------
        var_weights = torch.sigmoid(self.var_gate(context)) # [B, O, E]
        # boet (Batch, Target, Exo, Time) * boe (Batch, Target, Exo) -> bot (Batch, Target, Time)
        gated_exo = torch.einsum('boet, boe -> bot', aligned_exo, var_weights) 

        # ----------------------------------------------------
        # 步骤 5: 切块映射与 FiLM 注入 (Patching & FiLM)
        # ----------------------------------------------------
        pad_len = patch_len + stride * (P - 1) - L
        if pad_len > 0:
            gated_exo_pad = F.pad(gated_exo, (0, pad_len), mode='replicate')
        else:
            gated_exo_pad = gated_exo
        
        z_exo_patches = gated_exo_pad.unfold(-1, patch_len, stride).transpose(1, 2) # [B, P, O, patch_len]
        
        # 完美复用主干的 val_emb，保证外生变量与内生变量在同一语义高维空间
        z_exo_embedded = val_emb_layer(z_exo_patches) # [B, P, O, d_model]

        gamma, beta = torch.chunk(self.film_gen(z_exo_embedded), 2, dim=-1)
        gate = torch.sigmoid(gamma)
        
        # 安全残差：z_endo只受到门控允许的轻微拨动，绝不喧宾夺主
        update = z_endo * (0.1 * gate) + 0.1 * beta
        
        return self.router_alpha * update

# =========================================================================
# 4. 严密的通道独立底座 (Strict CI PatchMLPBlock)
# =========================================================================
class CausalPatchMLPBlock(nn.Module):
    def __init__(self, patch_num, d_model, enc_in, dropout, ls_init=1e-4, expansion_factor=4):
        super().__init__()
        self.expansion_factor = expansion_factor

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(patch_num, patch_num * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_num * expansion_factor, patch_num)
        )

        self.feat_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model)
        )

        # 专门处理目标变量间的互相关 (Soft-CI)
        self.var_mlp = nn.Sequential(
            nn.Linear(enc_in, enc_in * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_in * expansion_factor, enc_in)
        )

        self.ls_time = nn.Parameter(torch.ones(d_model) * ls_init)
        self.ls_feat = nn.Parameter(torch.ones(d_model) * ls_init)

    def forward(self, z, ablation_mode='full'):
        z_time = z.permute(0, 2, 3, 1)
        y_0 = self.time_mlp(z_time)
        y_0 = y_0.permute(0, 3, 1, 2) 
        
        if ablation_mode == 'wo_layerscale':
            z = z + y_0
        else:
            z = z + y_0 * self.ls_time.view(1, 1, 1, -1)
        z = self.norm1(z) 

        z_var = z.permute(0, 1, 3, 2) 
        y_1 = self.var_mlp(z_var)
        y_1 = y_1.permute(0, 1, 3, 2) 
        
        if ablation_mode == 'wo_layerscale':
            z = z + y_1
        else:
            z = z + y_1 * self.ls_feat.view(1, 1, 1, -1)
        z = self.norm2(z) 

        return z

# =========================================================================
# 5. 最终主模型 (完美双流交汇)
# =========================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.ablation_mode = getattr(configs, 'ablation_mode', 'full')
        self.C = configs.enc_in
        self.features = getattr(configs, 'features', 'M')
        explicit_use_exo = getattr(configs, 'use_exo', None)
        
        if explicit_use_exo is not None:
            self.use_exo = explicit_use_exo == 1
        else:
            self.use_exo = (self.features == 'MS' and self.C > 1) or self.features == 'M'
            
        self.out_dim = 1 if self.features == 'MS' else self.C
        self.exo_dim = self.C - 1 if self.features == 'MS' else self.C

        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.d_model = configs.d_model
        self.e_layers = getattr(configs, 'e_layers', 2)

        self.revin = RevIN(self.C)

        self.decomp = EMADecomp(
            kernel_size=getattr(configs, 'ema_kernel', 25), # Weather建议改为 9
            alpha=getattr(configs, 'alpha', 0.5)
        )

        self.val_emb = nn.Linear(self.patch_len, self.d_model)

        # ----------------- 双流架构 -----------------
        # 主干流：只有 out_dim 进入，防泄露
        self.mlp_blocks = nn.ModuleList([
            CausalPatchMLPBlock(
                self.patch_num, self.d_model, self.out_dim, configs.dropout,  
                ls_init=getattr(configs, 'ls_init', 1e-4),
                expansion_factor=getattr(configs, 'expansion_factor', 4)
            ) for _ in range(self.e_layers)
        ])

# 侧翼流：纯 MLP 条件路由
        self.driver_router = UnifiedMLPRouter(
            out_channels=self.out_dim,
            exo_dim=self.exo_dim if self.use_exo else 1, 
            d_model=self.d_model,
            max_lag=getattr(configs, 'max_lag', 24),
            bottleneck_dim=getattr(configs, 'bottleneck_dim', 16),
            router_init=getattr(configs, 'router_init', 0.1) # <---- 动态读取 configs
        )

        self.norm = nn.LayerNorm(self.d_model)
        self.predict_head = nn.Linear(self.d_model * self.patch_num, configs.pred_len)
        self.trend_linear = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B = x_enc.shape[0]
        
        x_norm = self.revin(x_enc, mode='norm')

        if self.ablation_mode == 'wo_decomp':
            seasonal, trend_pred = x_norm, 0.0
        else:
            seasonal, trend = self.decomp(x_norm)
            target_trend = trend[:, :, -self.out_dim:].transpose(1, 2)
            trend_pred = self.trend_linear(target_trend).transpose(1, 2)

        # --- 获取外部世界的原始序列 ---
        if self.use_exo and self.exo_dim > 0:
            if self.features == 'MS':
                exo_seasonal = seasonal[:, :, :-self.out_dim]
            else:
                exo_seasonal = seasonal
        else:
            exo_seasonal = None

        # --- 目标自己走过严格通道独立的 MLP ---
        if self.features == 'MS':
            endo_seasonal = seasonal[:, :, -self.out_dim:]
        else:
            endo_seasonal = seasonal

        pad_len = self.patch_len + self.stride * (self.patch_num - 1) - endo_seasonal.shape[1]
        if pad_len > 0:
            endo_pad = F.pad(endo_seasonal.permute(0, 2, 1), (0, pad_len), mode='replicate').permute(0, 2, 1)
        else:
            endo_pad = endo_seasonal

        z_endo = endo_pad.unfold(1, self.patch_len, self.stride) 
        z_endo = self.val_emb(z_endo)                               

        if self.ablation_mode != 'wo_intra':
            for block in self.mlp_blocks:
                z_endo = block(z_endo, ablation_mode=self.ablation_mode)

        # --- 终极融合点：主干获取路由生成的提货信息 ---
        if self.ablation_mode == 'wo_router' or exo_seasonal is None:
            z_fused = z_endo
        else:
            router_update = self.driver_router(
                z_endo=z_endo, 
                raw_exo_seasonal=exo_seasonal.transpose(1, 2), # 喂入 [B, E, L]
                val_emb_layer=self.val_emb,                    # 共享语义投影空间
                patch_len=self.patch_len, 
                stride=self.stride
            )
            z_fused = z_endo + router_update

        z_out = self.norm(z_fused) 
        z_out = z_out.transpose(1, 2).reshape(B, self.out_dim, -1) 
        seasonal_pred = self.predict_head(z_out).transpose(1, 2)

        return self.revin(seasonal_pred + trend_pred, mode='denorm', target_dim=self.out_dim)
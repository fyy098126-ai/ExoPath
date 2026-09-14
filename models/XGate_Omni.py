import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. RevIN & Decomp (数据平稳化与时间序列分解)
# =========================================================================
class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization)"""
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

class DynamicSMADecomp(nn.Module):
    """[消融专用] 传统的静态多尺度滑动平均分解 (SMA)"""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.scales = [kernel_size // 2, kernel_size, kernel_size * 2]
        self.scale_weights = nn.Parameter(torch.zeros(len(self.scales)))

    def forward(self, x):
        norm_w = F.softmax(self.scale_weights, dim=0)
        trends = []
        for k in self.scales:
            k = k if k % 2 != 0 else k + 1
            p = k // 2
            x_pad = F.pad(x.permute(0, 2, 1), (p, p), mode='replicate')
            trends.append(F.avg_pool1d(x_pad, kernel_size=k, stride=1).permute(0, 2, 1))
        final_trend = sum(t * norm_w[i] for i, t in enumerate(trends))
        return final_trend, x - final_trend

class ExoGuidedEMA(nn.Module):
    """
    [核心模块] 外生指导的指数移动平均分解 (Exo-Guided CC-EMA)
    利用外生变量的波动率 (Volatility) 动态调节内生序列的平滑系数 (Alpha)。
    """
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
        
        # 1. 计算外生波动作先验
        if self.exo_dim > 0:
            x_exo = x[:, :, :-self.out_dim]
            exo_diff = torch.abs(x_exo[:, 1:, :] - x_exo[:, :-1, :])
            exo_vol = exo_diff.mean(dim=[1, 2]) 
            exo_vol_prior = exo_vol.unsqueeze(1).detach() # [B, 1] 截断梯度，防止崩盘
            alpha_logits = self.base_alpha_logits.unsqueeze(0) - self.vol_sensitivity * exo_vol_prior
        else:
            alpha_logits = self.base_alpha_logits.unsqueeze(0).repeat(B, 1)
            
        alpha = torch.sigmoid(alpha_logits).to(torch.double) # [B, out_dim]

        # 2. 向量化 O(1) 并行 EMA 计算
        powers = torch.flip(torch.arange(T, dtype=torch.double, device=device), dims=(0,)).unsqueeze(1)
        weights = torch.pow((1.0 - alpha).unsqueeze(1), powers.unsqueeze(0)) 
        divisor = weights.clone()
        
        w_0 = weights[:, 0:1, :]
        w_rest = weights[:, 1:, :] * alpha.unsqueeze(1)
        weights_mod = torch.cat([w_0, w_rest], dim=1) 

        x_endo = x[:, :, -self.out_dim:].to(torch.double)
        out_endo = torch.cumsum(x_endo * weights_mod, dim=1) / divisor
        trend_endo = out_endo.to(torch.float32)

        # 3. 外生变量通道保持静态 0.5 EMA 平滑
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
# 2. MILE: Macro-Independent Linear Extrapolator (宏观独立线性外推器)
# =========================================================================
class MILE(nn.Module):
    """
    [趋势流核心] 纯通道独立 (CI) 趋势外推器。
    抛弃 GELU 和跨通道交互，依靠 AvgPool1d 强制平滑，保障 720 步等长线预测的绝对刚性。
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        # 容量翻倍以补偿无激活函数带来的拟合力下降
        self.lin1 = nn.Linear(seq_len, d_model * 2)
        # 核心平滑机制：通过池化强制提取低频宏观趋势
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.norm1 = nn.LayerNorm(d_model)
        self.lin2 = nn.Linear(d_model, d_model)

    def forward(self, trend_all, out_dim, mode='full'):
        # [核心消融逻辑]: 如果允许外生污染，必须在物理通道上施加混合
        if mode == 'w_exo_trend' and trend_all.shape[2] > out_dim:
            exo_mean = trend_all[:, :, :-out_dim].mean(dim=-1, keepdim=True)
            endo_trend = trend_all[:, :, -out_dim:] + exo_mean # 制造协变量偏移
        else:
            endo_trend = trend_all[:, :, -out_dim:] # 纯净内生隔离
        
        # 维度转换以适配 Linear 层沿时间维度 T 的映射
        x = endo_trend.transpose(1, 2)         # Shape: [B, out_dim, T]
        
        x = self.lin1(x)                       # Shape: [B, out_dim, d_model * 2]
        x = self.pool1(x)                      # Shape: [B, out_dim, d_model]
        x = self.norm1(x)                      # Shape: [B, out_dim, d_model]
        x = self.lin2(x)                       # Shape: [B, out_dim, d_model]
        
        return x # 纯净、刚性的内生趋势表征


# =========================================================================
# 3. LRER: Low-Rank Exogenous Router (低秩外生路由器)
# =========================================================================
class LRER(nn.Module):
    """
    [季节流组件] 全局语境低秩门控。
    通过瓶颈结构 (Bottleneck) 过滤高频外生噪音，生成全局降噪 Mask。
    """
    def __init__(self, c_in, rank=16, dropout=0.1):
        super().__init__()
        self.rank = min(rank, c_in)
        self.down = nn.Linear(c_in, self.rank, bias=False)
        self.up = nn.Linear(self.rank, c_in, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x Shape: [B, d_model, c_in_router]
        gate_logits = self.up(self.dropout(F.relu(self.down(x))))
        return torch.sigmoid(gate_logits)


# =========================================================================
# 4. XGate_Omni 终极主模型
# =========================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ablation_mode = getattr(configs, 'ablation_mode', 'full')
        
        # 维度解析
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

        # ── 基础平滑层 ──
        self.revin = RevIN(self.C)
        self.decomp = ExoGuidedEMA(c_in=self.C, out_dim=self.out_dim, features=self.features)
        
        if self.ablation_mode == 'w_sma_decomp':    
            self.sma_decomp = DynamicSMADecomp(kernel_size=getattr(configs, 'sma_kernel', 25))

        # ── 趋势流 (Trend Stream) ──
        self.trend_mixer = MILE(self.seq_len, self.d_model)
        self.trend_head = nn.Linear(self.d_model, self.pred_len)
        # 消融专供：开启即证明非线性对趋势有害
        self.ablation_trend_act = nn.GELU() if self.ablation_mode in ('w_trend_act') else nn.Identity()

        # ── 季节流 (Seasonal Stream) ──
        self.seasonal_proj = nn.Linear(self.seq_len, self.d_model)
        
        # EBG (Endogenous Blueprint Gate)：内生蓝图门控
        self.ebt_token = nn.Parameter(torch.ones(1, self.out_dim, self.d_model))
        self.ebg = nn.Sequential(
            nn.Linear(2 * self.d_model, t_ff), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(t_ff, 2 * self.d_model), nn.Sigmoid()
        )
        
        # 全量特征路由器 (Global Router)：通道数 = 外生通道 + 内生通道
        self.c_in_router = (self.C + self.out_dim) if self.features == 'M' else (self.exo_dim + self.out_dim)
        self.exo_router = LRER(self.c_in_router, rank=self.rank, dropout=drop)
        
        # 季节头：接收 C-LoRA 拼接后的特征，输入维度为 2 * d_model
        self.seasonal_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(2 * self.d_model, self.pred_len),
        )

        if self.ablation_mode == 'w_adaptive_fuse':
            self.trend_mix_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        mode = self.ablation_mode
        b = x_enc.shape[0]

        # 1. 实例归一化
        x = self.revin(x_enc, 'norm') if mode != 'wo_revin' else x_enc

        # 2. 分解 (CC-EMA)
        if mode == 'wo_decomp':
            seasonal_all = x
            trend_all = x 
        elif mode == 'w_sma_decomp': 
            trend_all, seasonal_all = self.sma_decomp(x)
        else:
            trend_all, seasonal_all = self.decomp(x)

        # ─── 趋势流 (MILE) ────────────────────────────────────────────────────────
        if mode not in ('wo_trend'):
            # 内部自动剔除外生变量，返回 Shape: [B, out_dim, d_model]
            z_trend = self.trend_mixer(trend_all, self.out_dim, mode) 
            
            # 消融专用 GELU，正常模式下等价于 Identity()
            z_trend = self.ablation_trend_act(z_trend)
            
            # 线性投射并转置为 [B, pred_len, out_dim]
            trend_pred = self.trend_head(z_trend).transpose(1, 2) 

        # ─── 季节流 (C-LoRA & Global Context) ────────────────────────────────────
        z_seas = self.seasonal_proj(seasonal_all.permute(0, 2, 1))  # [B, C, d_model]
        z_endo = z_seas[:, -self.out_dim:, :]                       # 内生 [B, out_dim, d_model]
        z_exo = z_seas[:, :-self.out_dim, :] if self.features == 'MS' else z_seas # 外生

        # [步骤 A]: EBG 蓝图提取
        if mode == 'wo_ebg': 
            origin_gated = z_endo
            ebt_gated = z_endo
        else:
            glob = self.ebt_token.repeat(b, 1, 1) # [B, out_dim, d_model]
            en_emb = torch.cat([z_endo, glob], dim=-1) # [B, out_dim, 2*d_model] 
            en_gated = en_emb * self.ebg(en_emb)
            # EBG 门控极其优雅的物理作用：将内生信号一分为二
            origin_gated = en_gated[:, :, :self.d_model] # 物理内生震荡 [B, out_dim, d_model]
            ebt_gated = en_gated[:, :, self.d_model:]    # 逻辑检索条件 [B, out_dim, d_model]

        # [步骤 B]: LRER 全局语境提取与降噪
        if mode != 'wo_router' and self.c_in_router > self.out_dim:
            query_vec = z_endo if mode == 'wo_query_raw' else ebt_gated
            
            # 拼接形成全量全局语境图 (Global Context) -> Shape: [B, c_in_router, d_model]
            ex_emb = torch.cat([z_exo, query_vec], dim=1) 
            
            if mode == 'wo_global':
                gate = torch.full((b, self.c_in_router, self.d_model), 0.5, device=x_enc.device)
            else:
                # 路由器依据特征维度 (d_model) 生成各通道的关注度 Mask
                # 输入转置为 [B, d_model, c_in_router]，输出再转回来
                gate = self.exo_router(ex_emb.transpose(1, 2)).transpose(1, 2) 
                
            # Gate 全局抑制无用的外生噪点
            gated_ex_emb = ex_emb * gate  
            
            # 截取联合路由中经过外生条件化的目标查询分支。
            # 纯外生表示不直接进入预测头，但其信息通过全局门控调制目标查询。
            refined_ebt = gated_ex_emb[:, -self.out_dim:, :]

        else:
            refined_ebt = ebt_gated

        # [步骤 C]: C-LoRA 哲学 —— 身份感知的拼接融合 (Identity-Aware Concatenation)
        if mode == 'wo_origin':
            seas_in = torch.cat([refined_ebt, refined_ebt], dim=-1)
        else:
            # 原生震荡 (origin_gated) 与 环境语境 (refined_ebt) 拼接拓宽带宽
            seas_in = torch.cat([origin_gated, refined_ebt], dim=-1) # Shape: [B, out_dim, 2 * d_model]
        
        # 季节预测
        seasonal_pred = self.seasonal_head(seas_in).transpose(1, 2)  # [B, pred_len, out_dim]

        # ── 5. Fusion ─────────────────────────────────────────────────────
        if mode in ('wo_trend'):
            res_out = seasonal_pred
        elif mode == 'w_adaptive_fuse':
            t_w = torch.sigmoid(self.trend_mix_alpha)
            res_out = t_w * trend_pred + seasonal_pred
        else:
            # Y = T + S，与 DBLoss 数学推导强同构
            res_out = trend_pred + seasonal_pred

        # ── 6. RevIN Inverse ──────────────────────────────────────────────
        if mode != 'wo_revin':
            res_out = self.revin(res_out, 'denorm', target_dim=self.out_dim if self.features == 'MS' else None)
            
        return res_out
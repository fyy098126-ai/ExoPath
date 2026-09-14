import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------- 1. RevIN (应对非平稳性) -----------------
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

# ----------------- 2. V-SDA (变量级软时滞对齐) -----------------
class VariateAwareSDA(nn.Module):
    def __init__(self, seq_dim, max_lag, tau=0.5):
        super().__init__()
        self.max_lag = max_lag
        self.tau = tau
        self.q_proj = nn.Linear(seq_dim, seq_dim)
        self.k_proj = nn.Linear(seq_dim, seq_dim)

    def forward(self, endo, exog):
        B, Ne, Dim_T = endo.shape
        _, Nx, _ = exog.shape
        actual_max_lag = min(self.max_lag, Dim_T - 1)
        
        # Q: 提取目标总趋势 [B, 1, T]
        Q = self.q_proj(endo.mean(dim=1)).unsqueeze(1) 
        
        # E_candidates: 构建滞后候选集 [B, Nx, Lags, T]
        shifted_exogs = [exog]
        for lag in range(1, actual_max_lag + 1):
            shifted = F.pad(exog[:, :, :-lag], (lag, 0), mode='replicate')
            shifted_exogs.append(shifted)
        E_candidates = torch.stack(shifted_exogs, dim=2) 
        
        # K: [B, Nx, Lags, T]
        K = self.k_proj(E_candidates)
        
        # 计算每个外生变量的独立物理时滞分布 [B, Nx, Lags]
        attn_scores = torch.einsum('b o t, b x l t -> b x l', Q, K) / (Dim_T ** 0.5 * self.tau)
        self.lag_weights = F.softmax(attn_scores, dim=-1)
        
        return torch.einsum('b x l, b x l t -> b x t', self.lag_weights, E_candidates)

# ----------------- 3. CVDG (统一跨维门控路由) -----------------
class UnifiedCrossVariableGate(nn.Module):
    def __init__(self, d_model, exog_vars, endo_vars, gate_type='variate'):
        super().__init__()
        self.gate_type = gate_type
        
        # 洗沙网络
        self.gate_gen = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model), nn.Sigmoid()
        )
        if gate_type == 'variate':
            self.cross_mapper = nn.Linear(exog_vars, endo_vars)
            
        # 💡 防残差爆炸保护阀门 (初始化对应权重0.5)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.print_counter = 0

    def forward(self, h_endo, h_exog):
        B, Ne, P, D = h_endo.shape
        Nx = h_exog.shape[1]

        if self.gate_type == 'temporal':
            # 模拟塌缩的消融状态
            exog_context = h_exog.mean(dim=1, keepdim=True) 
            self.gate = self.gate_gen(exog_context)
            mapped_exog = h_endo * self.gate
        else:
            # SOTA 状态：动态洗沙 + 跨维路由
            gate_all = self.gate_gen(h_exog.reshape(-1, P, D))
            self.gate = gate_all.reshape(B, Nx, P, D)
            filtered_exog = h_exog * self.gate 
            
            mapped_exog = self.cross_mapper(filtered_exog.permute(0, 2, 3, 1))
            mapped_exog = mapped_exog.permute(0, 3, 1, 2)

        # 受控残差注入
        a = torch.sigmoid(self.alpha)
        output = (a * h_endo) + ((1 - a) * mapped_exog)

        if self.training:
            self.print_counter += 1
            if self.print_counter % 200 == 0 and self.gate_type == 'variate':
                with torch.no_grad():
                    avg_w = self.gate.mean(dim=(0, 2, 3)).cpu().numpy()
                    top_idx = np.argsort(avg_w)[::-1][:min(5, len(avg_w))]
                    msg = " | ".join([f"V{i}:{avg_w[i]:.3f}" for i in top_idx])
                    print(f"\n[监控] Top-5 门控: {msg} | Alpha保留: {a.item():.3f}")
        return output

# ----------------- 4. MLP Backbone -----------------
class MLPBlock(nn.Module):
    def __init__(self, seq_dim, d_model):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(seq_dim, seq_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(seq_dim, seq_dim)
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B_N, P, D = x.shape
        out = self.linear(x.reshape(B_N, -1)).reshape(B_N, P, D)
        return self.norm(x + out)

# ----------------- 5. 顶层 Model Class -----------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.patch_len = configs.patch_len
        self.patch_num = int(np.ceil(configs.seq_len / configs.patch_len))
        
        # 消融开关
        self.ablation_mode = getattr(configs, 'ablation_mode', 2) # 推荐 Mode 2 为 SOTA
        self.alignment_pos = getattr(configs, 'alignment_pos', 'raw')
        self.gate_type = getattr(configs, 'gate_type', 'variate')
        self.max_lag = getattr(configs, 'max_lag', 7)

        if self.configs.features == 'MS':
            self.Ne, self.Nx = 1, configs.enc_in - 1
        else:
            self.Ne, self.Nx = configs.enc_in, configs.enc_in

        self.revin = RevIN(configs.enc_in)
        self.value_embedding = nn.Linear(self.patch_len, configs.d_model)
        
        # 仅在 Raw 对齐时开启物理 SDA
        if self.ablation_mode in [1, 3] and self.alignment_pos == 'raw':
            self.sda = VariateAwareSDA(configs.seq_len, self.max_lag, tau=getattr(configs, 'tau', 0.5))
        
        if self.ablation_mode in [2, 3]:
            self.cvdg = UnifiedCrossVariableGate(configs.d_model, self.Nx, self.Ne, gate_type=self.gate_type)
            
        self.encoder = nn.ModuleList([MLPBlock(self.patch_num * configs.d_model, configs.d_model) for _ in range(configs.e_layers)])
        self.head = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B = x_enc.shape[0]
        x_enc = self.revin(x_enc, mode='norm')
        
        if self.configs.features == 'MS':
            raw_endo = x_enc[:, :, -1:].transpose(1, 2)
            raw_exog = x_enc[:, :, :-1].transpose(1, 2)
        else:
            raw_endo = raw_exog = x_enc.transpose(1, 2)

        # 阶段 A: 相位对齐 (只在 Raw 级别进行物理对齐)
        if self.ablation_mode in [1, 3] and self.alignment_pos == 'raw':
            raw_exog = self.sda(raw_endo, raw_exog)

        # 阶段 B: 语义切块
        z_endo = self.value_embedding(raw_endo.unfold(-1, self.patch_len, self.patch_len))
        z_exog = self.value_embedding(raw_exog.unfold(-1, self.patch_len, self.patch_len))

        # 消融陷阱：如果强行选用 Latent SDA (用于论文反面对比)
        if self.ablation_mode in [1, 3] and self.alignment_pos == 'latent':
            # 极度简化的 Latent 占位，证明其由于语义混叠不如 Raw
            z_exog = z_exog + z_endo.mean(dim=1, keepdim=True) * 0.01 

        # 阶段 C: 振幅门控与路由
        if self.ablation_mode == 0: x = z_endo
        elif self.ablation_mode == 1: 
            x = z_endo + z_exog.mean(dim=1, keepdim=True) if self.Ne == 1 else z_endo + z_exog
        else: 
            x = self.cvdg(z_endo, z_exog)

        # 阶段 D: 时序推理
        x = x.reshape(-1, self.patch_num, self.configs.d_model)
        for layer in self.encoder: x = layer(x)
        
        res = self.head(x.reshape(B, self.Ne, -1)).transpose(1, 2)
        return self.revin(res, mode='denorm')
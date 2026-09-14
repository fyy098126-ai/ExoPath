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

# ----------------- 2. V-SDA (变量级软时滞对齐 - 相位解耦) -----------------
class VariateAwareSDA(nn.Module):
    def __init__(self, seq_dim, max_lag, tau=0.5):
        super().__init__()
        self.max_lag = max_lag
        self.tau = tau
        self.q_proj = nn.Linear(seq_dim, seq_dim)
        self.k_proj = nn.Linear(seq_dim, seq_dim)

    def forward(self, endo, exog):
        # endo: [B, Ne, T], exog: [B, Nx, T]
        B, Ne, Dim_T = endo.shape
        _, Nx, _ = exog.shape
        actual_max_lag = min(self.max_lag, Dim_T - 1)
        
        # 💡 核心修复：独立变量寻址，杜绝均值塌缩
        if Ne == 1: 
            # MS 模式：用唯一的目标序列作为 Query，广播给所有外生变量
            Q = self.q_proj(endo).expand(-1, Nx, -1) # [B, Nx, T]
        else:       
            # M 模式：每个变量独立作为 Query 去寻找属于自己的时滞
            Q = self.q_proj(endo) # [B, Ne, T] 此时 Ne=Nx

        # 构建滞后候选集 [B, Nx, Lags, T] (向右平移，无未来信息泄漏)
        shifted_exogs = [exog]
        for lag in range(1, actual_max_lag + 1):
            shifted = F.pad(exog[:, :, :-lag], (lag, 0), mode='replicate')
            shifted_exogs.append(shifted)
        E_candidates = torch.stack(shifted_exogs, dim=2) 
        
        K = self.k_proj(E_candidates) # [B, Nx, Lags, T]
        
        # 变量级独立注意力 (Variate-wise Attention)
        attn_scores = torch.einsum('b x t, b x l t -> b x l', Q, K) / (Dim_T ** 0.5 * self.tau)
        self.lag_weights = F.softmax(attn_scores, dim=-1) # [B, Nx, Lags]
        
        # 根据算出的独立时滞分布，重组外生波形
        aligned_exog = torch.einsum('b x l, b x l t -> b x t', self.lag_weights, E_candidates)
        return aligned_exog

# ----------------- 3. CVDG (跨维动态门控 - 振幅调制与路由) -----------------
class UnifiedCrossVariableGate(nn.Module):
    def __init__(self, d_model, exog_vars, endo_vars, gate_type='variate', initial_alpha=0.5):
        super().__init__()
        self.gate_type = gate_type
        
        # 1. 提取上帝视角的特征提取网络
        self.gate_feat_extractor = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2), 
            nn.GELU()
        )
        
        # 2. 标量打分器 (每个变量输出 1 个总分)
        self.gate_scorer = nn.Linear(d_model // 2, 1)
        
        if gate_type == 'variate':
            self.cross_mapper = nn.Linear(exog_vars, endo_vars)
            
        # 💡 顶级设计：带物理先验的自适应阀门
        # 将传入的 initial_alpha 转化为 logit 空间
        prob = np.clip(initial_alpha, 1e-4, 1 - 1e-4) 
        initial_logit = np.log(prob / (1 - prob))
        self.alpha = nn.Parameter(torch.tensor([initial_logit], dtype=torch.float32))
        
        self.print_counter = 0

    # 终端极客进度条辅助函数
    def _draw_bar(self, value, length=20):
        filled = int(round(value * length))
        return '█' * filled + '░' * (length - filled)

    def forward(self, h_endo, h_exog):
        B, Ne, P, D = h_endo.shape
        Nx = h_exog.shape[1]

        if self.gate_type == 'temporal':
            exog_context = h_exog.mean(dim=1, keepdim=True) 
            temp_feat = self.gate_feat_extractor(torch.cat([exog_context, exog_context], dim=-1))
            self.gate = torch.sigmoid(self.gate_scorer(temp_feat))
            mapped_exog = h_endo * self.gate
        else:
            # 1. 提取宏观表征 (上帝视角)
            var_repr = h_exog.mean(dim=2) # [B, Nx, D]
            global_repr = var_repr.mean(dim=1, keepdim=True).expand(-1, Nx, -1) # [B, Nx, D]
            
            # 2. 拼接并提取隐层特征
            combined_feat = torch.cat([var_repr, global_repr], dim=-1) # [B, Nx, D * 2]
            hidden_feat = self.gate_feat_extractor(combined_feat) # [B, Nx, D//2]
            raw_logits = self.gate_scorer(hidden_feat) # [B, Nx, 1]
            
            # 💡 致命缺陷修复：跨变量竞争归一化 (Cross-Variate Normalization)
            if Nx > 1:
                mu = raw_logits.mean(dim=1, keepdim=True)
                sigma = raw_logits.std(dim=1, keepdim=True) + 1e-5
                normed_logits = (raw_logits - mu) / sigma
            else:
                normed_logits = raw_logits
            
            # 3. 温和极化激活 (Sigmoid 映射到 0~1)
            self.gate = torch.sigmoid(normed_logits) # [B, Nx, 1]
            
            # 4. 门控洗沙与路由映射
            filtered_exog = h_exog * self.gate.unsqueeze(2) 
            mapped_exog = self.cross_mapper(filtered_exog.permute(0, 2, 3, 1))
            mapped_exog = mapped_exog.permute(0, 3, 1, 2)

        # 5. 安全退化注入
        a = torch.sigmoid(self.alpha)
        output = (a * h_endo) + ((1 - a) * mapped_exog)

        # 实时精简监控打印
        if self.training:
            self.print_counter += 1
            if self.print_counter % 200 == 0 and self.gate_type == 'variate':
                with torch.no_grad():
                    avg_weights = self.gate.mean(dim=0).squeeze().cpu().numpy()
                    if avg_weights.ndim == 0: avg_weights = np.array([avg_weights])
                    
                    top_n = min(5, Nx)
                    top_indices = np.argsort(avg_weights)[::-1][:top_n]
                    worst_idx = np.argsort(avg_weights)[0] # 最差的1个用于对比
                    
                    weight_msg = " | ".join([f"V{idx}:{avg_weights[idx]:.4f}" for idx in top_indices])
                    print(f"[监控-CVDG] Step:{self.print_counter:<5} | Alpha(内生保留率):{a.item():.4f} | 核心驱动: {weight_msg} | 抑制极值: V{worst_idx}:{avg_weights[worst_idx]:.4f}")
                    
        return output

# ----------------- 4. MLP Backbone (通道独立特征提取) -----------------
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
        
        # 默认推荐架构: ablation_mode=2 (仅开门控为 SOTA，特殊有延迟图网络可开 mode=3 带上 SDA)
        self.ablation_mode = getattr(configs, 'ablation_mode', 2) 
        self.alignment_pos = getattr(configs, 'alignment_pos', 'raw')
        self.gate_type = getattr(configs, 'gate_type', 'variate')
        self.max_lag = getattr(configs, 'max_lag', 7)
        # 获取先验 Alpha 设定，未设置时默认 0.5
        self.initial_alpha = getattr(configs, 'initial_alpha', 0.5)

        if self.configs.features == 'MS':
            self.Ne, self.Nx = 1, configs.enc_in - 1
        else:
            self.Ne, self.Nx = configs.enc_in, configs.enc_in

        self.revin = RevIN(configs.enc_in)
        self.value_embedding = nn.Linear(self.patch_len, configs.d_model)
        
        # 仅在 Raw 级别安全地开启物理 SDA
        if self.ablation_mode in [1, 3] and self.alignment_pos == 'raw':
            self.sda = VariateAwareSDA(configs.seq_len, self.max_lag, tau=getattr(configs, 'tau', 0.5))
            
        if self.ablation_mode in [2, 3]:
            self.cvdg = UnifiedCrossVariableGate(
                configs.d_model, self.Nx, self.Ne, 
                gate_type=self.gate_type, 
                initial_alpha=self.initial_alpha
            )
            
        self.encoder = nn.ModuleList([MLPBlock(self.patch_num * configs.d_model, configs.d_model) for _ in range(configs.e_layers)])
        self.head = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B = x_enc.shape[0]
        
        # Step 1: RevIN 归一化
        x_enc = self.revin(x_enc, mode='norm')
        
        # Step 2: 目标与驱动特征剥离
        if self.configs.features == 'MS':
            raw_endo = x_enc[:, :, -1:].transpose(1, 2)
            raw_exog = x_enc[:, :, :-1].transpose(1, 2)
        else:
            raw_endo = raw_exog = x_enc.transpose(1, 2)

        # Step 3: V-SDA 相位对齐 (避免均值塌缩)
        if self.ablation_mode in [1, 3] and self.alignment_pos == 'raw':
            raw_exog = self.sda(raw_endo, raw_exog)

        # Step 4: Patching & 升维
        z_endo = self.value_embedding(raw_endo.unfold(-1, self.patch_len, self.patch_len))
        z_exog = self.value_embedding(raw_exog.unfold(-1, self.patch_len, self.patch_len))

        # 机制退化展示 (Latent 模式下的错误示范，作为论文反面教材)
        if self.ablation_mode in [1, 3] and self.alignment_pos == 'latent':
            z_exog = z_exog + z_endo.mean(dim=1, keepdim=True) * 0.01 

        # Step 5: CVDG 振幅门控与因果路由
        if self.ablation_mode == 0: 
            x = z_endo
        elif self.ablation_mode == 1: 
            x = z_endo + z_exog.mean(dim=1, keepdim=True) if self.Ne == 1 else z_endo + z_exog
        else: 
            x = self.cvdg(z_endo, z_exog)

        # Step 6: 通道独立 (CI) MLP 主干预测
        x = x.reshape(-1, self.patch_num, self.configs.d_model)
        for layer in self.encoder: 
            x = layer(x)
        
        # Step 7: 降维与 RevIN 逆归一化
        res = self.head(x.reshape(B, self.Ne, -1)).transpose(1, 2)
        return self.revin(res, mode='denorm')
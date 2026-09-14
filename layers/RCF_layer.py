import torch
import torch.nn as nn

class RCFModule(nn.Module):
    def __init__(self, seq_len, pred_len, period):
        super(RCFModule, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = period
        
        # 学习周期性基准 (针对每个时间步在周期内的偏移)
        self.cycle_emb = nn.Parameter(torch.randn(period, 1))

    def forward(self, x):
        # x: [B, L, N]
        B, L, N = x.shape
        
        # 假设输入序列的最后一个点对应周期的某个位置（简化处理，实际可结合时间特征）
        # 这里提取当前序列对应的周期分量
        indices = torch.arange(L).to(x.device) % self.period
        P_enc = self.cycle_emb[indices].repeat(1, N) # [L, N]
        P_enc = P_enc.unsqueeze(0).repeat(B, 1, 1) # [B, L, N]
        
        # 提取预测区间对应的周期分量
        future_indices = torch.arange(L, L + self.pred_len).to(x.device) % self.period
        P_pred = self.cycle_emb[future_indices].repeat(1, N)
        P_pred = P_pred.unsqueeze(0).repeat(B, 1, 1) # [B, P, N]
        
        return P_enc, P_pred
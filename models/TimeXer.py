import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        # 关键修改：调整线性层维度，确保输出长度=target_window
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)  # [bs, nvars, d_model*patch_num]
        x = self.linear(x)   # [bs, nvars, target_window]
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.d_model = d_model  # 保存 d_model 为实例变量
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [bs, n_vars, seq_len]
        bs, n_vars, seq_len = x.shape
        # 全局token扩展
        glb = self.glb_token.repeat(bs, 1, 1, 1)  # [bs, n_vars, 1, d_model]
        
        # 关键修改：修复patch切分逻辑，避免维度错误
        # 确保seq_len能被patch_len整除（若不能则补零）
        if seq_len % self.patch_len != 0:
            pad_len = self.patch_len - (seq_len % self.patch_len)
            x = F.pad(x, (0, pad_len), mode='constant', value=0)
            seq_len += pad_len
        
        # 切分patch: [bs, n_vars, seq_len] -> [bs, n_vars, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        patch_num = x.shape[-2]
        
        # 维度调整 + Embedding
        x = x.reshape(bs * n_vars, patch_num, self.patch_len)  # [bs*nvars, patch_num, patch_len]
        x = self.value_embedding(x) + self.position_embedding(x)  # [bs*nvars, patch_num, d_model]
        x = x.reshape(bs, n_vars, patch_num, self.d_model)  # 使用 self.d_model 修复错误
        
        # 拼接全局token: [bs, n_vars, patch_num+1, d_model]
        x = torch.cat([x, glb], dim=2)
        x = x.reshape(bs * n_vars, patch_num + 1, self.d_model)  # 也使用 self.d_model
        
        return self.dropout(x), n_vars, patch_num + 1  # 新增返回patch_num+1


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        # Self-Attention
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # Cross-Attention (全局token)
        x_glb_ori = x[:, -1, :].unsqueeze(1)  # [B*nvars, 1, D]
        x_glb = x_glb_ori.reshape(B, -1, D)   # [B, nvars, D]
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
        )[0])  # [B, nvars, D]
        x_glb_attn = x_glb_attn.reshape(B * x_glb_attn.shape[1], 1, D)  # [B*nvars, 1, D]
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # 拼接全局token
        x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(np.ceil(configs.seq_len / configs.patch_len))  # 向上取整
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 关键修改：计算head的输入维度（patch_num+1包含全局token）
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 单变量预测逻辑（MS模式）
        if self.use_norm:
            # 归一化
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # 调整输入维度: [bs, seq_len, 1] -> [bs, 1, seq_len]
        x_enc_input = x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)  # [bs, 1, seq_len]
        # Embedding（新增patch_num返回值）
        en_embed, n_vars, patch_num = self.en_embedding(x_enc_input)
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)  # 外生变量Embedding

        # Encoder前向
        enc_out = self.encoder(en_embed, ex_embed)
        # 维度调整: [bs*nvars, patch_num+1, d_model] -> [bs, nvars, patch_num+1, d_model]
        enc_out = enc_out.reshape(-1, n_vars, patch_num, enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num+1]

        # 预测头输出: [bs, nvars, pred_len]
        dec_out = self.head(enc_out)
        # 维度调整: [bs, nvars, pred_len] -> [bs, pred_len, nvars]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # 反归一化
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out 

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 多变量预测逻辑（M模式）
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # 调整输入维度: [bs, seq_len, n_vars] -> [bs, n_vars, seq_len]
        x_enc_input = x_enc.permute(0, 2, 1)  # [bs, n_vars, seq_len]
        en_embed, n_vars, patch_num = self.en_embedding(x_enc_input)
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = enc_out.reshape(-1, n_vars, patch_num, enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out  

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 最终输出：直接返回dec_out（长度已等于pred_len）
            return dec_out
        else:
            return None
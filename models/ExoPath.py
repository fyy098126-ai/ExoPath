import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. RevIN and trend-residual decomposition
# =========================================================================
class RevIN(nn.Module):
    """Reversible instance normalization."""
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

class ExoGuidedEMA(nn.Module):
    """
    Exogenous-variation-guided EMA decomposition.

    Historical exogenous variation adjusts the smoothing rate used to form
    the endogenous trend. The remaining component is treated as residual.
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
        
        # Summarize historical exogenous variation for endogenous smoothing.
        if self.exo_dim > 0:
            x_exo = x[:, :, :-self.out_dim]
            exo_abs_change = torch.abs(x_exo[:, 1:, :] - x_exo[:, :-1, :])
            exo_variation = exo_abs_change.mean(dim=[1, 2])
            exo_variation_summary = exo_variation.unsqueeze(1).detach()  # [B, 1]
            smoothing_logits = (
                self.base_alpha_logits.unsqueeze(0)
                - self.vol_sensitivity * exo_variation_summary
            )
        else:
            smoothing_logits = self.base_alpha_logits.unsqueeze(0).repeat(B, 1)
            
        smoothing_rate = torch.sigmoid(smoothing_logits).to(torch.double)  # [B, out_dim]

        powers = torch.flip(torch.arange(T, dtype=torch.double, device=device), dims=(0,)).unsqueeze(1)
        weights = torch.pow((1.0 - smoothing_rate).unsqueeze(1), powers.unsqueeze(0))
        divisor = weights.clone()
        
        w_0 = weights[:, 0:1, :]
        w_rest = weights[:, 1:, :] * smoothing_rate.unsqueeze(1)
        weights_mod = torch.cat([w_0, w_rest], dim=1) 

        x_endo = x[:, :, -self.out_dim:].to(torch.double)
        out_endo = torch.cumsum(x_endo * weights_mod, dim=1) / divisor
        endogenous_trend = out_endo.to(torch.float32)

        # Extract exogenous residual context with a constant smoothing rate.
        if self.exo_dim > 0:
            exogenous_smoothing_rate = torch.full(
                (1, self.exo_dim), 0.5, dtype=torch.double, device=device
            )
            w_exo = torch.pow(
                (1.0 - exogenous_smoothing_rate).unsqueeze(1),
                powers.unsqueeze(0),
            )
            div_exo = w_exo.clone()
            w_0_exo = w_exo[:, 0:1, :]
            w_rest_exo = w_exo[:, 1:, :] * exogenous_smoothing_rate.unsqueeze(1)
            w_mod_exo = torch.cat([w_0_exo, w_rest_exo], dim=1)
            x_exo_d = x_exo.to(torch.double)
            out_exo = torch.cumsum(x_exo_d * w_mod_exo, dim=1) / div_exo
            exogenous_trend = out_exo.to(torch.float32)
            trend_all = torch.cat([exogenous_trend, endogenous_trend], dim=-1)
        else:
            trend_all = endogenous_trend

        residual_all = x - trend_all
        return trend_all, residual_all


class ChannelIndependentTrendExtrapolator(nn.Module):
    """
    Channel-independent trend extrapolator.

    Each endogenous trend channel is mapped independently along the temporal
    axis, with average pooling used to emphasize smooth evolution.
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        # Expand before pooling so the temporal representation retains d_model features.
        self.lin1 = nn.Linear(seq_len, d_model * 2)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.norm1 = nn.LayerNorm(d_model)
        self.lin2 = nn.Linear(d_model, d_model)

    def forward(self, trend_all, out_dim):
        endogenous_trend = trend_all[:, :, -out_dim:]
        
        # Map each endogenous channel independently along the temporal dimension.
        trend_features = endogenous_trend.transpose(1, 2)  # [B, out_dim, T]
        
        trend_features = self.lin1(trend_features)   # [B, out_dim, 2 * d_model]
        trend_features = self.pool1(trend_features)  # [B, out_dim, d_model]
        trend_features = self.norm1(trend_features)  # [B, out_dim, d_model]
        trend_features = self.lin2(trend_features)   # [B, out_dim, d_model]
        
        return trend_features


# =========================================================================
# 2. Low-rank context routing
# =========================================================================
class LRER(nn.Module):
    """
    Generate a contextual routing mask through a low-rank bottleneck.
    """
    def __init__(self, c_in, rank=16, dropout=0.1):
        super().__init__()
        self.rank = min(rank, c_in)
        self.down = nn.Linear(c_in, self.rank, bias=False)
        self.up = nn.Linear(self.rank, c_in, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, d_model, c_in_router]
        routing_logits = self.up(self.dropout(F.relu(self.down(x))))
        return torch.sigmoid(routing_logits)


# =========================================================================
# 3. ExoPath: temporal-component-aware asymmetric routing
# =========================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        # Dimensions and forecasting setting.
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

        # Normalization and component formation.
        self.revin = RevIN(self.C)
        self.decomp = ExoGuidedEMA(c_in=self.C, out_dim=self.out_dim, features=self.features)
        
        # Trend path.
        self.trend_extrapolator = ChannelIndependentTrendExtrapolator(self.seq_len, self.d_model)
        self.trend_head = nn.Linear(self.d_model, self.pred_len)


        # Residual path. Registered attribute names are retained for checkpoint compatibility.
        self.seasonal_proj = nn.Linear(self.seq_len, self.d_model)
        
        # Endogenous Query Gate: jointly gate the residual carrier and query token.
        self.ebt_token = nn.Parameter(torch.ones(1, self.out_dim, self.d_model))
        self.ebg = nn.Sequential(
            nn.Linear(2 * self.d_model, t_ff), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(t_ff, 2 * self.d_model), nn.Sigmoid()
        )
        
        # Low-rank context router over context entries and conditioned queries.
        self.c_in_router = (self.C + self.out_dim) if self.features == 'M' else (self.exo_dim + self.out_dim)
        self.exo_router = LRER(self.c_in_router, rank=self.rank, dropout=drop)
        
        # Residual head after feature-axis fusion.
        self.seasonal_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(2 * self.d_model, self.pred_len),
        )


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        b = x_enc.shape[0]

        # 1. Instance normalization.
        x = self.revin(x_enc, 'norm') 

        # 2. Exogenous-variation-guided trend-residual decomposition.
        trend_all, residual_all = self.decomp(x)

        
        # The trend path retains endogenous channels only.
        trend_features = self.trend_extrapolator(trend_all, self.out_dim)
            
        trend_forecast = self.trend_head(trend_features).transpose(1, 2)

        # 3. Residual-specific contextual routing.
        projected_residuals = self.seasonal_proj(
            residual_all.permute(0, 2, 1)
        )  # [B, C, d_model]
        endogenous_residual = projected_residuals[:, -self.out_dim:, :]
        context_residuals = (
            projected_residuals[:, :-self.out_dim, :]
            if self.features == 'MS'
            else projected_residuals
        )

        # 3a. Endogenous query gating.
        learnable_query = self.ebt_token.repeat(b, 1, 1)
        query_gate_input = torch.cat(
            [endogenous_residual, learnable_query], dim=-1
        )  # [B, out_dim, 2 * d_model]
        gated_query_pair = query_gate_input * self.ebg(query_gate_input)
        endogenous_residual_carrier = gated_query_pair[:, :, :self.d_model]
        conditioned_query = gated_query_pair[:, :, self.d_model:]

        # 3b. Concatenate context residuals and conditioned queries by variate.
        joint_context = torch.cat(
            [context_residuals, conditioned_query], dim=1
        )  # [B, c_in_router, d_model]

        # Generate a routing mask through the context-axis low-rank bottleneck.
        routing_mask = self.exo_router(
            joint_context.transpose(1, 2)
        ).transpose(1, 2)
                
        routed_context = joint_context * routing_mask
            
        # Only the query rows continue to the prediction head.
        context_refined_query = routed_context[:, -self.out_dim:, :]



        # 3c. Feature-axis fusion and residual forecasting.
        residual_head_input = torch.cat(
            [endogenous_residual_carrier, context_refined_query], dim=-1
        )  # [B, out_dim, 2 * d_model]
        
        residual_forecast = self.seasonal_head(
            residual_head_input
        ).transpose(1, 2)  # [B, pred_len, out_dim]

        # 4. Component forecast fusion.
        normalized_forecast = trend_forecast + residual_forecast

        # 5. Restore the original scale.
        forecast = self.revin(
            normalized_forecast,
            'denorm',
            target_dim=self.out_dim if self.features == 'MS' else None,
        )
            
        return forecast

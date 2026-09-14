import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. RevIN
# =========================================================================
class RevIN(nn.Module):
    """Reversible Instance Normalization."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode="norm", target_dim=None):
        if mode == "norm":
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x

        mean = self.mean[:, :, -target_dim:] if target_dim else self.mean
        stdev = self.stdev[:, :, -target_dim:] if target_dim else self.stdev
        if self.affine:
            weight = (
                self.affine_weight[-target_dim:]
                if target_dim
                else self.affine_weight
            )
            bias = (
                self.affine_bias[-target_dim:]
                if target_dim
                else self.affine_bias
            )
            x = (x - bias) / weight
        return x * stdev + mean


# =========================================================================
# 2. Exogenous-guided one-sided EMA decomposition
# =========================================================================
class ExoGuidedEMA(nn.Module):
    """The Full forward path is copied from the original XGate_Omni code."""

    def __init__(self, c_in, out_dim, features="M"):
        super().__init__()
        self.out_dim = out_dim
        self.exo_dim = c_in - out_dim if features == "MS" else 0
        self.base_alpha_logits = nn.Parameter(torch.zeros(out_dim))
        if self.exo_dim > 0:
            self.vol_sensitivity = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x):
        """Original Full decomposition path; do not refactor this method."""
        B, T, C = x.shape
        device = x.device

        if self.exo_dim > 0:
            x_exo = x[:, :, :-self.out_dim]
            exo_diff = torch.abs(
                x_exo[:, 1:, :] - x_exo[:, :-1, :]
            )
            exo_vol = exo_diff.mean(dim=[1, 2])
            exo_vol_prior = exo_vol.unsqueeze(1).detach()
            alpha_logits = (
                self.base_alpha_logits.unsqueeze(0)
                - self.vol_sensitivity * exo_vol_prior
            )
        else:
            alpha_logits = self.base_alpha_logits.unsqueeze(0).repeat(B, 1)

        alpha = torch.sigmoid(alpha_logits).to(torch.double)

        powers = torch.flip(
            torch.arange(T, dtype=torch.double, device=device),
            dims=(0,),
        ).unsqueeze(1)
        weights = torch.pow(
            (1.0 - alpha).unsqueeze(1),
            powers.unsqueeze(0),
        )
        divisor = weights.clone()

        w_0 = weights[:, 0:1, :]
        w_rest = weights[:, 1:, :] * alpha.unsqueeze(1)
        weights_mod = torch.cat([w_0, w_rest], dim=1)

        x_endo = x[:, :, -self.out_dim:].to(torch.double)
        out_endo = (
            torch.cumsum(x_endo * weights_mod, dim=1) / divisor
        )
        trend_endo = out_endo.to(torch.float32)

        if self.exo_dim > 0:
            alpha_exo = torch.full(
                (1, self.exo_dim),
                0.5,
                dtype=torch.double,
                device=device,
            )
            w_exo = torch.pow(
                (1.0 - alpha_exo).unsqueeze(1),
                powers.unsqueeze(0),
            )
            div_exo = w_exo.clone()
            w_0_exo = w_exo[:, 0:1, :]
            w_rest_exo = (
                w_exo[:, 1:, :] * alpha_exo.unsqueeze(1)
            )
            w_mod_exo = torch.cat(
                [w_0_exo, w_rest_exo],
                dim=1,
            )
            x_exo_d = x_exo.to(torch.double)
            out_exo = (
                torch.cumsum(x_exo_d * w_mod_exo, dim=1)
                / div_exo
            )
            trend_exo = out_exo.to(torch.float32)
            trend_all = torch.cat(
                [trend_exo, trend_endo],
                dim=-1,
            )
        else:
            trend_all = trend_endo

        seasonality = x - trend_all
        return trend_all, seasonality

    def forward_fixed(self, x):
        """Ablation: keep EMA but remove exogenous-volatility guidance.

        This method is separate from ``forward`` so that the Full path remains
        byte-for-byte equivalent in its numerical operation order.
        """
        B, T, C = x.shape
        device = x.device
        x_exo = (
            x[:, :, :-self.out_dim]
            if self.exo_dim > 0
            else None
        )

        alpha_logits = (
            self.base_alpha_logits.unsqueeze(0).repeat(B, 1)
        )
        alpha = torch.sigmoid(alpha_logits).to(torch.double)

        powers = torch.flip(
            torch.arange(T, dtype=torch.double, device=device),
            dims=(0,),
        ).unsqueeze(1)
        weights = torch.pow(
            (1.0 - alpha).unsqueeze(1),
            powers.unsqueeze(0),
        )
        divisor = weights.clone()

        w_0 = weights[:, 0:1, :]
        w_rest = weights[:, 1:, :] * alpha.unsqueeze(1)
        weights_mod = torch.cat([w_0, w_rest], dim=1)

        x_endo = x[:, :, -self.out_dim:].to(torch.double)
        out_endo = (
            torch.cumsum(x_endo * weights_mod, dim=1) / divisor
        )
        trend_endo = out_endo.to(torch.float32)

        if self.exo_dim > 0:
            alpha_exo = torch.full(
                (1, self.exo_dim),
                0.5,
                dtype=torch.double,
                device=device,
            )
            w_exo = torch.pow(
                (1.0 - alpha_exo).unsqueeze(1),
                powers.unsqueeze(0),
            )
            div_exo = w_exo.clone()
            w_0_exo = w_exo[:, 0:1, :]
            w_rest_exo = (
                w_exo[:, 1:, :] * alpha_exo.unsqueeze(1)
            )
            w_mod_exo = torch.cat(
                [w_0_exo, w_rest_exo],
                dim=1,
            )
            out_exo = (
                torch.cumsum(
                    x_exo.to(torch.double) * w_mod_exo,
                    dim=1,
                )
                / div_exo
            )
            trend_exo = out_exo.to(torch.float32)
            trend_all = torch.cat(
                [trend_exo, trend_endo],
                dim=-1,
            )
        else:
            trend_all = trend_endo

        return trend_all, x - trend_all



class ChannelIndependentTrendExtrapolator(nn.Module):

    def __init__(self, seq_len, d_model):
        super().__init__()
        self.lin1 = nn.Linear(seq_len, d_model * 2)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.norm1 = nn.LayerNorm(d_model)
        self.lin2 = nn.Linear(d_model, d_model)

    def forward(self, trend_all, out_dim, mode="full"):

        endo_trend = trend_all[:, :, -out_dim:]

        x = endo_trend.transpose(1, 2)
        x = self.lin1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.lin2(x)
        return x


# =========================================================================
# 4. LRER context router
# =========================================================================
class LRER(nn.Module):
    """Low-rank context router."""

    def __init__(self, c_in, rank=16, dropout=0.1):
        super().__init__()
        self.rank = min(rank, c_in)
        self.down = nn.Linear(c_in, self.rank, bias=False)
        self.up = nn.Linear(self.rank, c_in, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate_logits = self.up(
            self.dropout(F.relu(self.down(x)))
        )
        return torch.sigmoid(gate_logits)


# =========================================================================
# 5. ExoPath exact-Full ablation model
# =========================================================================
class Model(nn.Module):
    """Ablation model whose ``full`` path follows the original code exactly.

    Modes
    -----
    full
        Original complete architecture.
    fixed_ema
        Retain the same one-sided EMA but remove exogenous-volatility guidance.
        Only meaningful in MS mode.
    wo_decomp
        Remove explicit trend/residual decomposition; both paths receive the
        normalized input.
    wo_ebg
        Remove EBG; the raw target residual embedding serves as native carrier
        and routing query.
    wo_router
        Remove LRER; the target query passes without context-conditioned gating.
    trend_only_route
        The learned target-side context gate acts only on the trend path.
    symmetric_route
        The same learned gate acts on both trend and residual paths.
    wo_origin
        Remove the target-native residual carrier while keeping the seasonal
        head input dimension unchanged.
    """

    VALID_MODES = {
        "full",
        "fixed_ema",
        "wo_decomp",
        "wo_ebg",
        "wo_router",
        "trend_only_route",
        "symmetric_route",
        "wo_origin",
    }

    def __init__(self, configs):
        super().__init__()
        self.ablation_mode = getattr(
            configs,
            "ablation_mode",
            "full",
        )
        if self.ablation_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown ablation_mode={self.ablation_mode!r}; "
                f"choose from {sorted(self.VALID_MODES)}"
            )

        self.C = configs.enc_in
        self.features = getattr(configs, "features", "M")
        self.out_dim = (
            1 if self.features == "MS" else self.C
        )
        self.exo_dim = (
            self.C - self.out_dim
            if self.features == "MS"
            else 0
        )

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        t_ff = (
            getattr(configs, "t_ff", 0)
            or self.d_model * 4
        )
        self.rank = (
            getattr(configs, "num_groups", 0)
            or (
                16
                if self.C >= 321
                else 8
                if self.C >= 21
                else 4
            )
        )
        drop = getattr(configs, "dropout", 0.1)

        # Module creation order matches the original Full model.
        self.revin = RevIN(self.C)
        self.decomp = ExoGuidedEMA(
            c_in=self.C,
            out_dim=self.out_dim,
            features=self.features,
        )

        self.trend_extrapolator = ChannelIndependentTrendExtrapolator(
            self.seq_len, self.d_model
        )
        self.trend_head = nn.Linear(
            self.d_model,
            self.pred_len,
        )
        # Original Full code contains this Identity call.
        self.ablation_trend_act = nn.Identity()

        self.seasonal_proj = nn.Linear(
            self.seq_len,
            self.d_model,
        )
        self.ebt_token = nn.Parameter(
            torch.ones(
                1,
                self.out_dim,
                self.d_model,
            )
        )
        self.ebg = nn.Sequential(
            nn.Linear(
                2 * self.d_model,
                t_ff,
            ),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(
                t_ff,
                2 * self.d_model,
            ),
            nn.Sigmoid(),
        )

        self.c_in_router = (
            self.C + self.out_dim
            if self.features == "M"
            else self.exo_dim + self.out_dim
        )
        self.exo_router = LRER(
            self.c_in_router,
            rank=self.rank,
            dropout=drop,
        )

        self.seasonal_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(
                2 * self.d_model,
                self.pred_len,
            ),
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        mode = self.ablation_mode
        b = x_enc.shape[0]

        # 1. RevIN: unchanged from Full.
        x = self.revin(x_enc, "norm")

        # 2. Decomposition.
        if mode == "wo_decomp":
            seasonal_all = x
            trend_all = x
        elif mode == "fixed_ema":
            trend_all, seasonal_all = (
                self.decomp.forward_fixed(x)
            )
        else:
            # Full calls the original decomposition method directly.
            trend_all, seasonal_all = self.decomp(x)

        # 3. Trend path.
        z_trend = self.trend_extrapolator(trend_all, self.out_dim)
        z_trend = self.ablation_trend_act(z_trend)

        route_to_trend = mode in {
            "trend_only_route",
            "symmetric_route",
        }

        # In Full and all non-trend-routing modes, execute trend_head at the
        # exact original location, before the residual/context path.
        if not route_to_trend:
            trend_pred = self.trend_head(
                z_trend
            ).transpose(1, 2)

        # 4. Residual/context path.
        z_seas = self.seasonal_proj(
            seasonal_all.permute(0, 2, 1)
        )
        z_endo = z_seas[:, -self.out_dim:, :]
        z_exo = (
            z_seas[:, :-self.out_dim, :]
            if self.features == "MS"
            else z_seas
        )

        # 4A. EBG.
        if mode == "wo_ebg":
            origin_gated = z_endo
            ebt_gated = z_endo
        else:
            glob = self.ebt_token.repeat(b, 1, 1)
            en_emb = torch.cat(
                [z_endo, glob],
                dim=-1,
            )
            en_gated = en_emb * self.ebg(en_emb)
            origin_gated = en_gated[
                :,
                :,
                :self.d_model,
            ]
            ebt_gated = en_gated[
                :,
                :,
                self.d_model:,
            ]

        # 4B. LRER.
        if mode == "wo_router":
            refined_ebt = ebt_gated
            target_gate = None
        else:
            # Full preserves the original sequence:
            # concatenate -> obtain complete gate -> multiply complete map ->
            # slice the routed target branch.
            ex_emb = torch.cat(
                [z_exo, ebt_gated],
                dim=1,
            )
            gate = self.exo_router(
                ex_emb.transpose(1, 2)
            ).transpose(1, 2)
            gated_ex_emb = ex_emb * gate
            refined_ebt = gated_ex_emb[
                :,
                -self.out_dim:,
                :,
            ]

            if route_to_trend:
                target_gate = gate[
                    :,
                    -self.out_dim:,
                    :,
                ]
            else:
                target_gate = None

        # 4C. Routing-location counterfactuals.
        if route_to_trend:
            if target_gate is None:
                raise RuntimeError(
                    f"{mode} requires LRER target gate"
                )
            z_trend = z_trend * target_gate
            trend_pred = self.trend_head(
                z_trend
            ).transpose(1, 2)

            if mode == "trend_only_route":
                # Context affects only trend; residual query is unconditioned.
                refined_ebt = ebt_gated
            # symmetric_route keeps the routed residual branch.

        # 4D. Target-native carrier ablation.
        if mode == "wo_origin":
            # Zeroing truly removes the native carrier while preserving the
            # seasonal head's input dimensionality.
            origin_gated = torch.zeros_like(
                origin_gated
            )

        seas_in = torch.cat(
            [origin_gated, refined_ebt],
            dim=-1,
        )
        seasonal_pred = self.seasonal_head(
            seas_in
        ).transpose(1, 2)

        # 5. Original additive fusion.
        res_out = trend_pred + seasonal_pred

        # 6. Original RevIN inverse.
        res_out = self.revin(
            res_out,
            "denorm",
            target_dim=(
                self.out_dim
                if self.features == "MS"
                else None
            ),
        )
        return res_out

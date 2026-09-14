"""Exploratory global-local routing variant of ExoPath.

This file intentionally leaves ``models/ExoPath.py`` unchanged.  The only
architectural difference is the residual-path router:

* a sample-shared low-rank operator represents stable routing structure;
* the original low-rank branch supplies a window-specific correction.

The variant is meant for controlled screening, not as a replacement for the
paper's main model before empirical validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ExoPath import Model as ExoPathModel


class GlobalLocalLRER(nn.Module):
    """Low-rank router with shared structure and instance-specific evidence.

    Parameters
    ----------
    c_in:
        Number of entries in the residual routing map.
    d_model:
        Latent feature dimension of every entry.
    local_rank:
        Bottleneck rank of the window-specific branch.
    global_rank:
        Rank of the sample-shared routing template.
    dropout:
        Dropout applied only to the window-specific branch.
    """

    def __init__(
        self,
        c_in,
        d_model,
        local_rank=16,
        global_rank=4,
        dropout=0.1,
        branch_mode="both",
    ):
        super().__init__()
        valid_modes = {"both", "shared_only", "local_only"}
        if branch_mode not in valid_modes:
            raise ValueError(
                f"Unknown branch_mode={branch_mode!r}; "
                f"choose from {sorted(valid_modes)}"
            )
        self.c_in = c_in
        self.d_model = d_model
        self.local_rank = min(local_rank, c_in)
        self.global_rank = min(global_rank, c_in, d_model)
        self.branch_mode = branch_mode

        # Dataset-level linear relation operator. Its weights are shared by all
        # windows, while its output is evaluated on the current context.
        self.global_down = nn.Linear(
            c_in,
            self.global_rank,
            bias=False,
        )
        self.global_up = nn.Linear(
            self.global_rank,
            c_in,
            bias=False,
        )

        # Window-specific correction; this matches the role of the original
        # LRER branch and preserves its low-rank channel bottleneck.
        self.local_down = nn.Linear(
            c_in,
            self.local_rank,
            bias=False,
        )
        self.local_up = nn.Linear(
            self.local_rank,
            c_in,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

        # Separate bounded strengths make the contribution of each branch
        # directly inspectable without allowing unbounded scalar rescaling.
        self.global_strength_logit = nn.Parameter(torch.tensor(0.0))
        self.local_strength_logit = nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.global_down.weight,
            a=5 ** 0.5,
        )
        nn.init.kaiming_uniform_(
            self.global_up.weight,
            a=5 ** 0.5,
        )
        nn.init.kaiming_uniform_(
            self.local_down.weight,
            a=5 ** 0.5,
        )
        nn.init.kaiming_uniform_(
            self.local_up.weight,
            a=5 ** 0.5,
        )

    def routing_logits(self, x):
        """Return shared and window-specific logits for diagnostics."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected [batch, d_model, channels], got {tuple(x.shape)}"
            )
        if x.shape[1:] != (self.d_model, self.c_in):
            raise ValueError(
                "Router input shape mismatch: expected "
                f"(*, {self.d_model}, {self.c_in}), got {tuple(x.shape)}"
            )

        shared = self.global_up(self.global_down(x))

        local = self.local_up(
            self.dropout(F.relu(self.local_down(x)))
        )
        return shared, local

    def shared_operator(self):
        """Return the explicit sample-shared channel relation matrix."""
        return self.global_up.weight @ self.global_down.weight

    def forward(self, x):
        shared, local = self.routing_logits(x)
        shared_weight = 2.0 * torch.sigmoid(
            self.global_strength_logit
        )
        local_weight = 2.0 * torch.sigmoid(
            self.local_strength_logit
        )
        if self.branch_mode == "shared_only":
            local_weight = 0.0
        elif self.branch_mode == "local_only":
            shared_weight = 0.0
        return torch.sigmoid(
            shared_weight * shared + local_weight * local
        )


class Model(ExoPathModel):
    """ExoPath with a global-local residual router."""

    def __init__(self, configs):
        super().__init__(configs)
        global_rank = getattr(configs, "router_prior_rank", 4)
        branch_mode = getattr(
            configs,
            "router_branch_mode",
            "both",
        )
        self.exo_router = GlobalLocalLRER(
            c_in=self.c_in_router,
            d_model=self.d_model,
            local_rank=self.rank,
            global_rank=global_rank,
            dropout=getattr(configs, "dropout", 0.1),
            branch_mode=branch_mode,
        )

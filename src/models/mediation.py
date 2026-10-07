"""Chapter 3.3 Mediating-effect models (Equations 4 and 5).

Eq.4  M_ict = α0 + α1 (treat_i × time_t) + α2 X_ict + μ_c + θ_t + ε_ict
      M ∈ {HI, EF}

Eq.5  Y_ict = γ0 + γ1 (treat_i × time_t) + γ2 M_ict + γ3 X_ict + μ_c + θ_t + ε_ict

Table 8 reports the first-stage behavioral channels (Eq. 4) together with the
baseline CES-D DID. A complete mediation claim additionally uses Eq. 5 and
bootstrap confidence intervals for the indirect effect.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.time_varying_did import TimeVaryingDID


class MediationFirstStage(TimeVaryingDID):
    """Equation (4): PE reform → mediator (HI or EF)."""

    pass


class MediationSecondStage(nn.Module):
    """Equation (5): PE reform + mediator → CES-D."""

    def __init__(
        self,
        n_controls: int,
        n_colleges: int,
        n_times: int,
        n_students: int,
        n_mediators: int = 1,
        use_controls: bool = True,
        use_time_fe: bool = True,
        use_individual_fe: bool = True,
    ):
        super().__init__()
        self.use_controls = use_controls
        self.beta0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))  # remaining DID (direct) effect
        self.gamma2 = nn.Linear(n_mediators, 1, bias=False)
        nn.init.zeros_(self.gamma2.weight)
        self.control_layer = nn.Linear(n_controls, 1, bias=False) if use_controls else None
        if self.control_layer is not None:
            nn.init.zeros_(self.control_layer.weight)
        self.time_fe = nn.Embedding(n_times, 1) if use_time_fe else None
        self.individual_fe = nn.Embedding(n_students, 1) if use_individual_fe else None
        for emb in (self.time_fe, self.individual_fe):
            if emb is not None:
                nn.init.zeros_(emb.weight)

    def forward(
        self,
        did: torch.Tensor,
        M: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        wave: Optional[torch.Tensor] = None,
        student_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        did = did.view(-1, 1)
        if M.ndim == 1:
            M = M.view(-1, 1)
        y = self.beta0 + self.gamma1 * did + self.gamma2(M)
        if self.control_layer is not None and X is not None:
            y = y + self.control_layer(X)
        if self.time_fe is not None and wave is not None:
            y = y + self.time_fe(wave.view(-1))
        if self.individual_fe is not None and student_idx is not None:
            y = y + self.individual_fe(student_idx.view(-1))
        return y.view(-1)

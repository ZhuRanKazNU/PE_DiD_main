"""Chapter 3.1 Time-varying DID (Equation 1).

Y_ict = β0 + β1 (treat_i × time_t) + β2 X_ict + μ_c + θ_t + ε_ict

College (or individual) and time fixed effects are implemented as embeddings.
Four nested specifications match Table 3:
  (1) DID only
  (2) DID + controls
  (3) DID + individual FE + time FE
  (4) DID + controls + two-way FE
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TimeVaryingDID(nn.Module):
    """PyTorch module for the time-varying / multi-period DID outcome model."""

    def __init__(
        self,
        n_controls: int,
        n_colleges: int,
        n_times: int,
        n_students: int = 0,
        use_controls: bool = True,
        use_college_fe: bool = False,
        use_time_fe: bool = False,
        use_individual_fe: bool = False,
    ):
        super().__init__()
        self.use_controls = use_controls and n_controls > 0
        self.use_college_fe = use_college_fe
        self.use_time_fe = use_time_fe
        self.use_individual_fe = use_individual_fe

        self.beta0 = nn.Parameter(torch.zeros(1))
        self.beta1 = nn.Parameter(torch.zeros(1))  # DID coefficient of interest
        self.control_layer = nn.Linear(n_controls, 1, bias=False) if self.use_controls else None
        if self.control_layer is not None:
            nn.init.zeros_(self.control_layer.weight)

        self.college_fe = nn.Embedding(n_colleges, 1) if use_college_fe else None
        self.time_fe = nn.Embedding(n_times, 1) if use_time_fe else None
        self.individual_fe = nn.Embedding(n_students, 1) if use_individual_fe and n_students > 0 else None
        for emb in (self.college_fe, self.time_fe, self.individual_fe):
            if emb is not None:
                nn.init.zeros_(emb.weight)

    def forward(
        self,
        did: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        college_id: Optional[torch.Tensor] = None,
        wave: Optional[torch.Tensor] = None,
        student_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        did = did.view(-1, 1)
        y = self.beta0 + self.beta1 * did
        if self.control_layer is not None:
            if X is None:
                raise ValueError("Controls were enabled but X is None.")
            y = y + self.control_layer(X)
        if self.college_fe is not None:
            y = y + self.college_fe(college_id.view(-1))
        if self.time_fe is not None:
            y = y + self.time_fe(wave.view(-1))
        if self.individual_fe is not None:
            y = y + self.individual_fe(student_idx.view(-1))
        return y.view(-1)

    def did_coefficient(self) -> float:
        return float(self.beta1.detach().cpu().item())


def build_baseline_specs(n_controls: int, n_colleges: int, n_times: int, n_students: int) -> dict:
    """Table 3 nested specifications."""
    return {
        "model1": TimeVaryingDID(n_controls, n_colleges, n_times, n_students, False, False, False, False),
        "model2": TimeVaryingDID(n_controls, n_colleges, n_times, n_students, True, False, False, False),
        "model3": TimeVaryingDID(n_controls, n_colleges, n_times, n_students, False, False, True, True),
        "model4": TimeVaryingDID(n_controls, n_colleges, n_times, n_students, True, False, True, True),
        "eq1_college_fe": TimeVaryingDID(n_controls, n_colleges, n_times, n_students, True, True, True, False),
    }

"""Overall PE-DiD network assembling Chapter 3 modules.

Workflow (Figure 1):
  questionnaire → variables (3.4) → PSM logit (3.2 / Eq.2)
                → time-varying DID (3.1 / Eq.1)
                → PSM-DID (3.2 / Eq.3)
                → mediation (3.3 / Eq.4-5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.mediation import MediationFirstStage, MediationSecondStage
from src.models.psm_did import PSMDID
from src.models.psm_logit import PSMLogit
from src.models.time_varying_did import TimeVaryingDID


@dataclass
class NetworkSpec:
    n_controls: int
    n_colleges: int
    n_times: int
    n_students: int


class PEDiDNetwork(nn.Module):
    """Composite network with independently trained submodules (sequential estimation)."""

    def __init__(self, spec: NetworkSpec):
        super().__init__()
        self.spec = spec
        n, c, t, s = spec.n_controls, spec.n_colleges, spec.n_times, spec.n_students
        self.psm = PSMLogit(n)
        # Eq.1 / Table 3 column (4): controls + college FE + time FE
        self.did = TimeVaryingDID(n, c, t, s, True, True, True, False)
        self.psm_did = PSMDID(n, c, t, s, True, True, True, False)
        self.med_hi = MediationFirstStage(n, c, t, s, True, True, True, False)
        self.med_ef = MediationFirstStage(n, c, t, s, True, True, True, False)
        self.med_y = MediationSecondStage(n, c, t, s, n_mediators=2, use_controls=True)

    def propensity(self, X: torch.Tensor) -> torch.Tensor:
        return self.psm(X)

    def predict_cesd(
        self,
        did: torch.Tensor,
        X: torch.Tensor,
        wave: torch.Tensor,
        student_idx: torch.Tensor,
        college_id: Optional[torch.Tensor] = None,
        matched: bool = False,
    ) -> torch.Tensor:
        model = self.psm_did if matched else self.did
        return model(did=did, X=X, college_id=college_id, wave=wave, student_idx=student_idx)

    def predict_mediators(
        self,
        did: torch.Tensor,
        X: torch.Tensor,
        wave: torch.Tensor,
        student_idx: torch.Tensor,
        college_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        kw = dict(did=did, X=X, college_id=college_id, wave=wave, student_idx=student_idx)
        return {"HI": self.med_hi(**kw), "EF": self.med_ef(**kw)}

    def counterfactual_cesd(
        self,
        X: torch.Tensor,
        wave: torch.Tensor,
        student_idx: torch.Tensor,
        college_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference: CES-D with DID=0 vs DID=1 holding covariates and FE fixed."""
        zeros = torch.zeros(X.size(0), device=X.device)
        ones = torch.ones(X.size(0), device=X.device)
        y0 = self.predict_cesd(zeros, X, wave, student_idx, college_id, matched=False)
        y1 = self.predict_cesd(ones, X, wave, student_idx, college_id, matched=False)
        return {"y0": y0, "y1": y1, "att": y1 - y0}

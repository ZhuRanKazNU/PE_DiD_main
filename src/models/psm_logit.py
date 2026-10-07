"""Chapter 3.2 PSM Logit (Equation 2).

P_i = Pr(treat_i = 1 | X_i) = Λ(X_i' δ)

Λ(·) is the logistic (sigmoid) function. After scores are estimated, 1:1
nearest-neighbor matching with a caliper is applied on the student-level
baseline covariates listed in Table 1 / Table 4.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from src.data.schema import CONTROLS


class PSMLogit(nn.Module):
    """Logistic propensity-score network: P(treat=1 | X) = sigmoid(δ0 + X'δ)."""

    def __init__(self, n_covariates: int):
        super().__init__()
        self.linear = nn.Linear(n_covariates, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(X)).view(-1)

    def logits(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).view(-1)


def student_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """One row per student using first-wave covariates (matching is student-level)."""
    first = df.sort_values(["student_idx", "wave"]).groupby("student_idx", as_index=False).first()
    return first[["student_idx", "treat", "college_id"] + CONTROLS].reset_index(drop=True)


def match_students(
    baseline: pd.DataFrame,
    scores: np.ndarray,
    caliper: float = 0.05,
    n_neighbors: int = 1,
    replacement: bool = False,
    target_matched_obs: int | None = None,
    waves_per_student: int = 2,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """1:1 NN matching on the propensity score with a caliper.

    Returns matched student_idx values and a pair table.
    """
    base = baseline.copy()
    base["ps"] = scores
    treat = base[base["treat"] == 1].reset_index(drop=True)
    ctrl = base[base["treat"] == 0].reset_index(drop=True)
    if len(treat) == 0 or len(ctrl) == 0:
        return np.array([], dtype=int), pd.DataFrame()

    ps_std = float(base["ps"].std()) or 1.0
    radius = max(float(caliper), 0.25 * ps_std)

    nn = NearestNeighbors(n_neighbors=min(12, len(ctrl)), metric="euclidean")
    nn.fit(ctrl[["ps"]].to_numpy())
    dist, idx = nn.kneighbors(treat[["ps"]].to_numpy())

    used = set()
    pairs = []
    order = np.argsort(dist[:, 0])  # match closest first
    for ti in order:
        chosen = None
        for k in range(dist.shape[1]):
            if dist[ti, k] > radius:
                break
            cj = int(idx[ti, k])
            if (not replacement) and cj in used:
                continue
            chosen = (cj, float(dist[ti, k]))
            break
        if chosen is None:
            continue
        cj, d = chosen
        used.add(cj)
        pairs.append(
            {
                "treat_student": int(treat.loc[ti, "student_idx"]),
                "control_student": int(ctrl.loc[cj, "student_idx"]),
                "ps_treat": float(treat.loc[ti, "ps"]),
                "ps_control": float(ctrl.loc[cj, "ps"]),
                "ps_diff": d,
            }
        )

    pair_df = pd.DataFrame(pairs)
    if target_matched_obs is not None and len(pair_df) > 0:
        n_pairs_target = max(1, int(round(target_matched_obs / float(waves_per_student) / 2.0)))
        if len(pair_df) > n_pairs_target:
            pair_df = pair_df.nsmallest(n_pairs_target, "ps_diff").reset_index(drop=True)

    if pair_df.empty:
        matched = np.array([], dtype=int)
    else:
        matched = np.unique(
            np.concatenate([pair_df["treat_student"].to_numpy(), pair_df["control_student"].to_numpy()])
        )
    return matched, pair_df

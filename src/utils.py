"""Shared utilities: config, seeds, IO, clustered OLS, significance stars."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Optional[Union[str, Path]] = None) -> dict:
    if path is None:
        path = ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: dict) -> None:
    for key in ("raw_dir", "processed_dir", "checkpoint_dir", "result_dir", "figure_dir", "table_dir"):
        p = ROOT / cfg["paths"][key]
        p.mkdir(parents=True, exist_ok=True)


def resolve_path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def get_device(cfg: dict) -> torch.device:
    name = str(cfg.get("device", "cpu")).lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def clustered_ols(
    y: np.ndarray,
    X: np.ndarray,
    cluster: np.ndarray,
    var_names: Optional[Sequence[str]] = None,
) -> dict:
    """OLS with cluster-robust sandwich standard errors (Liang-Zeger).

    Default clustering follows the paper: student-level SEs.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    sst = np.sum((y - y.mean()) ** 2)
    sse = np.sum(resid ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0

    meat = np.zeros((k, k), dtype=float)
    clusters = pd.Series(cluster)
    for _, idx in clusters.groupby(clusters).groups.items():
        Xi = X[idx]
        ei = resid[idx]
        score = Xi.T @ ei
        meat += np.outer(score, score)
    vcov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(vcov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    from scipy.stats import t as student_t

    df = max(int(clusters.nunique() - 1), 1)
    pvalues = 2.0 * student_t.sf(np.abs(tstat), df=df)
    names = list(var_names) if var_names is not None else [f"x{i}" for i in range(k)]
    return {
        "beta": beta,
        "se": se,
        "tstat": tstat,
        "pvalue": pvalues,
        "r2": float(r2),
        "n": int(n),
        "n_clusters": int(clusters.nunique()),
        "names": names,
        "residuals": resid,
    }


def format_coef(beta: float, tstat: float, pvalue: float) -> Tuple[str, str]:
    return f"{beta:.2f}{stars(pvalue)}", f"({tstat:.2f})"


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_conv)


def within_transform(df: pd.DataFrame, cols: Iterable[str], group: str) -> pd.DataFrame:
    out = df.copy()
    means = out.groupby(group)[list(cols)].transform("mean")
    out[list(cols)] = out[list(cols)] - means
    return out

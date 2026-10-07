"""Construct analysis matrices from the questionnaire panel (Section 3.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.schema import CONTROLS
from src.utils import ROOT, load_config


class PEDiDDataset(Dataset):
    """Observation-level dataset for PyTorch DID / PSM / mediation modules."""

    def __init__(self, df: pd.DataFrame, outcome: str = "CES-D"):
        self.df = df.reset_index(drop=True)
        self.outcome = outcome
        self.X = self.df[CONTROLS].to_numpy(dtype=np.float32)
        self.did = self.df["did"].to_numpy(dtype=np.float32)
        self.treat = self.df["treat"].to_numpy(dtype=np.float32)
        self.time = self.df["time"].to_numpy(dtype=np.float32)
        self.y = self.df[outcome].to_numpy(dtype=np.float32)
        self.hi = self.df["HI"].to_numpy(dtype=np.float32)
        self.ef = self.df["EF"].to_numpy(dtype=np.float32)
        self.college = self.df["college_id"].to_numpy(dtype=np.int64)
        self.wave = self.df["wave"].to_numpy(dtype=np.int64)
        self.student = self.df["student_idx"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": torch.from_numpy(self.X[idx]),
            "did": torch.tensor(self.did[idx]),
            "treat": torch.tensor(self.treat[idx]),
            "time": torch.tensor(self.time[idx]),
            "y": torch.tensor(self.y[idx]),
            "hi": torch.tensor(self.hi[idx]),
            "ef": torch.tensor(self.ef[idx]),
            "college_id": torch.tensor(self.college[idx]),
            "wave": torch.tensor(self.wave[idx]),
            "student_idx": torch.tensor(self.student[idx]),
        }


def load_panel(cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    path = ROOT / cfg["paths"]["processed_dir"] / "analysis_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python main.py generate")
    df = pd.read_csv(path)
    df["student_idx"] = df["student_idx"].astype(int)
    df["college_id"] = df["college_id"].astype(int)
    df["wave"] = df["wave"].astype(int)
    return df


def add_event_dummies(df: pd.DataFrame, window: Tuple[int, int] = (-2, 2)) -> pd.DataFrame:
    """Event-study dummies for the parallel-trend test (Figure 2). k = -1 is omitted."""
    out = df.copy()
    lo, hi = window
    ev = out["event_time"]
    for k in range(lo, hi + 1):
        if k == -1:
            continue
        out[f"e{k:+d}".replace("+", "p").replace("-", "m")] = ((out["treat"] == 1) & (ev == k)).astype(int)
    return out


def design_matrix(
    df: pd.DataFrame,
    controls: bool = True,
    time_fe: bool = False,
    drop_first_time: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    cols = [df["did"].to_numpy(dtype=float).reshape(-1, 1)]
    names = ["did"]
    if controls:
        cols.append(df[CONTROLS].to_numpy(dtype=float))
        names.extend(CONTROLS)
    if time_fe:
        waves = sorted(df["wave"].unique())
        use = waves[1:] if drop_first_time else waves
        for w in use:
            cols.append((df["wave"] == w).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"wave_{w}")
    intercept = np.ones((len(df), 1), dtype=float)
    X = np.hstack([intercept] + cols)
    return X, ["intercept"] + names


def split_student_ids(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = df["student_idx"].unique()
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return np.array(sorted(train_ids)), np.array(sorted(val_ids))


def make_loaders(df: pd.DataFrame, cfg: dict, outcome: str = "CES-D"):
    train_ids, val_ids = split_student_ids(df, cfg["train"]["val_ratio"], cfg["seed"])
    train_df = df[df["student_idx"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["student_idx"].isin(val_ids)].reset_index(drop=True)
    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(PEDiDDataset(train_df, outcome), batch_size=bs, shuffle=True)
    val_loader = DataLoader(PEDiDDataset(val_df, outcome), batch_size=bs, shuffle=False)
    full_loader = DataLoader(PEDiDDataset(df, outcome), batch_size=bs, shuffle=False)
    return train_loader, val_loader, full_loader, train_df, val_df

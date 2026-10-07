"""Score raw questionnaire items and build the analysis panel (Section 3.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import CESD_ITEM_COLS, CESD_ITEMS, CONTROLS, HI_ITEM_COLS
from src.utils import ROOT, load_config


def score_cesd(df: pd.DataFrame) -> pd.Series:
    """CES-D-10 total: items 5 and 8 reverse-coded, each 0-3, total 0-30."""
    scored = df[CESD_ITEM_COLS].to_numpy(dtype=float).copy()
    reverse_idx = [i for i, it in enumerate(CESD_ITEMS) if it["reverse"]]
    scored[:, reverse_idx] = 3.0 - scored[:, reverse_idx]
    return pd.Series(scored.sum(axis=1), index=df.index, name="CES-D")


def score_hi(df: pd.DataFrame) -> pd.DataFrame:
    total = df[HI_ITEM_COLS].sum(axis=1)
    mean = total / float(len(HI_ITEM_COLS))
    level = pd.cut(total, bins=[-np.inf, 23, 33, 46, 56, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    return pd.DataFrame({"HI_total": total, "HI": mean, "HI_level": level})


def quality_report(raw: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("n_observations", len(panel), 3066),
        ("n_students", int(panel["student_idx"].nunique()), 1533),
        ("n_cities", int(panel["college_id"].nunique()), 26),
        ("n_treated_cities", int(panel.loc[panel["treat"] == 1, "college_id"].nunique()), 15),
        ("cesd_min", int(panel["CES-D"].min()), 0),
        ("cesd_max", int(panel["CES-D"].max()), 30),
        ("hi_min", float(panel["HI"].min()), 1.0),
        ("hi_max", float(panel["HI"].max()), 5.0),
        ("balanced_two_waves", int((panel.groupby("student_idx").size() == 2).all()), 1),
        ("consent_all", int((raw["consent"] == 1).all()), 1),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "target"])


def build_analysis_panel(raw: pd.DataFrame) -> pd.DataFrame:
    panel = raw.copy()
    panel["CES-D"] = score_cesd(panel)
    hi = score_hi(panel)
    for c in hi.columns:
        panel[c] = hi[c]
    keep = [
        "student_id", "student_idx", "college_id", "college_name", "city", "wave", "survey_year",
        "treat", "time", "did", "reform_start_wave", "event_time", "consent", "irb_id",
        "CES-D", "HI", "HI_total", "HI_level", "EF",
    ] + CONTROLS
    return panel[keep]


def preprocess(cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    raw_path = ROOT / cfg["paths"]["raw_dir"] / "questionnaire_items.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run: python main.py generate")
    raw = pd.read_csv(raw_path)
    panel = build_analysis_panel(raw)
    out = ROOT / cfg["paths"]["processed_dir"] / "analysis_panel.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False, encoding="utf-8-sig")
    report = quality_report(raw, panel)
    report.to_csv(ROOT / cfg["paths"]["processed_dir"] / "data_quality.csv", index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    preprocess()

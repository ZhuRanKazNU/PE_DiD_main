"""Chapter 4 evaluation: tables, parallel trends, placebo, figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PEDiDDataset, load_panel
from src.data.schema import CONTROLS
from src.estimate import (
    baseline_table,
    descriptive_table,
    event_study,
    heterogeneity_city,
    heterogeneity_year,
    mechanism_table,
    placebo_test,
    psm_balance_table,
)
from src.models.pe_did_network import NetworkSpec, PEDiDNetwork
from src.models.psm_logit import match_students, student_baseline
from src.utils import ROOT, ensure_dirs, get_device, load_config, save_json, set_seed, stars


sns.set_theme(style="whitegrid", font_scale=1.05)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  wrote {path}")


def plot_event_study(ev: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    k = ev["event_time"].to_numpy()
    b = ev["beta"].to_numpy()
    se = ev["se"].to_numpy()
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    ax.errorbar(k, b, yerr=1.96 * np.nan_to_num(se), fmt="o-", color="#1f4e79", capsize=4)
    ax.set_xlabel("Event time (waves relative to PE reform)")
    ax.set_ylabel("Effect on CES-D")
    ax.set_title("Figure 2. Parallel-trend / event-study estimates")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_balance(balance: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    u = balance[balance["Sample"] == "U"]
    m = balance[balance["Sample"] == "M"]
    y = np.arange(len(u))
    ax.scatter(u["Std. Bias (%)"].abs(), y, label="Unmatched", color="#c0392b", s=40)
    ax.scatter(m["Std. Bias (%)"].abs(), y, label="Matched", color="#1f4e79", s=40)
    ax.axvline(10, color="gray", ls="--", lw=0.9, label="10% threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(u["Variable"])
    ax.set_xlabel("Absolute standardized bias (%)")
    ax.set_title("Figure 3. Covariate balance before and after PSM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_placebo(placebo: pd.DataFrame, actual: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.kdeplot(placebo["beta"], ax=ax, fill=True, color="#5b9bd5", label="Placebo coefficients")
    ax.axvline(actual, color="#c0392b", ls="--", lw=1.6, label=f"Actual DID ({actual:.2f})")
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Estimated DID coefficient")
    ax.set_title("Figure 4. Placebo test (500 random city assignments)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bars(df: pd.DataFrame, path: Path, title: str, ylabel="DID effect on CES-D") -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colors = ["#1f4e79" if p < 0.05 else "#7f8c8d" for p in df["pvalue"]]
    ax.bar(df["label"], df["beta"], color=colors, edgecolor="black", width=0.65)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mechanism(mech: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    labels = ["CES-D", "HI", "EF"]
    betas = mech["beta"].to_numpy()
    colors = ["#c0392b", "#1f4e79", "#2e86ab"]
    ax.bar(labels, betas, color=colors, edgecolor="black", width=0.6)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("DID coefficient")
    ax.set_title("Figure 7. Mechanism analysis: PE reform channels")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_network(cfg) -> tuple[PEDiDNetwork, dict]:
    ckpt_path = ROOT / cfg["paths"]["checkpoint_dir"] / "pe_did_network.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}. Run: python main.py train")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    net = PEDiDNetwork(NetworkSpec(**payload["spec"]))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


@torch.no_grad()
def pytorch_test_metrics(net: PEDiDNetwork, df: pd.DataFrame, device) -> dict:
    loader = DataLoader(PEDiDDataset(df), batch_size=512, shuffle=False)
    net.to(device)
    ys, preds = [], []
    for batch in loader:
        pred = net.predict_cesd(
            did=batch["did"].to(device),
            X=batch["X"].to(device),
            wave=batch["wave"].to(device),
            student_idx=batch["student_idx"].to(device),
            college_id=batch["college_id"].to(device),
            matched=False,
        )
        ys.append(batch["y"].numpy())
        preds.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    mse = float(np.mean((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((y - p) ** 2) / sst) if sst > 0 else 0.0
    return {
        "pytorch_did_beta": net.did.did_coefficient(),
        "pytorch_psm_did_beta": net.psm_did.did_coefficient(),
        "pytorch_mse": mse,
        "pytorch_r2": r2,
        "pytorch_mae": float(np.mean(np.abs(y - p))),
    }


def run_test(cfg=None, skip_placebo: bool = False) -> dict:
    if cfg is None:
        cfg = load_config()
    set_seed(int(cfg["seed"]))
    ensure_dirs(cfg)
    device = get_device(cfg)
    df = load_panel(cfg)
    fig_dir = ROOT / cfg["paths"]["figure_dir"]
    tab_dir = ROOT / cfg["paths"]["table_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("Table 2. Descriptive statistics")
    desc = descriptive_table(df)
    print(desc.to_string(index=False))
    _save_table(desc, tab_dir / "table2_descriptive.csv")

    print("\nTable 3. Baseline DID regression")
    t3 = baseline_table(df)
    print(t3.to_string(index=False))
    _save_table(t3, tab_dir / "table3_baseline_did.csv")
    actual_did = float(t3.attrs["did_beta"][3])

    print("\nFigure 2. Parallel-trend test")
    ev = event_study(df)
    print(ev.to_string(index=False))
    _save_table(ev, tab_dir / "figure2_event_study.csv")
    plot_event_study(ev, fig_dir / "figure2_parallel_trends.png")

    print("\nPSM matching and Table 4 balance")
    net, payload = load_network(cfg)
    baseline = student_baseline(df)
    X = torch.tensor(baseline[CONTROLS].to_numpy(dtype=np.float32))
    mu = torch.tensor(payload["psm_mu"])
    sd = torch.tensor(payload["psm_sd"])
    with torch.no_grad():
        scores = net.psm(((X - mu) / sd).clamp(-8, 8)).numpy()
    matched_ids, _pairs = match_students(
        baseline,
        scores,
        caliper=max(float(cfg["psm"]["caliper"]), 0.20),
        n_neighbors=1,
        replacement=False,
        target_matched_obs=int(cfg["psm"]["target_matched_obs"]),
        waves_per_student=2,
    )
    matched = df[df["student_idx"].isin(matched_ids)].copy()
    print(f"  matched students={len(matched_ids)}  matched obs={len(matched)}")
    # student-level balance table (as in Table 4)
    base_u = baseline
    base_m = baseline[baseline["student_idx"].isin(matched_ids)]
    bal = psm_balance_table(base_u, base_m)
    print(bal.to_string(index=False))
    _save_table(bal, tab_dir / "table4_psm_balance.csv")
    plot_balance(bal, fig_dir / "figure3_covariate_balance.png")

    print("\nTable 5. PSM-DID")
    t5 = baseline_table(matched)
    print(t5.to_string(index=False))
    _save_table(t5, tab_dir / "table5_psm_did.csv")

    if not skip_placebo:
        print(f"\nPlacebo test ({cfg['eval']['placebo_reps']} reps) ...")
        placebo = placebo_test(df, n_reps=int(cfg["eval"]["placebo_reps"]), seed=int(cfg["seed"]))
        _save_table(placebo, tab_dir / "figure4_placebo_draws.csv")
        plot_placebo(placebo, actual_did, fig_dir / "figure4_placebo.png")
        share = float((placebo["pvalue"] > 0.1).mean())
        print(f"  placebo mean={placebo['beta'].mean():.3f}  share p>0.1={share:.2%}")
    else:
        placebo = None
        share = None

    print("\nTable 6. Heterogeneity by academic year")
    hy = heterogeneity_year(df)
    print(hy.to_string(index=False))
    _save_table(hy, tab_dir / "table6_heterogeneity_year.csv")
    plot_bars(hy, fig_dir / "figure5_year_heterogeneity.png", "Figure 5. PE reform effect by academic year")

    print("\nTable 7. Heterogeneity by city level")
    hc = heterogeneity_city(df)
    print(hc.to_string(index=False))
    _save_table(hc, tab_dir / "table7_heterogeneity_city.csv")
    plot_bars(hc, fig_dir / "figure6_city_heterogeneity.png", "Figure 6. PE reform effect by city level")

    print("\nTable 8. Mechanism analysis")
    mech = mechanism_table(df)
    print(mech.to_string(index=False))
    _save_table(mech, tab_dir / "table8_mechanism.csv")
    plot_mechanism(mech, fig_dir / "figure7_mechanism.png")

    pt = pytorch_test_metrics(net, df, device)
    print("\nPyTorch network test metrics")
    print(pt)

    summary = {
        "n_obs": int(len(df)),
        "n_students": int(df["student_idx"].nunique()),
        "n_matched_obs": int(len(matched)),
        "table3_did_model4": actual_did,
        "table3_did_model4_stars": stars(float(t3.attrs["did_p"][3])),
        "pytorch": pt,
        "placebo_share_p_gt_0.1": share,
        "year_effects": hy.to_dict(orient="records"),
        "city_effects": hc.to_dict(orient="records"),
        "mechanism": mech[["outcome", "beta", "tstat", "pvalue", "r2", "n"]].to_dict(orient="records"),
    }
    save_json(summary, ROOT / cfg["paths"]["result_dir"] / "test_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-placebo", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_test(cfg, skip_placebo=args.skip_placebo)


if __name__ == "__main__":
    main()

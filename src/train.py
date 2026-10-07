"""Train Chapter 3 modules (PSM logit, time-varying DID, PSM-DID, mediation)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PEDiDDataset, load_panel, make_loaders
from src.data.schema import CONTROLS
from src.models.pe_did_network import NetworkSpec, PEDiDNetwork
from src.models.psm_logit import match_students, student_baseline
from src.utils import ROOT, ensure_dirs, get_device, load_config, set_seed


def _forward_outcome(model, batch, device):
    return model(
        did=batch["did"].to(device),
        X=batch["X"].to(device),
        college_id=batch["college_id"].to(device),
        wave=batch["wave"].to(device),
        student_idx=batch["student_idx"].to(device),
    )


def _run_epochs(model, loader, device, epochs, lr, weight_decay, y_key="y", patience=50, log_every=50, desc="train"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best = float("inf")
    bad = 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            y = batch[y_key].to(device)
            pred = _forward_outcome(model, batch, device)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * y.size(0)
            n += y.size(0)
        mse = total / max(n, 1)
        history.append(mse)
        if mse + 1e-6 < best:
            best = mse
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            print(f"  [{desc}] epoch {ep:04d}  mse={mse:.4f}")
        if bad >= patience:
            print(f"  [{desc}] early stop at epoch {ep}, best mse={best:.4f}")
            break
    return {"best_mse": best, "history": history}


def train_psm(model: nn.Module, baseline: pd.DataFrame, device, epochs, lr) -> Dict:
    X = torch.tensor(baseline[CONTROLS].to_numpy(dtype=np.float32), device=device)
    y = torch.tensor(baseline["treat"].to_numpy(dtype=np.float32), device=device)
    # standardize covariates for stable logit training
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    Xz = (X - mu) / sd
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        pred = model(Xz)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            acc = float(((pred > 0.5).float() == y).float().mean().item())
            print(f"  [psm] epoch {ep:04d}  bce={loss.item():.4f}  acc={acc:.3f}")
    with torch.no_grad():
        scores = model(Xz).detach().cpu().numpy()
    return {"history": history, "scores": scores, "mu": mu.detach().cpu().numpy(), "sd": sd.detach().cpu().numpy()}


@torch.no_grad()
def eval_mse(model, loader, device, y_key="y") -> float:
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    for batch in loader:
        y = batch[y_key].to(device)
        pred = _forward_outcome(model, batch, device)
        total += float(loss_fn(pred, y).item())
        n += y.size(0)
    return total / max(n, 1)


def build_network(df: pd.DataFrame) -> PEDiDNetwork:
    spec = NetworkSpec(
        n_controls=len(CONTROLS),
        n_colleges=int(df["college_id"].max()) + 1,
        n_times=int(df["wave"].max()) + 1,
        n_students=int(df["student_idx"].max()) + 1,
    )
    return PEDiDNetwork(spec)


def train_all(cfg: Optional[dict] = None) -> Path:
    if cfg is None:
        cfg = load_config()
    set_seed(int(cfg["seed"]))
    ensure_dirs(cfg)
    device = get_device(cfg)
    df = load_panel(cfg)
    print(f"Loaded panel: {len(df)} obs, {df['student_idx'].nunique()} students, {df['college_id'].nunique()} colleges")

    net = build_network(df)
    tr = cfg["train"]
    full_loader = DataLoader(PEDiDDataset(df), batch_size=int(tr["batch_size"]), shuffle=True)
    _, val_loader, _, train_df, val_df = make_loaders(df, cfg)

    print("\n==> Training PSM logit (Eq. 2)")
    baseline = student_baseline(df)
    psm_out = train_psm(net.psm, baseline, device, epochs=max(200, tr["epochs"] // 2), lr=tr["lr"])
    matched_ids, pairs = match_students(
        baseline,
        psm_out["scores"],
        caliper=cfg["psm"]["caliper"],
        n_neighbors=cfg["psm"]["n_neighbors"],
        replacement=cfg["psm"]["replacement"],
        target_matched_obs=cfg["psm"]["target_matched_obs"],
        waves_per_student=2,
    )
    matched_df = df[df["student_idx"].isin(matched_ids)].reset_index(drop=True)
    print(f"  matched students={len(matched_ids)}  matched obs={len(matched_df)}  pairs={len(pairs)}")

    print("\n==> Training time-varying DID Model (4) (Eq. 1)")
    did_hist = _run_epochs(
        net.did, full_loader, device, tr["epochs"], tr["lr"], tr["weight_decay"],
        patience=tr["patience"], log_every=tr["log_every"], desc="did"
    )
    print(f"  learned DID coefficient β1 = {net.did.did_coefficient():.4f}")

    print("\n==> Training PSM-DID on matched sample (Eq. 3)")
    matched_loader = DataLoader(PEDiDDataset(matched_df), batch_size=int(tr["batch_size"]), shuffle=True)
    psm_did_hist = _run_epochs(
        net.psm_did, matched_loader, device, tr["epochs"], tr["lr"], tr["weight_decay"],
        patience=tr["patience"], log_every=tr["log_every"], desc="psm-did"
    )
    print(f"  learned PSM-DID coefficient β1 = {net.psm_did.did_coefficient():.4f}")

    print("\n==> Training mediation first stage: HI (Eq. 4)")
    _run_epochs(
        net.med_hi, full_loader, device, tr["epochs"], tr["lr"], tr["weight_decay"],
        y_key="hi", patience=tr["patience"], log_every=tr["log_every"], desc="med-HI"
    )
    print("\n==> Training mediation first stage: EF (Eq. 4)")
    _run_epochs(
        net.med_ef, full_loader, device, tr["epochs"], tr["lr"], tr["weight_decay"],
        y_key="ef", patience=tr["patience"], log_every=tr["log_every"], desc="med-EF"
    )

    print("\n==> Training mediation second stage (Eq. 5)")
    net.med_y.to(device)
    opt = torch.optim.Adam(net.med_y.parameters(), lr=tr["lr"])
    loss_fn = nn.MSELoss()
    for ep in range(1, tr["epochs"] + 1):
        net.med_y.train()
        total, n = 0.0, 0
        for batch in full_loader:
            M = torch.stack([batch["hi"], batch["ef"]], dim=1).to(device)
            y = batch["y"].to(device)
            pred = net.med_y(
                did=batch["did"].to(device),
                M=M,
                X=batch["X"].to(device),
                wave=batch["wave"].to(device),
                student_idx=batch["student_idx"].to(device),
            )
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * y.size(0)
            n += y.size(0)
        if ep == 1 or ep % tr["log_every"] == 0 or ep == tr["epochs"]:
            print(f"  [med-Y] epoch {ep:04d}  mse={total / max(n, 1):.4f}")

    ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "pe_did_network.pt"
    payload = {
        "state_dict": net.state_dict(),
        "spec": net.spec.__dict__,
        "psm_mu": psm_out["mu"],
        "psm_sd": psm_out["sd"],
        "matched_student_idx": matched_ids,
        "did_beta": net.did.did_coefficient(),
        "psm_did_beta": net.psm_did.did_coefficient(),
        "did_mse": did_hist["best_mse"],
        "psm_did_mse": psm_did_hist["best_mse"],
    }
    torch.save(payload, ckpt_path)

    pair_path = ROOT / cfg["paths"]["processed_dir"] / "psm_pairs.csv"
    pairs.to_csv(pair_path, index=False, encoding="utf-8-sig")
    matched_df.to_csv(ROOT / cfg["paths"]["processed_dir"] / "matched_panel.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved checkpoint: {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_all(cfg)


if __name__ == "__main__":
    main()

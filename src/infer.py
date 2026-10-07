"""Inference: propensity scores, CES-D predictions, and reform counterfactuals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_panel
from src.data.schema import CONTROLS
from src.models.pe_did_network import NetworkSpec, PEDiDNetwork
from src.utils import ROOT, get_device, load_config, set_seed


def load_network(cfg, device):
    ckpt = ROOT / cfg["paths"]["checkpoint_dir"] / "pe_did_network.pt"
    try:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location=device)
    net = PEDiDNetwork(NetworkSpec(**payload["spec"]))
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    net.eval()
    return net, payload


@torch.no_grad()
def infer_frame(df: pd.DataFrame, net: PEDiDNetwork, payload: dict, device) -> pd.DataFrame:
    X = torch.tensor(df[CONTROLS].to_numpy(dtype=np.float32), device=device)
    did = torch.tensor(df["did"].to_numpy(dtype=np.float32), device=device)
    wave = torch.tensor(df["wave"].to_numpy(dtype=np.int64), device=device)
    sid = torch.tensor(df["student_idx"].to_numpy(dtype=np.int64), device=device)
    cid = torch.tensor(df["college_id"].to_numpy(dtype=np.int64), device=device)
    mu = torch.tensor(payload["psm_mu"], device=device)
    sd = torch.tensor(payload["psm_sd"], device=device)
    ps = net.propensity((X - mu) / sd.clamp_min(1e-6))
    yhat = net.predict_cesd(did, X, wave, sid, cid, matched=False)
    cf = net.counterfactual_cesd(X, wave, sid, cid)
    med = net.predict_mediators(did, X, wave, sid, cid)
    out = df.copy()
    out["propensity_score"] = ps.cpu().numpy()
    out["cesd_hat"] = yhat.cpu().numpy()
    out["cesd_hat_no_reform"] = cf["y0"].cpu().numpy()
    out["cesd_hat_with_reform"] = cf["y1"].cpu().numpy()
    out["predicted_att"] = cf["att"].cpu().numpy()
    out["hi_hat"] = med["HI"].cpu().numpy()
    out["ef_hat"] = med["EF"].cpu().numpy()
    return out


def run_infer(cfg=None, student_id: str | None = None, n_examples: int | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    set_seed(int(cfg["seed"]))
    device = get_device(cfg)
    df = load_panel(cfg)
    net, payload = load_network(cfg, device)
    pred = infer_frame(df, net, payload, device)

    out_path = ROOT / cfg["paths"]["result_dir"] / "inference_panel.csv"
    pred.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote full inference panel: {out_path}")

    if student_id:
        sub = pred[pred["student_id"] == student_id]
        if sub.empty:
            print(f"student_id {student_id} not found.")
        else:
            cols = ["student_id", "city", "survey_year", "treat", "time", "did", "CES-D",
                    "cesd_hat", "cesd_hat_no_reform", "cesd_hat_with_reform", "predicted_att",
                    "propensity_score", "HI", "hi_hat", "EF", "ef_hat"]
            print(sub[cols].to_string(index=False))
        return sub

    n = int(n_examples or cfg.get("infer", {}).get("n_examples", 8))
    sample = pred.drop_duplicates("student_idx").head(n)
    cols = ["student_id", "city", "AY", "treat", "CES-D", "cesd_hat",
            "cesd_hat_no_reform", "cesd_hat_with_reform", "predicted_att", "propensity_score"]
    print("Example counterfactual predictions (first-wave rows):")
    print(sample[cols].to_string(index=False))
    print(f"\nMean predicted ATT (CES-D change if treated): {pred['predicted_att'].mean():.3f}")
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--student_id", type=str, default=None)
    parser.add_argument("--n_examples", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_infer(cfg, student_id=args.student_id, n_examples=args.n_examples)


if __name__ == "__main__":
    main()

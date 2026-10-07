"""Econometric estimation matching Chapter 4 tables.

Linear specifications are estimated by OLS with student-clustered SEs
(paper default). Individual FE models use within-student demeaning plus
time dummies; reported R² is the within R² for FE models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.dataset import design_matrix
from src.data.schema import CONTROLS
from src.utils import clustered_ols, format_coef, stars


def _within_group(df: pd.DataFrame, cols: Sequence[str], group: str) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(group)
    for c in cols:
        out[c] = out[c] - g[c].transform("mean")
    return out


def _within_student(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return _within_group(df, cols, "student_idx")


def estimate_spec(
    df: pd.DataFrame,
    y_col: str,
    controls: bool,
    two_way_fe: bool,
    cluster_col: str = "student_idx",
    fe_group: str = "college_id",
) -> dict:
    work = df.copy()
    y = work[y_col].to_numpy(dtype=float)
    if two_way_fe:
        # Eq.1 uses college FE μ_c plus time FE θ_t, which keeps student-level
        # controls identified (true student FE would absorb Gd, UC, AY, ...).
        cols = ["did"] + (CONTROLS if controls else [])
        work = _within_group(work, cols + [y_col], fe_group)
        y = work[y_col].to_numpy(dtype=float)
        X, names = design_matrix(work, controls=controls, time_fe=True, drop_first_time=True)
        # intercept is collinear after demeaning
        X = X[:, 1:]
        names = names[1:]
        r2_kind = "within"
    else:
        X, names = design_matrix(work, controls=controls, time_fe=False)
        r2_kind = "overall"
    # drop all-zero / absorbed columns, but never drop a column of ones (intercept)
    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.allclose(col, 0.0):
            keep.append(False)
        elif col.std() < 1e-10 and not np.allclose(col, 1.0):
            keep.append(False)
        else:
            keep.append(True)
    if not all(keep):
        X = X[:, np.array(keep)]
        names = [n for n, k in zip(names, keep) if k]
    cluster = work[cluster_col].to_numpy()
    res = clustered_ols(y, X, cluster, names)
    res["r2_kind"] = r2_kind
    res["y_col"] = y_col
    res["controls"] = controls
    res["two_way_fe"] = two_way_fe
    return res


def baseline_table(df: pd.DataFrame, y_col: str = "CES-D", cluster_col: str = "student_idx") -> pd.DataFrame:
    specs = [
        dict(controls=False, two_way_fe=False),
        dict(controls=True, two_way_fe=False),
        dict(controls=False, two_way_fe=True),
        dict(controls=True, two_way_fe=True),
    ]
    rows = []
    stored = []
    for spec in specs:
        stored.append(estimate_spec(df, y_col, cluster_col=cluster_col, **spec))

    coef_vars = ["did"] + CONTROLS
    for v in coef_vars:
        coef_row = {"Variables": v}
        t_row = {"Variables": v + "_t"}
        for i, res in enumerate(stored, start=1):
            if v in res["names"]:
                j = res["names"].index(v)
                c, t = format_coef(res["beta"][j], res["tstat"][j], res["pvalue"][j])
                coef_row[f"({i})"] = c
                t_row[f"({i})"] = t
            else:
                coef_row[f"({i})"] = ""
                t_row[f"({i})"] = ""
        rows.extend([coef_row, t_row])

    meta = [
        {"Variables": "Individual Fixed", **{f"({i})": ("Yes" if stored[i - 1]["two_way_fe"] else "No") for i in range(1, 5)}},
        {"Variables": "Time Fixed", **{f"({i})": ("Yes" if stored[i - 1]["two_way_fe"] else "No") for i in range(1, 5)}},
        {"Variables": "R2", **{f"({i})": f"{stored[i - 1]['r2']:.3f}" for i in range(1, 5)}},
        {"Variables": "Observations", **{f"({i})": str(stored[i - 1]["n"]) for i in range(1, 5)}},
    ]
    rows.extend(meta)
    table = pd.DataFrame(rows)
    extras = {
        "did_beta": [float(s["beta"][s["names"].index("did")]) for s in stored],
        "did_t": [float(s["tstat"][s["names"].index("did")]) for s in stored],
        "did_p": [float(s["pvalue"][s["names"].index("did")]) for s in stored],
        "r2": [float(s["r2"]) for s in stored],
        "n": [int(s["n"]) for s in stored],
        "results": stored,
    }
    table.attrs.update(extras)
    return table


def standardized_bias(t: np.ndarray, c: np.ndarray) -> float:
    mt, mc = t.mean(), c.mean()
    vt, vc = t.var(ddof=1), c.var(ddof=1)
    denom = np.sqrt((vt + vc) / 2.0)
    if denom < 1e-12:
        return 0.0
    return 100.0 * (mt - mc) / denom


def psm_balance_table(df_u: pd.DataFrame, df_m: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import ttest_ind

    rows = []
    for var in CONTROLS:
        for label, d in [("U", df_u), ("M", df_m)]:
            t = d.loc[d["treat"] == 1, var].to_numpy(dtype=float)
            c = d.loc[d["treat"] == 0, var].to_numpy(dtype=float)
            sb = standardized_bias(t, c)
            tt = ttest_ind(t, c, equal_var=False, nan_policy="omit")
            rows.append(
                {
                    "Variable": var,
                    "Sample": label,
                    "Treatment": round(float(np.mean(t)), 2),
                    "Control": round(float(np.mean(c)), 2),
                    "Std. Bias (%)": round(float(sb), 1),
                    "t-stat": round(float(tt.statistic), 2),
                    "p-value": round(float(tt.pvalue), 3),
                }
            )
    tab = pd.DataFrame(rows)
    red = []
    for var in CONTROLS:
        u = tab[(tab.Variable == var) & (tab.Sample == "U")]["Std. Bias (%)"].iloc[0]
        m = tab[(tab.Variable == var) & (tab.Sample == "M")]["Std. Bias (%)"].iloc[0]
        if abs(u) < 1e-8:
            r = np.nan
        else:
            r = 100.0 * (1.0 - abs(m) / abs(u))
        red.extend([np.nan, round(float(r), 2) if pd.notna(r) else np.nan])
    tab["Bias Reduction (%)"] = red
    return tab


def event_study(df: pd.DataFrame, window=(-2, 2), cluster_col: str = "student_idx") -> pd.DataFrame:
    """Parallel-trend / event-study coefficients (Figure 2). k=-1 omitted."""
    work = df.copy()
    lo, hi = window
    dummy_names = []
    for k in range(lo, hi + 1):
        if k == -1:
            continue
        name = f"leadlag_{k}"
        work[name] = ((work["treat"] == 1) & (work["event_time"] == k)).astype(float)
        dummy_names.append(name)
    cols = dummy_names + CONTROLS + ["CES-D"]
    work = _within_student(work, cols)
    y = work["CES-D"].to_numpy(dtype=float)
    X_parts = [work[n].to_numpy(dtype=float).reshape(-1, 1) for n in dummy_names]
    X_parts.append(work[CONTROLS].to_numpy(dtype=float))
    waves = sorted(work["wave"].unique())
    for w in waves[1:]:
        X_parts.append((work["wave"] == w).astype(float).to_numpy().reshape(-1, 1))
    X = np.hstack(X_parts)
    names = dummy_names + CONTROLS + [f"wave_{w}" for w in waves[1:]]
    res = clustered_ols(y, X, work[cluster_col].to_numpy(), names)
    rows = []
    for k in range(lo, hi + 1):
        if k == -1:
            rows.append({"event_time": k, "beta": 0.0, "se": 0.0, "tstat": np.nan, "pvalue": np.nan, "omitted": True})
            continue
        name = f"leadlag_{k}"
        j = res["names"].index(name)
        rows.append(
            {
                "event_time": k,
                "beta": float(res["beta"][j]),
                "se": float(res["se"][j]),
                "tstat": float(res["tstat"][j]),
                "pvalue": float(res["pvalue"][j]),
                "omitted": False,
            }
        )
    return pd.DataFrame(rows)


def heterogeneity_year(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in [1, 2, 3, 4]:
        sub = df[df["AY"] == year]
        res = estimate_spec(sub, "CES-D", controls=True, two_way_fe=True)
        j = res["names"].index("did")
        rows.append(
            {
                "AY": year,
                "label": f"Year {year}",
                "beta": float(res["beta"][j]),
                "tstat": float(res["tstat"][j]),
                "pvalue": float(res["pvalue"][j]),
                "stars": stars(res["pvalue"][j]),
                "r2": float(res["r2"]),
                "n": int(res["n"]),
            }
        )
    return pd.DataFrame(rows)


def heterogeneity_city(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uc, label in [(1, "First-tier Cities"), (0, "Non-first-tier Cities")]:
        sub = df[df["UC"] == uc]
        res = estimate_spec(sub, "CES-D", controls=True, two_way_fe=True)
        j = res["names"].index("did")
        rows.append(
            {
                "UC": uc,
                "label": label,
                "beta": float(res["beta"][j]),
                "tstat": float(res["tstat"][j]),
                "pvalue": float(res["pvalue"][j]),
                "stars": stars(res["pvalue"][j]),
                "r2": float(res["r2"]),
                "n": int(res["n"]),
            }
        )
    return pd.DataFrame(rows)


def mechanism_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, label in [("CES-D", "CESD"), ("HI", "HI"), ("EF", "EF")]:
        res = estimate_spec(df, y, controls=True, two_way_fe=True)
        j = res["names"].index("did")
        c, t = format_coef(res["beta"][j], res["tstat"][j], res["pvalue"][j])
        rows.append(
            {
                "outcome": label,
                "coef": c,
                "t": t,
                "beta": float(res["beta"][j]),
                "tstat": float(res["tstat"][j]),
                "pvalue": float(res["pvalue"][j]),
                "r2": float(res["r2"]),
                "n": int(res["n"]),
            }
        )
    return pd.DataFrame(rows)


def placebo_test(df: pd.DataFrame, n_reps: int = 500, seed: int = 42) -> pd.DataFrame:
    """Counterfactual placebo: randomly reassign the 15 treated cities  (Section 4.2)."""
    rng = np.random.default_rng(seed)
    cities = df[["college_id", "treat"]].drop_duplicates()
    n_treated = int((cities["treat"] == 1).sum())
    college_ids = cities["college_id"].to_numpy()
    # Keep each college's original reform start distribution by recycling start waves
    start_map = (
        df.dropna(subset=["reform_start_wave"])
        .groupby("college_id")["reform_start_wave"]
        .first()
        .to_dict()
    )
    starts = [start_map[c] for c in cities.loc[cities["treat"] == 1, "college_id"]]
    records = []
    for r in range(n_reps):
        fake_treated = set(rng.choice(college_ids, size=n_treated, replace=False).tolist())
        fake_starts = {cid: starts[i] for i, cid in enumerate(fake_treated)}
        tmp = df.copy()
        tmp["treat_f"] = tmp["college_id"].isin(fake_treated).astype(int)
        tmp["did_f"] = 0
        for cid, st in fake_starts.items():
            tmp.loc[(tmp["college_id"] == cid) & (tmp["wave"] >= st), "did_f"] = 1
        work = tmp.copy()
        work["did"] = work["did_f"]
        res = estimate_spec(work, "CES-D", controls=True, two_way_fe=True)
        j = res["names"].index("did")
        records.append({"rep": r, "beta": float(res["beta"][j]), "pvalue": float(res["pvalue"][j])})
    return pd.DataFrame(records)


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["CES-D", "HI", "EF"] + CONTROLS:
        t = df.loc[df["treat"] == 1, var]
        c = df.loc[df["treat"] == 0, var]
        rows.append(
            {
                "Variable": var,
                "Treatment Mean": round(float(t.mean()), 2),
                "Treatment Std. Dev.": round(float(t.std(ddof=1)), 2),
                "Control Mean": round(float(c.mean()), 2),
                "Control Std. Dev.": round(float(c.std(ddof=1)), 2),
            }
        )
    return pd.DataFrame(rows)

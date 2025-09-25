"""
Barrier signature analysis using only existing results.

Idea: If barriers induce structured, real-world-aligned outcome shifts, then
the vector of metric changes (vs baseline) should reliably identify the
barrier type. We quantify this via a multinomial logistic regression on
delta-metrics and visualize per-class coefficients as a single, compact
heatmap suitable for the paper.

Outputs:
- analysis/derived/barrier_signature_data.csv (deltas dataset)
- analysis/figs/barrier_signature_logit.png (coefficient heatmap with CV score)

Run:
  python analysis/barrier_signature_analysis.py \
    --results_glob "results/exp_*_episode_all_neutralized" \
    --fig_dir analysis/figs --out_csv analysis/derived/barrier_signature_data.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


METRICS = ["BEL", "REL", "KNO", "GOAL", "CONFUSION", "MUTUAL"]


def load_table(root_glob: str) -> pd.DataFrame:
    rows: List[Dict] = []
    exp_roots = glob.glob(root_glob)
    modes = ["mode_baseline", "mode_semantic", "mode_cultural", "mode_emotional"]
    for root in exp_roots:
        model = os.path.basename(root).split("exp_")[-1].split("_episode")[0]
        for mode in modes:
            for scen in glob.glob(os.path.join(root, mode, "scenario_*")):
                eval_path = os.path.join(scen, "eval_result.json")
                if not os.path.exists(eval_path):
                    continue
                try:
                    with open(eval_path, "r", encoding="utf-8") as f:
                        ev = json.load(f)
                    epi = ev.get("aggregated_scores", {}).get("episode_level", {})
                    ag2 = ev.get("aggregated_scores", {}).get("agent_2", {})
                    row = {
                        "model": model,
                        "mode": mode.replace("mode_", ""),
                        "scenario": os.path.basename(scen),
                        "BEL": ag2.get("believability"),
                        "REL": ag2.get("relationship"),
                        "KNO": ag2.get("knowledge"),
                        "GOAL": ag2.get("goal_completion"),
                        "CONFUSION": epi.get("unresolved_confusion"),
                        "MUTUAL": epi.get("mutual_understanding"),
                    }
                    rows.append(row)
                except Exception:
                    continue
    return pd.DataFrame(rows)


def build_delta_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Keep rows that have baseline and barrier for same model+scenario
    base = df[df["mode"] == "baseline"][["model", "scenario"] + METRICS]
    base = base.rename(columns={m: f"base_{m}" for m in METRICS})
    merged = df[df["mode"] != "baseline"].merge(base, on=["model", "scenario"], how="inner")
    # Delta as (barrier - baseline) to capture direction
    for m in METRICS:
        merged[f"d_{m}"] = merged[m] - merged[f"base_{m}"]
        base = merged[f"base_{m}"]
        with np.errstate(divide='ignore', invalid='ignore'):
            merged[f"p_{m}"] = np.where(np.abs(base) > 1e-8, (merged[m] - base) / base * 100.0, np.nan)
    keep_cols = ["model", "scenario", "mode"] + [f"d_{m}" for m in METRICS] + [f"p_{m}" for m in METRICS]
    return merged[keep_cols].dropna()


def plot_coeff_heatmap(clf: LogisticRegression, scaler: StandardScaler, feature_names: List[str],
                       classes: List[str], title: str, out_path: str):
    # For multinomial, coef_.shape = (n_classes, n_features), aligned to classes_
    coefs = clf.coef_.copy()
    # Map indices to class names
    class_order = list(clf.classes_)
    # Build DataFrame with readable labels
    dfc = pd.DataFrame(coefs, index=class_order, columns=feature_names)
    # Order rows in ['semantic','cultural','emotional'] if possible
    order = [c for c in ["semantic", "cultural", "emotional"] if c in class_order]
    dfc = dfc.loc[order]

    plt.figure(figsize=(6.6, 2.8))
    sns.heatmap(dfc, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                cbar_kws={"label": "standardized coefficient"})
    plt.title(title)
    plt.xlabel("Metric deltas (barrier − baseline)")
    plt.ylabel("Barrier type")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_glob", type=str, default="results/exp_*_episode_all_neutralized")
    ap.add_argument("--out_csv", type=str, default="analysis/derived/barrier_signature_data.csv")
    ap.add_argument("--fig_dir", type=str, default="analysis/figs")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    df = load_table(args.results_glob)
    if df.empty:
        print("No data loaded.")
        return

    ds = build_delta_dataset(df)
    ds.to_csv(args.out_csv, index=False)

    X = ds[[f"d_{m}" for m in METRICS]].values
    y = ds["mode"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()

    # Fit on all data for coefficient visualization
    pipe.fit(X, y)
    clf: LogisticRegression = pipe.named_steps["clf"]
    scaler: StandardScaler = pipe.named_steps["scaler"]

    title = f"Barrier identifiability from outcome shifts (5-fold acc = {acc:.2f})"
    out_path = os.path.join(args.fig_dir, "barrier_signature_logit.png")
    plot_coeff_heatmap(clf, scaler, [f"d_{m}" for m in METRICS], list(np.unique(y)), title, out_path)
    print(f"Saved figure to {out_path}")

    # ----- Single line chart (one figure) -----
    # Each line: a barrier; X: metrics; Y: mean percent change vs baseline with bootstrap 95% CI
    def bootstrap_ci(a: np.ndarray, n_boot: int = 1000) -> (float, float, float):
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return np.nan, np.nan, np.nan
        means = []
        n = len(a)
        for _ in range(n_boot):
            idx = np.random.randint(0, n, n)
            means.append(np.nanmean(a[idx]))
        means = np.array(means)
        return float(np.nanmean(means)), float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))

    order = ["MUTUAL", "REL", "GOAL", "KNO", "CONFUSION", "BEL"]
    plot_df = []
    for mode in ["semantic", "cultural", "emotional"]:
        sub = ds[ds["mode"] == mode]
        for m in order:
            mean_, lo, hi = bootstrap_ci(sub[f"p_{m}"].values)
            plot_df.append(dict(mode=mode, metric=m, mean=mean_, lo=lo, hi=hi))
    plot_df = pd.DataFrame(plot_df)

    plt.figure(figsize=(6.8, 3.0))
    colors = {"semantic": "#1f77b4", "cultural": "#2ca02c", "emotional": "#d62728"}
    x = np.arange(len(order))
    for mode in ["semantic", "cultural", "emotional"]:
        s = plot_df[plot_df["mode"] == mode].set_index("metric").loc[order]
        plt.plot(x, s["mean"].values, marker="o", lw=2, label=mode, color=colors[mode])
        plt.fill_between(x, s["lo"].values, s["hi"].values, color=colors[mode], alpha=0.15, linewidth=0)
    plt.axhline(0, color="#888", lw=1, ls="--")
    plt.xticks(x, [m.title() if m not in ("BEL","KNO") else ("Bel" if m=="BEL" else "Kno") for m in order], rotation=0)
    plt.ylabel("% change vs baseline")
    plt.title("Barrier impact patterns across metrics (mean ± 95% CI)")
    plt.legend(frameon=False, ncol=3, loc="upper right")
    plt.tight_layout()
    line_path = os.path.join(args.fig_dir, "barrier_line_signature.png")
    plt.savefig(line_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved line figure to {line_path}")


if __name__ == "__main__":
    main()


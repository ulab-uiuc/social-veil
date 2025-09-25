"""
Improved percent-change line chart per barrier, with 95% CI and significance markers.
Enhanced with professional academic styling for ICLR submission.

Y-axis: Δ% vs baseline for each metric: (barrier - baseline) / baseline * 100.
Lines: semantic / cultural / emotional. Stars indicate CI not overlapping 0.

Run:
  python analysis/percent_line_effects.py \
    --results_glob "results/exp_*_episode_all_neutralized" \
    --fig_dir analysis/figs --out_csv analysis/derived/percent_line_table.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set professional styling for paper
plt.style.use('default')  # Use cleaner default style
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.grid': False,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'lightgray',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

METRICS = ["BEL", "REL", "KNO", "GOAL", "CONFUSION", "MUTUAL"]
ORDER = ["MUTUAL", "REL", "GOAL", "KNO", "CONFUSION", "BEL"]


def load_results(root_glob: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for exp_root in glob.glob(root_glob):
        model = os.path.basename(exp_root).split("exp_")[-1].split("_episode")[0]
        for mode in ["mode_baseline", "mode_semantic", "mode_cultural", "mode_emotional"]:
            short = mode.replace("mode_", "")
            for scen in glob.glob(os.path.join(exp_root, mode, "scenario_*")):
                eval_path = os.path.join(scen, "eval_result.json")
                if not os.path.exists(eval_path):
                    continue
                try:
                    with open(eval_path, "r", encoding="utf-8") as f:
                        ev = json.load(f)
                    epi = ev.get("aggregated_scores", {}).get("episode_level", {})
                    ag2 = ev.get("aggregated_scores", {}).get("agent_2", {})
                    rows.append(dict(
                        model=model,
                        mode=short,
                        scenario=os.path.basename(scen),
                        BEL=ag2.get("believability"),
                        REL=ag2.get("relationship"),
                        KNO=ag2.get("knowledge"),
                        GOAL=ag2.get("goal_completion"),
                        CONFUSION=epi.get("unresolved_confusion"),
                        MUTUAL=epi.get("mutual_understanding"),
                    ))
                except Exception:
                    continue
    return pd.DataFrame(rows)


def build_percent(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["mode"] == "baseline"][["model", "scenario"] + METRICS]
    base = base.rename(columns={m: f"base_{m}" for m in METRICS})
    merged = df[df["mode"] != "baseline"].merge(base, on=["model", "scenario"], how="inner")
    for m in METRICS:
        denom = merged[f"base_{m}"]
        with np.errstate(divide='ignore', invalid='ignore'):
            merged[f"p_{m}"] = np.where(np.abs(denom) > 1e-8, (merged[m] - denom) / denom * 100.0, np.nan)
    keep = ["model", "scenario", "mode"] + [f"p_{m}" for m in METRICS]
    return merged[keep].dropna()


def bootstrap_mean(a: np.ndarray, n_boot: int = 2000):
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, ""
    n = len(a)
    means = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        means.append(np.nanmean(a[idx]))
    means = np.array(means)
    mu = float(np.nanmean(means))
    lo = float(np.nanpercentile(means, 2.5))
    hi = float(np.nanpercentile(means, 97.5))
    sig = "*" if lo > 0 or hi < 0 else ""
    return mu, lo, hi, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_glob", type=str, default="results/exp_*_episode_all_neutralized")
    ap.add_argument("--fig_dir", type=str, default="analysis/figs")
    ap.add_argument("--out_csv", type=str, default="analysis/derived/percent_line_table.csv")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df = load_results(args.results_glob)
    if df.empty:
        print("No data loaded.")
        return

    ds = build_percent(df)

    # Build contrast signature C_b(m) = Δ%_b(m) − mean(Δ%_other(m)) at scenario level
    rows_c: List[Dict] = []
    for (model, scen, metric), grp in ds.melt(
        id_vars=["model", "scenario", "mode"],
        value_vars=[f"p_{m}" for m in METRICS],
        var_name="p_metric", value_name="p_val",
    ).assign(metric=lambda d: d["p_metric"].str.replace("p_", "", regex=False)).groupby(["model", "scenario", "metric"]):
        vals = {r["mode"]: r["p_val"] for _, r in grp.iterrows()}
        for b in ["semantic", "cultural", "emotional"]:
            if b in vals and len(vals) >= 3:
                others = [vals[o] for o in ["semantic", "cultural", "emotional"] if o != b and o in vals]
                if len(others) >= 2:
                    contrast = vals[b] - float(np.nanmean(others))
                    rows_c.append(dict(model=model, scenario=scen, barrier=b, metric=metric, contrast=contrast))
    contrast_df = pd.DataFrame(rows_c)

    # Bootstrap mean/CI per barrier×metric
    rows: List[Dict] = []
    for b in ["semantic", "cultural", "emotional"]:
        subb = contrast_df[contrast_df["barrier"] == b]
        for m in ORDER:
            mu, lo, hi, sig = bootstrap_mean(subb[subb["metric"] == m]["contrast"].values)
            rows.append(dict(mode=b, metric=m, mean=mu, lo=lo, hi=hi, sig=sig))
    tab = pd.DataFrame(rows)
    tab.to_csv(args.out_csv, index=False)

    # Enhanced professional visualization - compact size for paper layout
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Professional color scheme - more distinct and paper-friendly
    colors = {
        "semantic": "#E74C3C",    # Clear red
        "cultural": "#3498DB",    # Clear blue  
        "emotional": "#2ECC71"    # Clear green
    }
    
    # Line and marker styles
    styles = {
        "semantic": {"marker": "o", "markersize": 8, "linewidth": 2.5},
        "cultural": {"marker": "s", "markersize": 7, "linewidth": 2.5},
        "emotional": {"marker": "D", "markersize": 7, "linewidth": 2.5}
    }
    
    x = np.arange(len(ORDER))
    
    # Expected directions for significance testing
    expected = {
        ("semantic", "MUTUAL"): "neg", 
        ("emotional", "REL"): "neg", 
        ("cultural", "CONFUSION"): "pos"
    }

    # Plot each barrier type
    for mode in ["semantic", "cultural", "emotional"]:
        s = tab[tab["mode"] == mode].set_index("metric").loc[ORDER]
        
        # Main line with confidence intervals
        line = ax.plot(x, s["mean"].values, 
                      color=colors[mode], 
                      label=mode.capitalize(),
                      **styles[mode])
        
        # Confidence interval shading
        ax.fill_between(x, s["lo"].values, s["hi"].values, 
                       color=colors[mode], alpha=0.2, linewidth=0)
        
        # Add value labels with significance markers
        for xi, m in enumerate(ORDER):
            mu = s.loc[m, "mean"]
            sig = s.loc[m, "sig"]
            
            # Enhanced significance testing
            if (mode, m) in expected:
                lo, hi = s.loc[m, ["lo", "hi"]]
                if expected[(mode, m)] == "neg" and hi < 0:
                    sig = "*"
                elif expected[(mode, m)] == "pos" and lo > 0:
                    sig = "*"
            
            # Position labels smartly to avoid overlap
            offset = 1.2 if mu >= 0 else -1.8
            ax.text(xi, mu + offset, f"{mu:.1f}{sig}", 
                   ha="center", va="center" if mu >= 0 else "center",
                   fontsize=10, fontweight='bold', color=colors[mode],
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                           edgecolor=colors[mode], alpha=0.8))

    # Styling improvements
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    
    # Compact axis labels - single line where possible
    metric_labels = ["Mutual", "REL", "Goal", 
                    "KNO", "Confus", "BEL"]
    
    ax.set_xticks(x)
    ax.set_xticklabels(
                metric_labels, 
                fontsize=12, 
                fontfamily='DejaVu Sans', 
                fontweight='bold'
            )
    ax.set_ylabel("Contrast vs Other Barriers\n(Δ% − Mean of Others)", fontsize=12, fontfamily='DejaVu Sans', fontweight='bold')
    
    # Compact legend inside plot area
    legend = ax.legend(title="Barrier Type", loc='upper right', 
                      fontsize=12, title_fontsize=12,
                      fontfamily='DejaVu Sans', fontweight='bold',
                      bbox_to_anchor=(0.98, 0.98), frameon=True, framealpha=0.9)
    legend.get_title().set_fontweight('bold')
    
    # Grid and spine styling - cleaner for small size
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Tight layout for compact paper figure
    plt.tight_layout()
    
    # Save with high quality settings for publication
    output_path = os.path.join(args.fig_dir, "barrier_signatures_enhanced.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", 
               facecolor='white', edgecolor='none')
    
    # Also save as PDF for LaTeX inclusion
    pdf_path = os.path.join(args.fig_dir, "barrier_signatures_enhanced.pdf")
    plt.savefig(pdf_path, bbox_inches="tight", 
               facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"Enhanced visualization saved to:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {pdf_path}")


if __name__ == "__main__":
    main()
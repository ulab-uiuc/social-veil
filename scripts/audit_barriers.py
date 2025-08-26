#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # optional dependency

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    import pandas as pd
except Exception:
    pd = None


MODES = ["baseline", "semantic", "cultural", "emotional"]


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def iter_scenarios(base_dir: str) -> Dict[str, Dict[str, str]]:
    """Return mapping: scenario_id -> {mode: convo_path} for available modes."""
    out: Dict[str, Dict[str, str]] = {}
    for mode in MODES:
        mode_dir = os.path.join(base_dir, f"mode_{mode}")
        if not os.path.isdir(mode_dir):
            continue
        for name in os.listdir(mode_dir):
            if not name.startswith("scenario_"):
                continue
            scenario_id = name
            convo_path = os.path.join(mode_dir, name, "conversation_log.txt")
            if not os.path.isfile(convo_path):
                continue
            out.setdefault(scenario_id, {})[mode] = convo_path
    return out


def parse_utterances(text: str) -> List[str]:
    """Extract utterance contents from conversation_log.txt.

    Primary strategy: pull JSON "argument" fields from speak actions.
    Fallback: use text after the first colon on each line when reasonable.
    """
    utts: List[str] = []

    # 1) Extract values of "argument": "..." (robust to multiple JSON blobs per line)
    arg_re = re.compile(r'"argument"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
    for m in arg_re.finditer(text):
        val = m.group(1)
        # Unescape simple JSON escapes and collapse whitespace
        try:
            # Handle common escapes without full JSON decode
            val = val.replace('\\"', '"').replace('\\\\', '\\')
            val = re.sub(r"\\[nrt]", " ", val)
        except Exception:
            pass
        val = re.sub(r"\s+", " ", val).strip()
        if val:
            utts.append(val)

    if utts:
        return utts

    # 2) Fallback: take content after the first ':' if it seems like an utterance line
    for raw_ln in text.splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if ':' in ln:
            left, right = ln.split(':', 1)
            # Heuristic: short speaker label on the left
            if len(left.split()) <= 3:
                cand = right.strip()
                # Skip bare JSON-looking strings in fallback
                if not (cand.startswith('{') and cand.endswith('}')):
                    cand = re.sub(r"\s+", " ", cand).strip('" ').strip()
                    if cand:
                        utts.append(cand)

    return utts


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-mpnet-base-v2") -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Install to enable embedding audits.")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are 1D vectors L2-normalized
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


def mean_cosine_distance(utts_a: List[str], utts_b: List[str]) -> float:
    if not utts_a or not utts_b:
        return float("nan")
    try:
        # Doc-level embedding by averaging utterance embeddings (simple baseline)
        emb_a = embed_texts(utts_a)
        emb_b = embed_texts(utts_b)
        doc_a = np.mean(emb_a, axis=0)
        doc_b = np.mean(emb_b, axis=0)
        # Re-normalize
        doc_a /= max(1e-12, np.linalg.norm(doc_a))
        doc_b /= max(1e-12, np.linalg.norm(doc_b))
        return cosine_dist(doc_a, doc_b)
    except Exception:
        return float("nan")


def count_lexicon_hits(utts: List[str], lex: List[str]) -> int:
    if not lex:
        return 0
    total = 0
    for u in utts:
        low = u.lower()
        for w in lex:
            if w and w.lower() in low:
                total += 1
    return total


def audit_pair(baseline_path: str, barrier_path: str) -> Dict[str, float]:
    base_text = read_text(baseline_path)
    barr_text = read_text(barrier_path)
    base_utts = parse_utterances(base_text)
    barr_utts = parse_utterances(barr_text)

    result: Dict[str, float] = {}
    # Global similarity (cosine distance)
    result["cosine_doc"] = mean_cosine_distance(base_utts, barr_utts)

    # Targeted simple features (placeholders for fuller feature set)
    hedges = ["perhaps", "might", "it seems", "maybe", "sort of", "kind of"]
    result["hedges_baseline"] = count_lexicon_hits(base_utts, hedges)
    result["hedges_barrier"] = count_lexicon_hits(barr_utts, hedges)
    # Affect/exclamation
    affect = ["frustrated", "irritated", "impatient", "angry"]
    result["affect_baseline"] = count_lexicon_hits(base_utts, affect)
    result["affect_barrier"] = count_lexicon_hits(barr_utts, affect)
    result["exclaim_baseline"] = sum(u.count("!") for u in base_utts)
    result["exclaim_barrier"] = sum(u.count("!") for u in barr_utts)

    # Length proxies
    result["avg_len_baseline"] = float(np.mean([len(u.split()) for u in base_utts])) if base_utts else float("nan")
    result["avg_len_barrier"] = float(np.mean([len(u.split()) for u in barr_utts])) if barr_utts else float("nan")

    return result


def main():
    ap = argparse.ArgumentParser(description="Audit barrier effectiveness from existing conversation logs")
    ap.add_argument("--base_dir", type=str, required=True, help="Results root (contains mode_* subdirs)")
    ap.add_argument("--out_csv", type=str, default="", help="Optional CSV output path")
    ap.add_argument("--out_json", type=str, default="", help="Optional JSON output path")
    ap.add_argument("--out_dir", type=str, default="scripts/audit_reports", help="Directory to save figures")
    args = ap.parse_args()

    pairs = iter_scenarios(args.base_dir)

    rows: List[Dict[str, float]] = []
    for scenario_id, paths in sorted(pairs.items()):
        base = paths.get("baseline")
        for mode in ("semantic", "cultural", "emotional", "baseline"):
            barr = paths.get(mode)
            if not (base and barr):
                continue
            res = audit_pair(base, barr)
            res_row = {"scenario": scenario_id, "mode": mode}
            res_row.update(res)
            # Compute simple deltas
            for kbase, kbarr, kout in (
                ("hedges_baseline", "hedges_barrier", "delta_hedges"),
                ("affect_baseline", "affect_barrier", "delta_affect"),
                ("exclaim_baseline", "exclaim_barrier", "delta_exclaim"),
                ("avg_len_baseline", "avg_len_barrier", "delta_avg_len"),
            ):
                bv = res_row.get(kbase)
                rv = res_row.get(kbarr)
                try:
                    res_row[kout] = (float(rv) if rv is not None else np.nan) - (
                        float(bv) if bv is not None else np.nan
                    )
                except Exception:
                    res_row[kout] = np.nan
            rows.append(res_row)

    # Write JSON
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to {args.out_json}")

    # Write CSV
    if args.out_csv:
        import csv

        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        # header union
        header_keys = set()
        for r in rows:
            header_keys.update(r.keys())
        header = ["scenario", "mode"] + sorted([k for k in header_keys if k not in ("scenario", "mode")])
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([r.get(k, "") for k in header])
        print(f"Saved CSV to {args.out_csv}")

    # Print brief summary
    if rows:
        cos = [r.get("cosine_doc") for r in rows if isinstance(r.get("cosine_doc"), (int, float))]
        if cos:
            print(f"Mean cosine distance (doc-level): {np.nanmean(cos):.4f}")
        print(f"Audited pairs: {len(rows)}")
    else:
        print("No paired scenarios found. Ensure baseline and barrier modes exist under base_dir.")

    # Visualization
    if rows and plt is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        try:
            if pd is not None:
                df = pd.DataFrame(rows)
                # 1) Cosine distance by mode
                if "cosine_doc" in df.columns:
                    dfc = df.copy()
                    # coerce to numeric and drop NaN/inf
                    dfc["cosine_doc"] = pd.to_numeric(dfc["cosine_doc"], errors="coerce")
                    dfc = dfc[np.isfinite(dfc["cosine_doc"])].copy()
                    if not dfc.empty:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        if sns is not None:
                            sns.boxplot(data=dfc, x="mode", y="cosine_doc", ax=ax)
                            sns.stripplot(data=dfc, x="mode", y="cosine_doc", ax=ax, color="k", alpha=0.3, jitter=0.2)
                        else:
                            # simple grouped scatter
                            modes_sorted = sorted(dfc["mode"].unique())
                            for i, mode in enumerate(modes_sorted):
                                ys = dfc.loc[dfc["mode"] == mode, "cosine_doc"].values
                                xs = np.full_like(ys, i, dtype=float) + (np.random.rand(len(ys)) - 0.5) * 0.15
                                ax.plot(xs, ys, "o", alpha=0.6)
                            ax.set_xticks(range(len(modes_sorted)))
                            ax.set_xticklabels(modes_sorted)
                        ax.set_title("Cosine distance (baseline vs barrier)")
                        ax.set_ylabel("cosine distance")
                        fig.tight_layout()
                        fig.savefig(os.path.join(args.out_dir, "cosine_distance_by_mode.png"), dpi=150)
                        plt.close(fig)

                # 2) Feature deltas by mode (boxplots)
                long_feats = []
                delta_cols = ("delta_hedges", "delta_affect", "delta_exclaim", "delta_avg_len")
                for feat in delta_cols:
                    if feat in df.columns:
                        tmp = df[["mode", feat]].copy()
                        tmp["delta"] = pd.to_numeric(tmp[feat], errors="coerce") if pd is not None else tmp[feat]
                        tmp.drop(columns=[feat], inplace=True)
                        tmp["feature"] = feat
                        long_feats.append(tmp)
                if long_feats:
                    dfl = pd.concat(long_feats, axis=0, ignore_index=True)
                    dfl = dfl[np.isfinite(dfl["delta"])].copy()
                    if not dfl.empty:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        if sns is not None:
                            sns.boxplot(data=dfl, x="feature", y="delta", hue="mode", ax=ax)
                        else:
                            # Fallback: per-feature scatter
                            feats = dfl["feature"].unique()
                            modes = sorted(dfl["mode"].unique())
                            for i, feat in enumerate(feats):
                                sub = dfl[dfl["feature"] == feat]
                                for j, mode in enumerate(modes):
                                    ys = sub.loc[sub["mode"] == mode, "delta"].values
                                    xs = np.full_like(ys, i + j * 0.2, dtype=float)
                                    ax.plot(xs, ys, "o", alpha=0.5, label=f"{feat}-{mode}" if i == 0 else "")
                            ax.set_xticks(range(len(feats)))
                            ax.set_xticklabels(feats)
                            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
                        ax.set_title("Feature deltas (barrier - baseline)")
                        ax.set_ylabel("delta")
                        fig.tight_layout()
                        fig.savefig(os.path.join(args.out_dir, "feature_deltas_by_mode.png"), dpi=150)
                        plt.close(fig)
            else:
                print("Visualization skipped: pandas not installed.")
        except Exception as viz_err:
            print(f"Visualization error: {viz_err}")


if __name__ == "__main__":
    main()


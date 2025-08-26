#!/usr/bin/env python3
import argparse
import os
import re
import json
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

try:
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score
except Exception:
    PCA = None
    StratifiedKFold = None
    SVC = None
    accuracy_score = None
    f1_score = None


MODES = ["baseline", "semantic", "cultural", "emotional"]


def iter_scenarios(base_dir: str) -> Dict[str, Dict[str, str]]:
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


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def parse_utterances(text: str) -> List[str]:
    """Extract utterance contents from conversation_log.txt by reading JSON 'argument' fields.
    Falls back to simple colon split when JSON not present.
    """
    utts: List[str] = []
    arg_re = re.compile(r'"argument"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
    for m in arg_re.finditer(text):
        val = m.group(1)
        try:
            val = val.replace('\\"', '"').replace('\\\\', '\\')
            val = re.sub(r"\\[nrt]", " ", val)
        except Exception:
            pass
        val = re.sub(r"\s+", " ", val).strip()
        if val:
            utts.append(val)
    if utts:
        return utts
    for raw_ln in text.splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if ':' in ln:
            left, right = ln.split(':', 1)
            if len(left.split()) <= 3:
                cand = right.strip()
                if not (cand.startswith('{') and cand.endswith('}')):
                    cand = re.sub(r"\s+", " ", cand).strip('" ').strip()
                    if cand:
                        utts.append(cand)
    return utts


def load_model(model_id: str, device: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers not installed. pip install transformers torch")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        model.to(device)
    return tokenizer, model


def embed_utterances(
    texts: List[str], tokenizer, model, device: str, max_length: int, batch_size: int,
    layer_index: int, pooling: str
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch not installed. pip install torch")
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if device.startswith("cuda") and torch.cuda.is_available():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hiddens = out.hidden_states  # tuple[layer] of (B, T, H)
        if layer_index < 0:
            layer = hiddens[layer_index]
        else:
            layer = hiddens[layer_index]
        if pooling == "last":
            last_idx = attention_mask.sum(dim=1) - 1  # (B,)
            batch_vecs = []
            for b in range(layer.size(0)):
                idx = int(last_idx[b].item())
                v = layer[b, idx, :].detach().float().cpu().numpy()
                batch_vecs.append(v)
            vecs.extend(batch_vecs)
        else:
            # mean pooling over tokens with mask
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            masked = layer * mask
            sum_vec = masked.sum(dim=1)  # (B, H)
            lengths = mask.sum(dim=1).clamp(min=1)
            mean_vec = (sum_vec / lengths).detach().float().cpu().numpy()
            vecs.extend(list(mean_vec))
    return np.vstack(vecs) if vecs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def make_dataset(
    base_dir: str,
    baseline_dir: str,
    model_id: str,
    out_dir: str,
    layer_index: int,
    pooling: str,
    max_length: int,
    batch_size: int,
    max_utts_per_mode: int,
    device: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    os.makedirs(out_dir, exist_ok=True)

    tokenizer, model = load_model(model_id, device)

    def collect_pairs(root: str) -> Dict[str, Dict[str, str]]:
        return iter_scenarios(root)

    barrier_pairs = collect_pairs(base_dir)
    baseline_pairs = collect_pairs(baseline_dir if baseline_dir else base_dir)
    common = sorted(set(baseline_pairs.keys()) & set(barrier_pairs.keys()))

    texts: List[str] = []
    labels: List[str] = []
    modes: List[str] = []

    def add_samples(paths: Dict[str, str], mode: str):
        nonlocal texts, labels, modes
        count = 0
        for scenario_id in common:
            p = barrier_pairs[scenario_id].get(mode) if mode != "baseline" else baseline_pairs[scenario_id].get("baseline")
            if not p:
                continue
            ut = parse_utterances(read_text(p))
            for u in ut:
                texts.append(u)
                labels.append("barrier" if mode != "baseline" else "baseline")
                modes.append(mode)
                count += 1
                if max_utts_per_mode and count >= max_utts_per_mode:
                    return

    # Collect per mode
    add_samples(baseline_pairs, "baseline")
    for m in ("semantic", "cultural", "emotional"):
        add_samples(barrier_pairs, m)

    # Compute embeddings
    X = embed_utterances(texts, tokenizer, model, device, max_length, batch_size, layer_index, pooling)

    # Save raw dataset metadata
    meta = {
        "model_id": model_id,
        "layer_index": layer_index,
        "pooling": pooling,
        "num_samples": len(texts),
        "modes": modes[:100],  # preview
    }
    with open(os.path.join(out_dir, "probe_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if pd is not None:
        df = pd.DataFrame({"text": texts, "label": labels, "mode": modes})
        df.to_csv(os.path.join(out_dir, "probe_samples.csv"), index=False)
    np.save(os.path.join(out_dir, "probe_embeddings.npy"), X)

    return X, labels, modes


def run_pca_and_plot(X: np.ndarray, labels: List[str], modes: List[str], out_dir: str):
    if PCA is None or plt is None:
        print("PCA/plot skipped: scikit-learn or matplotlib not installed.")
        return
    os.makedirs(out_dir, exist_ok=True)
    p = PCA(n_components=2, random_state=42)
    try:
        Z = p.fit_transform(X)
    except Exception as e:
        print(f"PCA failed: {e}")
        return
    evr = p.explained_variance_ratio_.sum()
    if pd is not None:
        d = pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "label": labels, "mode": modes})
        # Baseline vs barrier
        fig, ax = plt.subplots(figsize=(6, 5))
        if sns is not None:
            sns.scatterplot(data=d, x="x", y="y", hue="label", ax=ax, s=12, alpha=0.7)
        else:
            for lab in sorted(set(labels)):
                sub = d[d["label"] == lab]
                ax.plot(sub["x"], sub["y"], ".", label=lab, alpha=0.7)
            ax.legend()
        ax.set_title(f"PCA (var={evr:.2f}) baseline vs barrier")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pca_baseline_vs_barrier.png"), dpi=150)
        plt.close(fig)

        # By mode
        fig, ax = plt.subplots(figsize=(7, 5))
        if sns is not None:
            sns.scatterplot(data=d, x="x", y="y", hue="mode", ax=ax, s=12, alpha=0.7)
        else:
            for md in sorted(set(modes)):
                sub = d[d["mode"] == md]
                ax.plot(sub["x"], sub["y"], ".", label=md, alpha=0.7)
            ax.legend()
        ax.set_title(f"PCA by mode (var={evr:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pca_by_mode.png"), dpi=150)
        plt.close(fig)
    else:
        print("pandas not installed; PCA plot skipped.")


def run_svm(X: np.ndarray, labels: List[str], out_dir: str):
    if SVC is None:
        print("SVM skipped: scikit-learn not installed.")
        return
    os.makedirs(out_dir, exist_ok=True)
    y_bin = np.array([1 if l != "baseline" else 0 for l in labels], dtype=int)
    y_mc = np.array(labels)

    def cv_eval(y, kernel: str) -> Tuple[float, float]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            clf = SVC(kernel=kernel, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            accs.append(float(accuracy_score(y[te], pred)))
            if y.ndim == 1 and y.dtype.kind in ("i", "U", "S"):
                avg = "binary" if np.unique(y).shape[0] == 2 else "macro"
                f1s.append(float(f1_score(y[te], pred, average=avg)))
        return float(np.mean(accs)), float(np.mean(f1s) if f1s else 0.0)

    results = {}
    if len(np.unique(y_bin)) == 2:
        acc, f1 = cv_eval(y_bin, kernel="linear")
        results["svm_binary_linear"] = {"acc": acc, "f1": f1}
        acc, f1 = cv_eval(y_bin, kernel="rbf")
        results["svm_binary_rbf"] = {"acc": acc, "f1": f1}

    # Multi-class over modes (baseline vs each barrier type)
    if len(np.unique(y_mc)) >= 2:
        # Relabel to 4 classes: baseline/semantic/cultural/emotional
        y4 = np.array([l if l in ("baseline", "semantic", "cultural", "emotional") else "other" for l in y_mc])
        keep = y4 != "other"
        if keep.sum() > 0 and len(np.unique(y4[keep])) >= 2:
            acc, f1 = cv_eval(y4[keep], kernel="linear")
            results["svm_multiclass_linear"] = {"acc": acc, "f1_macro": f1}
            acc, f1 = cv_eval(y4[keep], kernel="rbf")
            results["svm_multiclass_rbf"] = {"acc": acc, "f1_macro": f1}

    with open(os.path.join(out_dir, "probe_svm_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("SVM results:", json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Probe internal hidden states to detect barrier signals")
    ap.add_argument("--base_dir", type=str, required=True, help="Barrier results root (contains mode_* subdirs)")
    ap.add_argument("--baseline_dir", type=str, default="", help="Optional separate baseline results root (defaults to base_dir)")
    ap.add_argument("--model_id", type=str, required=True, help="HF model id or local path, e.g., Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out_dir", type=str, default="scripts/probe_reports", help="Output directory for figures and reports")
    ap.add_argument("--layer_index", type=int, default=-1, help="Hidden state layer index (-1=last)")
    ap.add_argument("--pooling", type=str, choices=["last", "mean"], default="last", help="Token pooling method")
    ap.add_argument("--max_length", type=int, default=256, help="Max tokens per utterance")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding")
    ap.add_argument("--max_utts_per_mode", type=int, default=2000, help="Cap utterances per mode for memory")
    ap.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = ap.parse_args()

    X, labels, modes = make_dataset(
        base_dir=args.base_dir,
        baseline_dir=args.baseline_dir,
        model_id=args.model_id,
        out_dir=args.out_dir,
        layer_index=args.layer_index,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_utts_per_mode=args.max_utts_per_mode,
        device=args.device,
    )

    if X.shape[0] == 0:
        print("No embeddings computed; check inputs.")
        return

    run_pca_and_plot(X, labels, modes, args.out_dir)
    run_svm(X, labels, args.out_dir)


if __name__ == "__main__":
    main()


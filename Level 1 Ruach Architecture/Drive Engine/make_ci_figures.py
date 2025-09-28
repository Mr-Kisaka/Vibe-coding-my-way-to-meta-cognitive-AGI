#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make paper figures for affect-vector of choice vs Random-vector results (lexicon-free).

Inputs:
  JSON array OR JSONL with objects:
    {"group":"affect"|"random", "text":"...", "sample_id":"optional"}

Outputs (default names):
  figures/fig1_distributions.png
  figures/fig2_effect_sizes.png
  figures/fig3_null_logodds.png   (if --permutations > 0)
  figures/fig4_null_tfidf.png     (if --permutations > 0)
"""

import os, json, re, math, argparse, pathlib, random
from collections import Counter
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

WORD_RE = re.compile(r"[A-Za-z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())

def load_records(path: str) -> List[Dict[str, Any]]:
    txt = pathlib.Path(path).read_text(encoding="utf-8").strip()
    # Try JSON array
    try:
        data = json.loads(txt)
        if isinstance(data, dict): data = [data]
        if not isinstance(data, list): raise ValueError
        return data
    except Exception:
        # Fallback: JSONL
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs

def welch_ci_mean(x: np.ndarray):
    n = len(x); m = float(np.mean(x))
    if n <= 1: return (m, float("nan"), float("nan"))
    s = float(np.std(x, ddof=1))
    tcrit = 1.96 if n > 60 else 2.00 if n >= 30 else 2.13 if n >= 20 else 2.57 if n >= 10 else 4.30
    half = tcrit * s / math.sqrt(n)
    return (m, m-half, m+half)

def bootstrap_ci(arr: np.ndarray, fn=np.mean, B=20000, alpha=0.05, seed=1337):
    rng = np.random.default_rng(seed)
    base = float(fn(arr))
    boots = [float(fn(rng.choice(arr, size=len(arr), replace=True))) for _ in range(B)]
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return base, float(lo), float(hi)

def hedges_g(x: np.ndarray, y: np.ndarray):
    nx, ny = len(x), len(y)
    sx2 = float(np.var(x, ddof=1)) if nx>1 else 0.0
    sy2 = float(np.var(y, ddof=1)) if ny>1 else 0.0
    sp2 = ((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2) if nx>1 and ny>1 else float("nan")
    d = (np.mean(x) - np.mean(y)) / math.sqrt(sp2) if sp2>0 else float("nan")
    J = 1 - (3/(4*(nx+ny)-9)) if (nx+ny)>2 else 1.0
    return d*J

def permutation_null_delta(A: np.ndarray, B: np.ndarray, iters=20000, seed=1337):
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([A, B])
    nA = len(A)
    null = []
    for _ in range(iters):
        rng.shuffle(pooled)
        null.append(float(np.mean(pooled[:nA]) - np.mean(pooled[nA:])))
    return np.array(null, float)

# -------- Metric 1: Discriminative log-odds (Dirichlet-smoothed) --------
def logodds_scores(tokens_docs: List[List[str]], groups: List[str], alpha0=0.01) -> np.ndarray:
    counts = {"affect": Counter(), "random": Counter()}
    for toks, g in zip(tokens_docs, groups):
        counts[g].update(toks)
    V = set(counts["affect"]) | set(counts["random"])
    nf = sum(counts["affect"].values()) or 1
    nr = sum(counts["random"].values()) or 1
    denom_f = nf + alpha0 * len(V)
    denom_r = nr + alpha0 * len(V)
    logodds = {}
    for w in V:
        cf = counts["affect"][w]; cr = counts["random"][w]
        pf = (cf + alpha0) / denom_f
        pr = (cr + alpha0) / denom_r
        logodds[w] = math.log((pf / max(1e-12, 1-pf))) - math.log((pr / max(1e-12, 1-pr)))
    scores = []
    for toks in tokens_docs:
        s = (sum(logodds.get(t, 0.0) for t in toks) / max(1, len(toks))) if toks else 0.0
        scores.append(float(s))
    return np.array(scores, float)

# -------- Metric 2: TF-IDF centroid margin --------
def tfidf_centroid_margin(tokens_docs: List[List[str]], groups: List[str]) -> np.ndarray:
    df = Counter()
    for toks in tokens_docs:
        for w in set(toks):
            df[w] += 1
    N = len(tokens_docs)
    vecs = []
    for toks in tokens_docs:
        tf = Counter(toks)
        v = {}
        for w, c in tf.items():
            idf = math.log((N + 1) / (df[w] + 1)) + 1.0
            v[w] = c * idf
        vecs.append(v)
    def centroid(idx_list):
        acc = {}
        for i in idx_list:
            for k, v in vecs[i].items():
                acc[k] = acc.get(k, 0.0) + v
        if not idx_list: return {}
        scale = 1.0 / len(idx_list)
        for k in list(acc.keys()):
            acc[k] *= scale
        return acc
    fear_idx = [i for i,g in enumerate(groups) if g=="affect"]
    rand_idx = [i for i,g in enumerate(groups) if g=="random"]
    c_fear = centroid(fear_idx)
    c_rand = centroid(rand_idx)
    def dot(a, b):
        if len(a) < len(b): a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())
    def norm(a):
        return math.sqrt(sum(v*v for v in a.values()))
    def cos(a, b):
        na, nb = norm(a), norm(b)
        if na == 0 or nb == 0: return 0.0
        return dot(a, b) / (na * nb)
    margins = [float(cos(v, c_fear) - cos(v, c_rand)) for v in vecs]
    return np.array(margins, float)

# -------- Plot helpers (matplotlib only, no seaborn, no styles) --------
def save_distributions(figpath, A, B, name, xlabel):
    plt.figure(figsize=(7.5, 4.0))
    # Overlaid histograms (density)
    bins = 20
    plt.hist(A, bins=bins, density=True, alpha=0.6, label="affect")
    plt.hist(B, bins=bins, density=True, alpha=0.6, label="random")
    # Means and Welch 95% CIs
    mA, A_lo, A_hi = welch_ci_mean(A); mB, B_lo, B_hi = welch_ci_mean(B)
    for m, lo, hi, lab_y in [(mA, A_lo, A_hi, plt.ylim()[1]*0.85),
                             (mB, B_lo, B_hi, plt.ylim()[1]*0.75)]:
        plt.axvline(m, linestyle="--")
        plt.plot([lo, hi], [lab_y, lab_y], linewidth=2)
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(name)
    plt.legend()
    pathlib.Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(figpath, dpi=300); plt.close()

def save_effectsize(figpath, deltas, ci_pairs, labels, title="Effect sizes (Δ = affect − Random)"):
    plt.figure(figsize=(6.5, 3.8))
    y = np.arange(len(deltas))[::-1]
    for i, (d, (lo, hi)) in enumerate(zip(deltas, ci_pairs)):
        plt.errorbar(d, y[i], xerr=[[d - lo], [hi - d]], fmt="o", capsize=4)
    plt.yticks(y, labels)
    plt.axvline(0, linestyle=":")
    plt.xlabel("Mean difference (Δ)"); plt.title(title)
    pathlib.Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(figpath, dpi=300); plt.close()

def save_null_hist(figpath, null_deltas, observed_delta, title):
    plt.figure(figsize=(6.5, 3.8))
    plt.hist(null_deltas, bins=40, density=True, alpha=0.8)
    plt.axvline(observed_delta, linestyle="--")
    plt.xlabel("Δ under label shuffles"); plt.ylabel("Density"); plt.title(title)
    pathlib.Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(figpath, dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path", help="Path to JSON or JSONL data")
    ap.add_argument("--outdir", default="figures", help="Output directory for figures")
    ap.add_argument("--permutations", type=int, default=0, help="Null permutations (0 to skip)")
    args = ap.parse_args()

    recs = load_records(args.input_path)
    # normalize groups
    recs = [r for r in recs if r.get("group") in ("affect","random") or r.get("group","").lower() in ("fear","random")]
    for r in recs:
        r["group"] = r["group"].lower().strip()
    groups = [r["group"] for r in recs]
    texts  = [r.get("text","") for r in recs]
    tokens_docs = [tokenize(t) for t in texts]

    # Scores
    scores_lo = logodds_scores(tokens_docs, groups)     # per-doc
    scores_tm = tfidf_centroid_margin(tokens_docs, groups)

    # Split
    A_lo = scores_lo[[i for i,g in enumerate(groups) if g=="affect"]]
    B_lo = scores_lo[[i for i,g in enumerate(groups) if g=="random"]]
    A_tm = scores_tm[[i for i,g in enumerate(groups) if g=="affect"]]
    B_tm = scores_tm[[i for i,g in enumerate(groups) if g=="random"]]

    # Δ and CIs
    mA_lo, loA_lo, hiA_lo = welch_ci_mean(A_lo); mB_lo, loB_lo, hiB_lo = welch_ci_mean(B_lo)
    mA_tm, loA_tm, hiA_tm = welch_ci_mean(A_tm); mB_tm, loB_tm, hiB_tm = welch_ci_mean(B_tm)
    delta_lo = float(np.mean(A_lo) - np.mean(B_lo))
    delta_tm = float(np.mean(A_tm) - np.mean(B_tm))
    _, dlo_lo, dhi_lo = bootstrap_ci(A_lo - A_lo + delta_lo, fn=lambda x: delta_lo)  # dummy, replaced below
    # Proper bootstrap for Δ:
    def boot_diff(A, B, Bn=20000, seed=1337):
        rng = np.random.default_rng(seed)
        deltas = []
        nA, nB = len(A), len(B)
        for _ in range(Bn):
            Ab = rng.choice(A, size=nA, replace=True)
            Bb = rng.choice(B, size=nB, replace=True)
            deltas.append(float(np.mean(Ab) - np.mean(Bb)))
        return np.quantile(deltas, [0.025, 0.975])
    dlo_lo, dhi_lo = boot_diff(A_lo, B_lo)
    dlo_tm, dhi_tm = boot_diff(A_tm, B_tm)

    # FIG 1: distributions
    save_distributions(os.path.join(args.outdir, "fig1a_logodds_dist.png"), A_lo, B_lo,
                       "Discriminative log-odds (per token)", "Log-odds score")
    save_distributions(os.path.join(args.outdir, "fig1b_tfidf_margin_dist.png"), A_tm, B_tm,
                       "TF-IDF centroid margin", "Cosine margin (affect − random)")

    # FIG 2: effect sizes (Δ with CI)
    save_effectsize(os.path.join(args.outdir, "fig2_effect_sizes.png"),
                    deltas=[delta_lo, delta_tm],
                    ci_pairs=[(dlo_lo, dhi_lo), (dlo_tm, dhi_tm)],
                    labels=["Log-odds", "TF-IDF margin"],
                    title="Effect sizes with 95% bootstrap CIs")

    # FIG 3–4: null histograms (optional)
    if args.permutations and args.permutations > 0:
        null_lo = permutation_null_delta(A_lo, B_lo, iters=args.permutations)
        null_tm = permutation_null_delta(A_tm, B_tm, iters=args.permutations)
        save_null_hist(os.path.join(args.outdir, "fig3_null_logodds.png"),
                       null_lo, delta_lo, "Null Δ (log-odds) via label shuffles")
        save_null_hist(os.path.join(args.outdir, "fig4_null_tfidf.png"),
                       null_tm, delta_tm, "Null Δ (TF-IDF margin) via label shuffles")

    print(f"Saved figures in: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()

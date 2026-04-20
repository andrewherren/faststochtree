#!/usr/bin/env python3
"""
bench/plot_results.py — generate blog-post figures from bench_results.csv

Produces four PNGs:
    bench/results/speedup_bart.png   — MCMC sampler wall-time progression
    bench/results/speedup_gfr.png    — GFR/XBART wall-time progression
    bench/results/rmse_bart.png      — MCMC sampler test-RMSE by version
    bench/results/rmse_gfr.png       — GFR/XBART test-RMSE by version

Usage:
    python bench/plot_results.py                         # uses default results CSV
    python bench/plot_results.py --csv path/to/file.csv  # explicit path
    python bench/plot_results.py --show                  # print results to terminal as well
    python bench/plot_results.py --n 200000 --p 10       # restrict results to a specific simulation
"""

import argparse
import csv
import os
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CSV           = SCRIPT_DIR / "results" / "bench_results.csv"
DEFAULT_OUT_BART      = SCRIPT_DIR / "results" / "speedup_bart.png"
DEFAULT_OUT_GFR       = SCRIPT_DIR / "results" / "speedup_gfr.png"
DEFAULT_OUT_RMSE_BART = SCRIPT_DIR / "results" / "rmse_bart.png"
DEFAULT_OUT_RMSE_GFR  = SCRIPT_DIR / "results" / "rmse_gfr.png"

# Which scenario to highlight (n_train, p).  Both tracks have full coverage here.
SCENARIO = (50_000, 50)

# Display labels and ordering for each track
BART_VERSIONS = [
    ("v1-naive-cpp",     "v1\nnaive C++"),
    ("v2-float",         "v2\nfloat"),
    ("v3-quantized",     "v3\nquantized"),
    ("v4-col-major",     "v4\ncol-major"),
    ("v5-fixed-depth",   "v5\nfixed-depth"),
    ("v6-branchless",    "v6\nbranchless"),
    ("v7-leaf-cache",    "v7\nleaf cache"),
    ("v8-scatter-buffer","v8\nscatter buf"),
    ("v9-multilane",     "v9\nmultilane"),
    ("v10-fused-gather", "v10\nfused gather"),
    ("v11-zero-alloc",   "v11\nzero alloc"),
    ("v12-leaf-counts",  "v12\nleaf counts"),
    ("v13-flat-obs",     "v13\nflat obs"),
]

GFR_VERSIONS = [
    ("gfr-v1-naive",             "v1\nnaive"),
    ("gfr-v2-presort",           "v2\npresort"),
    ("gfr-v3-two-stage",         "v3\ntwo-stage"),
    ("gfr-v4-pregather",         "v4\npregather"),
    ("gfr-v5-subsampling",       "v5\nsubsampling"),
    ("gfr-v6-node-parallel",     "v6\nnode-par"),
    ("gfr-v7-feat-parallel",     "v7\nfeat-par"),
    ("gfr-v8-zero-alloc",        "v8\nzero-alloc"),
    ("gfr-v9-histogram",         "v9\nhistogram"),
    ("gfr-v10-flat-node-range",  "v10\nflat-node"),
    ("gfr-v11-prefetch-hist",    "v11\nprefetch-hist"),
    ("gfr-v12-neon-partition",   "v12\nneon-partition"),
    ("gfr-v13-ws-reuse",         "v13\nws-reuse"),
]

# v1 had only 3 iterations (40 min/iter) — not comparable to other versions.
# Omit from RMSE chart; speedup chart is unaffected.
GFR_RMSE_VERSIONS = [v for v in GFR_VERSIONS if v[0] != "gfr-v1-naive"]

# ---------------------------------------------------------------------------
# Parse CSV
# ---------------------------------------------------------------------------

def load_data(csv_path):
    """Return (times, rmses, time_stds, rmse_stds) dicts keyed by (tag, n_train, p).

    For single-run CSVs (mc=False): time = min, rmse = from that row; stds are empty.
    For MC CSVs (mc=True): time/rmse = mean across all rows; stds = sample std.
    """
    import math
    with open(csv_path) as f:
        reader = csv.DictReader(f)

        sum_time:    dict = defaultdict(float)
        sum_time_sq: dict = defaultdict(float)
        sum_rmse:    dict = defaultdict(float)
        sum_rmse_sq: dict = defaultdict(float)
        counts:      dict = defaultdict(int)
        for row in reader:
            key = (row["tag"], int(row["n_train"]), int(row["p"]))
            t = float(row["time_ms"])
            r = float(row["rmse"])
            sum_time[key]    += t;  sum_time_sq[key]    += t * t
            sum_rmse[key]    += r;  sum_rmse_sq[key]    += r * r
            counts[key]      += 1
        times      = {k: sum_time[k] / counts[k] for k in counts}
        rmses      = {k: sum_rmse[k] / counts[k] for k in counts}
        time_stds  = {}
        rmse_stds  = {}
        for k, n in counts.items():
            if n > 1:
                time_stds[k] = math.sqrt(max(0.0, (sum_time_sq[k] - sum_time[k]**2 / n) / (n - 1)))
                rmse_stds[k] = math.sqrt(max(0.0, (sum_rmse_sq[k] - sum_rmse[k]**2 / n) / (n - 1)))
            else:
                time_stds[k] = 0.0
                rmse_stds[k] = 0.0

    return times, rmses, time_stds, rmse_stds

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_track(ax, labels, times, errs, base_color, title, plt, ticker):
    BEST     = "#4C72B0"
    GREY     = "#cccccc"

    valid = [t for t in times if t is not None]
    best_t = min(valid) if valid else None
    colors = [BEST if t == best_t else (GREY if t is None else base_color) for t in times]

    xs = list(range(len(labels)))
    bars = ax.bar(xs, [t if t is not None else 0 for t in times],
                  color=colors, width=0.6, zorder=3, edgecolor="white", linewidth=0.5)

    baseline = times[0] if times[0] is not None else None
    for i, (bar, t) in enumerate(zip(bars, times)):
        if t is None:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.5, "N/A",
                    ha="center", va="bottom", fontsize=8, color="#888888")
            continue
        # time label above bar
        ax.text(bar.get_x() + bar.get_width() / 2,
                t + (baseline or t) * 0.012,
                f"{t:.0f}s", ha="center", va="bottom", fontsize=9)
        # speedup badge inside bar (skip v1 baseline)
        if baseline and i > 0 and t > (baseline * 0.04):
            sp = baseline / t
            ax.text(bar.get_x() + bar.get_width() / 2,
                    t / 2,
                    f"{sp:.1f}×", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Wall time (seconds)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.6, len(labels) - 0.4)

    # overall speedup badge
    if baseline and best_t:
        ax.text(0.98, 0.97,
                f"Overall: {baseline / best_t:.0f}× faster",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10.5, fontweight="bold", color=BEST,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef3fb",
                          edgecolor=BEST, linewidth=1))


def plot_rmse_track(ax, labels, rmses, errs, base_color, title, sigma_true, plt, ticker):
    GREY     = "#cccccc"
    GREEN    = "#4C72B0"   # highlight: lowest RMSE

    valid = [r for r in rmses if r is not None]
    best_r = min(valid) if valid else None
    colors = [GREEN if r == best_r else (GREY if r is None else base_color) for r in rmses]

    xs = list(range(len(labels)))
    yerr = [e if e is not None else 0 for e in errs] if any(e for e in errs if e) else None
    bars = ax.bar(xs, [r if r is not None else 0 for r in rmses],
                  color=colors, width=0.6, zorder=3, edgecolor="white", linewidth=0.5,
                  yerr=yerr, capsize=4,
                  error_kw=dict(elinewidth=1.2, ecolor="#555555", capthick=1.2, zorder=4))

    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
    for bar, r, e in zip(bars, rmses, yerr if yerr else [0] * len(rmses)):
        if r is None:
            ax.text(bar.get_x() + bar.get_width() / 2, ax.get_ylim()[0], "N/A",
                    ha="center", va="bottom", fontsize=8, color="#888888")
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                r + (e or 0) + gap,
                f"{r:.4f}", ha="center", va="bottom", fontsize=8.5)

    # Reference line at sigma_true
    ax.axhline(sigma_true, color="#7f8c8d", linewidth=1.2, linestyle="--", zorder=4,
               label=f"σ_true = {sigma_true:.1f}")
    ax.legend(fontsize=9, framealpha=0.7)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Test RMSE", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.6, len(labels) - 0.4)

    # y-axis: start at 0, top must clear bar + error cap + rotated label
    if valid:
        valid_errs = [e for e in (yerr or []) if e]
        tops = [r + e for r, e in zip(
                    [r for r in rmses if r is not None],
                    valid_errs if valid_errs else [0] * len(valid))]
        hi = max(max(tops) if tops else max(valid), sigma_true)
        pad = hi * 0.35 or 0.05   # extra top room for rotated labels
        ax.set_ylim(0, hi + pad)


def make_figures(best_times, best_rmses, time_stds, rmse_stds, scenario, out_bart, out_gfr,
                 out_rmse_bart, out_rmse_gfr, sigma_true, show):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not found. Install with:  pip install matplotlib")
        raise

    n, p = scenario

    def times_for(versions):
        return [
            (best_times[(tag, n, p)] / 1000.0 if (tag, n, p) in best_times else None)
            for tag, _ in versions
        ]

    def time_errs_for(versions):
        return [
            (time_stds[(tag, n, p)] / 1000.0 if (tag, n, p) in time_stds else None)
            for tag, _ in versions
        ]

    def rmses_for(versions):
        return [
            (best_rmses[(tag, n, p)] if (tag, n, p) in best_rmses else None)
            for tag, _ in versions
        ]

    def rmse_errs_for(versions):
        return [
            (rmse_stds[(tag, n, p)] if (tag, n, p) in rmse_stds else None)
            for tag, _ in versions
        ]

    bart_labels = [lbl for _, lbl in BART_VERSIONS]
    gfr_labels  = [lbl for _, lbl in GFR_VERSIONS]

    # ---- BART speedup figure ----
    fig_bart, ax_bart = plt.subplots(figsize=(11, 5))
    fig_bart.patch.set_facecolor("white")
    plot_track(ax_bart, bart_labels, times_for(BART_VERSIONS), time_errs_for(BART_VERSIONS), "#4C72B0",
               f"BART sampler  (n={n:,}, p={p})  ·  200 trees · 200 burn-in · 1,000 samples · 1 thread",
               plt, ticker)
    fig_bart.tight_layout()
    if out_bart:
        fig_bart.savefig(out_bart, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_bart}")

    # ---- GFR speedup figure ----
    fig_gfr, ax_gfr = plt.subplots(figsize=(13, 5))
    fig_gfr.patch.set_facecolor("white")
    plot_track(ax_gfr, gfr_labels, times_for(GFR_VERSIONS), time_errs_for(GFR_VERSIONS), "#4C72B0",
               f"GFR / XBART  (n={n:,}, p={p})  ·  200 trees · 15 burn-in · 25 samples · 8 threads",
               plt, ticker)
    fig_gfr.tight_layout()
    if out_gfr:
        fig_gfr.savefig(out_gfr, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_gfr}")

    # ---- BART RMSE figure ----
    fig_rmse_bart, ax_rmse_bart = plt.subplots(figsize=(11, 5))
    fig_rmse_bart.patch.set_facecolor("white")
    plot_rmse_track(ax_rmse_bart, bart_labels, rmses_for(BART_VERSIONS), rmse_errs_for(BART_VERSIONS), "#4C72B0",
                    f"BART sampler — test RMSE  (n={n:,}, p={p})  ·  200 trees · 200 burn-in · 1,000 samples",
                    sigma_true, plt, ticker)
    fig_rmse_bart.tight_layout()
    if out_rmse_bart:
        fig_rmse_bart.savefig(out_rmse_bart, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_rmse_bart}")

    # ---- GFR RMSE figure ----
    gfr_rmse_labels = [lbl for _, lbl in GFR_RMSE_VERSIONS]
    fig_rmse_gfr, ax_rmse_gfr = plt.subplots(figsize=(13, 5))
    fig_rmse_gfr.patch.set_facecolor("white")
    plot_rmse_track(ax_rmse_gfr, gfr_rmse_labels, rmses_for(GFR_RMSE_VERSIONS), rmse_errs_for(GFR_RMSE_VERSIONS), "#4C72B0",
                    f"GFR / XBART — test RMSE  (n={n:,}, p={p})  ·  200 trees · 15 burn-in · 25 samples",
                    sigma_true, plt, ticker)
    fig_rmse_gfr.tight_layout()
    if out_rmse_gfr:
        fig_rmse_gfr.savefig(out_rmse_gfr, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_rmse_gfr}")

    if show:
        plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",           default=None,                   help="path to CSV (auto-selects based on --mc)")
    ap.add_argument("--out-bart",      default=DEFAULT_OUT_BART,       help="output PNG for BART speedup figure")
    ap.add_argument("--out-gfr",       default=DEFAULT_OUT_GFR,        help="output PNG for GFR speedup figure")
    ap.add_argument("--out-rmse-bart", default=DEFAULT_OUT_RMSE_BART,  help="output PNG for BART RMSE figure")
    ap.add_argument("--out-rmse-gfr",  default=DEFAULT_OUT_RMSE_GFR,   help="output PNG for GFR RMSE figure")
    ap.add_argument("--show",          action="store_true",            help="open interactive window")
    ap.add_argument("--n",             type=int, default=SCENARIO[0],  help="n_train scenario")
    ap.add_argument("--p",             type=int, default=SCENARIO[1],  help="p scenario")
    ap.add_argument("--sigma",         type=float, default=1.0,        help="sigma_true for RMSE reference line")
    args = ap.parse_args()

    csv_path = args.csv or DEFAULT_CSV
    times, rmses, time_stds, rmse_stds = load_data(csv_path)
    make_figures(times, rmses, time_stds, rmse_stds, (args.n, args.p),
                 args.out_bart, args.out_gfr,
                 args.out_rmse_bart, args.out_rmse_gfr,
                 args.sigma, args.show)

if __name__ == "__main__":
    main()

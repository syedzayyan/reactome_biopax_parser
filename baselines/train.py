"""
train.py — Multi-Run Training & Evaluation
==========================================
Runs each of the three baselines (REM, GBDT, Hawkes) for N_RUNS independent
seeds, aggregates metrics across runs, and produces matplotlib plots.

Usage:
    python train.py

Outputs:
    results.json          — raw per-run metrics for all models
    plots/metrics_bar.png — mean ± std bar chart across all tasks
    plots/per_run.png     — per-run line plots showing run-to-run variance
    plots/heatmap.png     — metric × model heat-map (colour = mean value)
"""

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# All model/pipeline logic lives in models.py
from models import (
    build_history_counts,
    evaluate,
    fit_existence_classifier,
    fit_gbdt,
    fit_hawkes,
    fit_pca_bias,
    fit_rem,
    load_graph,
    make_feature_fns,
    make_split,
    make_type_meta,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N_RUNS = 5
BASE_SEED = 42
SEEDS = [BASE_SEED + i * 13 for i in range(N_RUNS)]  # deterministic but diverse
BIOPAX_PATH = "../data/biopax3/R-HSA-168256.xml"
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = ["REM", "GBDT", "Hawkes"]
METRICS = ["auc", "f1", "pair_acc", "mae"]
METRIC_LABELS = {
    "auc": "AUC-ROC (Task 1)",
    "f1": "Macro-F1 (Task 2)",
    "pair_acc": "Pairwise Acc (Task 3)",
    "mae": "MAE (Task 3, ↓)",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRAPH ONCE  (featurisation is expensive — share across all runs)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading graph and featurising nodes (done once)...")
print("=" * 60)
G, feat_dim, parser = load_graph(BIOPAX_PATH)
_, edge_feature = make_feature_fns(G, feat_dim)
all_edge_types, etype_le, N_MARKS, t_max = make_type_meta(G)

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-RUN LOOP
# ─────────────────────────────────────────────────────────────────────────────

# results[model_name][split]["run_i"] = metric_dict
all_results: dict[str, dict[str, list[dict]]] = {
    m: {"val": [], "test": []} for m in MODELS
}

for run_idx, seed in enumerate(SEEDS):
    print(f"\n{'═' * 60}")
    print(f"  RUN {run_idx + 1}/{N_RUNS}  (seed={seed})")
    print(f"{'═' * 60}")

    random.seed(seed)
    np.random.seed(seed)

    # ── Split ────────────────────────────────────────────────────────────────
    train_edges, val_edges, test_edges, G_train = make_split(G, seed=seed)
    train_seq = sorted(train_edges, key=lambda e: e[2].get("time", 0))

    # ── Shared preprocessing ─────────────────────────────────────────────────
    feat_bias = fit_pca_bias(train_edges, edge_feature, N_MARKS, seed)
    get_history = build_history_counts(train_seq, etype_le, N_MARKS, t_max)
    exist_clf, sample_neg, _ = fit_existence_classifier(
        train_edges, G, edge_feature, seed
    )

    # Shared eval kwargs
    eval_kwargs = dict(
        exist_clf=exist_clf,
        edge_feature=edge_feature,
        etype_le=etype_le,
        all_edge_types=all_edge_types,
        N_MARKS=N_MARKS,
        t_max=t_max,
        sample_negative_fn=sample_neg,
    )

    # ── REM ──────────────────────────────────────────────────────────────────
    print(f"\n── REM  (run {run_idx + 1}) ──")
    rem = fit_rem(
        train_seq,
        edge_feature,
        feat_bias,
        get_history,
        etype_le,
        all_edge_types,
        N_MARKS,
        t_max=t_max,
    )
    for split, edges in [("val", val_edges), ("test", test_edges)]:
        m = evaluate(
            edges,
            name=f"REM | {split} | run {run_idx + 1}",
            predict_type_fn=rem["predict_type"],
            predict_order_score_fn=rem["predict_order_score"],
            use_rank_normalise=False,
            **eval_kwargs,
        )
        all_results["REM"][split].append(m)

    # ── GBDT ─────────────────────────────────────────────────────────────────
    print(f"\n── GBDT  (run {run_idx + 1}) ──")
    gbdt = fit_gbdt(
        train_seq,
        edge_feature,
        feat_bias,
        get_history,
        etype_le,
        all_edge_types,
        N_MARKS,
        seed,
        t_max=t_max,
    )
    for split, edges in [("val", val_edges), ("test", test_edges)]:
        m = evaluate(
            edges,
            name=f"GBDT | {split} | run {run_idx + 1}",
            predict_type_fn=gbdt["predict_type"],
            predict_order_score_fn=gbdt["predict_order_score"],
            use_rank_normalise=False,
            **eval_kwargs,
        )
        all_results["GBDT"][split].append(m)

    # ── Hawkes ───────────────────────────────────────────────────────────────
    print(f"\n── Hawkes  (run {run_idx + 1}) ──")
    hwk = fit_hawkes(train_seq, edge_feature, feat_bias, etype_le, N_MARKS)
    for split, edges in [("val", val_edges), ("test", test_edges)]:
        m = evaluate(
            edges,
            name=f"Hawkes | {split} | run {run_idx + 1}",
            predict_type_fn=hwk["predict_type"],
            predict_order_score_fn=hwk["predict_order_score"],
            use_rank_normalise=True,  # Hawkes uses rank-normalised scores
            **eval_kwargs,
        )
        all_results["Hawkes"][split].append(m)

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE & PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 70}")
print("  AGGREGATE RESULTS  (mean ± std across 5 runs, Test split)")
print(f"{'═' * 70}")

summary: dict[str, dict[str, dict]] = {}  # summary[model][metric] = {mean, std}

header = f"{'Model':<10}" + "".join(f"{METRIC_LABELS[m]:>22}" for m in METRICS)
print(header)
print("─" * len(header))

for model in MODELS:
    summary[model] = {}
    row = f"{model:<10}"
    for metric in METRICS:
        vals = [r[metric] for r in all_results[model]["test"]]
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        summary[model][metric] = {"mean": mu, "std": sigma, "runs": vals}
        row += f"  {mu:.4f}±{sigma:.4f}  "
    print(row)

# Save raw results to JSON
results_path = "results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nRaw results saved → {results_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use("ggplot")
COLORS = {"REM": "#E24A33", "GBDT": "#348ABD", "Hawkes": "#988ED5"}


# ── Plot 1: Mean ± Std bar chart across all metrics ──────────────────────────

fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5), sharey=False)
fig.suptitle(
    "Baseline Comparison — Test Set (mean ± std over 5 runs)", fontsize=13, y=1.02
)

for ax, metric in zip(axes, METRICS):
    x = np.arange(len(MODELS))
    for i, model in enumerate(MODELS):
        mu = summary[model][metric]["mean"]
        sigma = summary[model][metric]["std"]
        ax.bar(i, mu, color=COLORS[model], alpha=0.85, width=0.5, label=model)
        ax.errorbar(
            i,
            mu,
            yerr=sigma,
            fmt="none",
            color="black",
            capsize=5,
            linewidth=1.5,
            capthick=1.5,
        )

    ax.set_title(METRIC_LABELS[metric], fontsize=10, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    # Annotate bars
    for i, model in enumerate(MODELS):
        mu = summary[model][metric]["mean"]
        ax.text(
            i,
            mu * 0.5,
            f"{mu:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], alpha=0.85) for m in MODELS]
fig.legend(handles, MODELS, loc="upper right", fontsize=9, title="Model")
fig.tight_layout()
bar_path = OUTPUT_DIR / "metrics_bar.png"
fig.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {bar_path}")


# ── Plot 2: Per-run line plots showing run-to-run variance ───────────────────

n_metrics = len(METRICS)
fig, axes = plt.subplots(1, n_metrics, figsize=(18, 5), sharey=False)
fig.suptitle("Per-Run Metric Trajectories — Test Set", fontsize=13, y=1.02)

run_labels = [f"R{i + 1}" for i in range(N_RUNS)]

for ax, metric in zip(axes, METRICS):
    for model in MODELS:
        vals = summary[model][metric]["runs"]
        ax.plot(
            run_labels,
            vals,
            marker="o",
            color=COLORS[model],
            label=model,
            linewidth=1.8,
            markersize=5,
        )
        ax.fill_between(
            range(N_RUNS),
            [np.mean(vals) - np.std(vals)] * N_RUNS,
            [np.mean(vals) + np.std(vals)] * N_RUNS,
            color=COLORS[model],
            alpha=0.12,
        )

    ax.set_title(METRIC_LABELS[metric], fontsize=10, pad=6)
    ax.set_xlabel("Run", fontsize=9)
    ax.tick_params(labelsize=8)

handles = [
    plt.Line2D([0], [0], color=COLORS[m], marker="o", linewidth=1.8, markersize=5)
    for m in MODELS
]
fig.legend(handles, MODELS, loc="upper right", fontsize=9, title="Model")
fig.tight_layout()
run_path = OUTPUT_DIR / "per_run.png"
fig.savefig(run_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {run_path}")


# ── Plot 3: Heat-map — metric × model  (colour = mean value) ─────────────────

data_matrix = np.array(
    [[summary[model][metric]["mean"] for metric in METRICS] for model in MODELS]
)  # shape: (n_models, n_metrics)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels(
    [METRIC_LABELS[m] for m in METRICS], fontsize=9, rotation=20, ha="right"
)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODELS, fontsize=10)
ax.set_title(
    "Model × Metric Heat-Map (mean over 5 runs, Test Set)", fontsize=11, pad=10
)

for i, model in enumerate(MODELS):
    for j, metric in enumerate(METRICS):
        mu = summary[model][metric]["mean"]
        sigma = summary[model][metric]["std"]
        text_color = "white" if mu < 0.35 or mu > 0.75 else "black"
        ax.text(
            j,
            i,
            f"{mu:.3f}\n±{sigma:.3f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
        )

fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Mean value")
fig.tight_layout()
hm_path = OUTPUT_DIR / "heatmap.png"
fig.savefig(hm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {hm_path}")

print(f"\n{'═' * 60}")
print("  All done. Plots saved to ./plots/")
print(f"{'═' * 60}")

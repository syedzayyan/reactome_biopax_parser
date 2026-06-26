#!/usr/bin/env bash
# run_comparison_5pathways.sh — HPC batch: tune + benchmark RGCN, TGAT and STHN
# across all 5 Reactome pathways (Immune, Cell Cycle, Hemostasis, Metabolism,
# Gene Expression) for a direct three-way comparison.
#
# For each pathway, runs:
#   ./run_benchmarks.sh --models sthn,rgcn,tgat --tune \
#       --biopax data/<RHSA>.xml --pathway-name <Name> \
#       --out-dir results/comparison/<slug> "$@"
#
# Per run_benchmarks.sh's model dispatch:
#   RGCN  — always runs both no-time and +time conditions (12 conditions).
#            Neighbour sampling is uniform random; no leakage concern.
#   TGAT  — with --time-ablation: 12 conditions. Neighbour table built from
#            training edges only (_build_neighbor_table), cached per epoch.
#   STHN  — with --time-ablation: 12 conditions. Adjacency table built from
#            training edges only (_build_train_adjacency), cached on first call.
#
# Results land under results/comparison/<pathway_slug>/.
#
# Usage
# -----
#   ./run_comparison_5pathways.sh [extra run_benchmarks.sh flags...]
#
# Examples
# --------
#   # Full sweep on GPU, with Hits@K and time ablation for TGAT + STHN
#   ./run_comparison_5pathways.sh --gpu --hits --time-ablation
#
#   # Smaller/faster first pass
#   ./run_comparison_5pathways.sh --gpu --n-trials 20 --epochs 100 --seeds 3
#
# Any flag accepted by run_benchmarks.sh (--gpu, --cpu, --tune, --n-trials,
# --epochs, --seeds, --hits, --time-ablation, --venv, --storage, ...) is
# forwarded as-is to every per-pathway invocation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# name:R-HSA-id pairs, in run order.
PATHWAYS=(
    "Immune:R-HSA-168256"
    "Cell Cycle:R-HSA-1640170"
    "Hemostasis:R-HSA-109582"
    "Metabolism:R-HSA-1430728"
    "Gene Expression:R-HSA-74160"
)

for entry in "${PATHWAYS[@]}"; do
    NAME="${entry%%:*}"
    RHSA="${entry##*:}"
    SLUG="$(echo "$NAME" | tr '[:upper:] ' '[:lower:]_')"

    echo
    echo "############################################################"
    echo "# COMPARISON (RGCN + TGAT + STHN): $NAME ($RHSA)"
    echo "# -> results/comparison/$SLUG"
    echo "############################################################"

    ./run_benchmarks.sh \
        --models sthn,rgcn,tgat \
        --tune \
        --biopax "data/${RHSA}.xml" \
        --pathway-name "$NAME" \
        --out-dir "results/comparison/${SLUG}" \
        "$@"
done

echo
echo "############################################################"
echo "# All 5 pathways done."
echo "# Results under results/comparison/<pathway>/"
echo "############################################################"

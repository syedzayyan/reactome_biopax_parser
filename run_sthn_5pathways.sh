#!/usr/bin/env bash
# run_sthn_5pathways.sh — HPC batch: tune + benchmark FISH STHN across 5
# Reactome pathways (Immune, Cell Cycle, Hemostasis, Metabolism, Gene
# Expression).
#
# For each pathway, runs:
#   ./run_benchmarks.sh --models sthn --tune --biopax <pathway.xml> \
#       --pathway-name <Name> --out-dir results/sthn/<slug> "$@"
#
# which (per run_benchmarks.sh's sthn case):
#   1. tunes STHN via tune_sthn.py (Optuna sweep)
#        -> best_params_sthn.json, optuna_history_sthn.csv
#   2. benchmarks STHN with the tuned params via test_sthn.py
#      (6 conditions x --seeds seeds)
#        -> results_sthn.json, results_sthn.csv
#
# All outputs land under results/sthn/<slug>/.
#
# Usage
# -----
#   ./run_sthn_5pathways.sh [extra run_benchmarks.sh flags...]
#
# Examples
# --------
#   # Full sweep on GPU, with Hits@K
#   ./run_sthn_5pathways.sh --gpu --hits
#
#   # Smaller/faster sweep (e.g. for a first pass)
#   ./run_sthn_5pathways.sh --gpu --n-trials 20 --epochs 100 --seeds 3
#
#   # Use a pre-built venv elsewhere
#   ./run_sthn_5pathways.sh --gpu --venv /path/to/.venv
#
# Any flag accepted by run_benchmarks.sh (--gpu, --cpu, --n-trials, --epochs,
# --seeds, --hits, --venv, --storage, ...) is forwarded as-is to every
# per-pathway invocation.

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
    echo "# STHN: $NAME ($RHSA) -> results/sthn/$SLUG"
    echo "############################################################"

    ./run_benchmarks.sh \
        --models sthn \
        --tune \
        --biopax "data/${RHSA}.xml" \
        --pathway-name "$NAME" \
        --out-dir "results/sthn/${SLUG}" \
        "$@"
done

echo
echo "############################################################"
echo "# All 5 pathways done. Results under results/sthn/<pathway>/"
echo "############################################################"

#!/bin/bash
#SBATCH -J fish_rgcn
#SBATCH -o %x.o%j
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

# Tune + evaluate RGCN on all 5 Reactome pathways with the corrected
# 3-way node-disjoint split (70/15/15 train/val/test).
# Runs Immune first so main-paper numbers are available before the rest finish.
#
# Hyperparameter tuning uses val_mask only (no test leakage).
# Final evaluation uses test_mask only.
# split_cache.json files are deleted so the old 20%-unseen format is replaced
# by the new 3-way cache (same rng seed => same splits across all three models).

set -euo pipefail

cd reactome_biopax_parser
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

REPO="$(pwd)"

# Remove stale split caches (old single-group 20% format).
find "${REPO}/results/comparison" -name "split_cache.json" -delete 2>/dev/null || true
echo "[setup] Removed stale split_cache.json files."

# run_comparison_5pathways.sh loops Immune→CellCycle→Hemostasis→Metabolism→GeneExpr.
# --models rgcn overrides the default sthn,rgcn,tgat.
# --tune is already hard-coded inside run_comparison_5pathways.sh.
./run_comparison_5pathways.sh \
    --models rgcn \
    --n-trials 100 \
    --seeds 5 \
    --gpu \
    --hits \
    --time-ablation

echo
echo "════════════════════════════════════════════════════════"
echo "  RGCN done. Results under results/comparison/<pathway>/results_rgcn.csv"
echo "════════════════════════════════════════════════════════"

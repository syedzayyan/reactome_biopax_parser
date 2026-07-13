#!/bin/bash
#SBATCH -J fish_tgat
#SBATCH -o %x.o%j
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 72:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

# Tune + evaluate TGAT on all 5 Reactome pathways with the corrected
# 3-way node-disjoint split (70/15/15 train/val/test).
# Runs Immune first so main-paper numbers are available before the rest finish.
#
# Hyperparameter tuning uses val_mask only (no test leakage).
# Final evaluation uses test_mask only.
# Submit after sbatch_tune_eval_rgcn.sh so split_cache files already exist;
# same rng seed guarantees identical splits regardless of which runs first.
# 72h limit: Immune TGAT alone takes ~17h (60 runs x ~1040s); 5 pathways ~50-60h.

set -euo pipefail

cd reactome_biopax_parser
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

./run_comparison_5pathways.sh \
    --models tgat \
    --n-trials 100 \
    --seeds 5 \
    --gpu \
    --hits \
    --time-ablation

echo
echo "════════════════════════════════════════════════════════"
echo "  TGAT done. Results under results/comparison/<pathway>/results_tgat.csv"
echo "════════════════════════════════════════════════════════"

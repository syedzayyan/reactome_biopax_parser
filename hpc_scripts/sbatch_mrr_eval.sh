#!/bin/bash
#SBATCH -J mrr_eval
#SBATCH -o %x.o%j
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

# Re-evaluate STHN and TGAT with compute_hits=True on all 5 pathways.
# Does NOT re-tune — reads existing best_params_*.json from the comparison
# results directory. Results overwrite results_sthn.csv / results_tgat.csv
# so gen_comparison_tables.py picks up MRR automatically.
#
# Requires the code fix in fish_sthn.py and fish_tgat.py (t_query threading
# into _hits_at_k_sthn / new _hits_at_k_tgat). Run this after pulling the
# updated baselines/GNNs/fish_sthn.py and fish_tgat.py.
#
# Estimated runtime per (model, pathway):
#   STHN: ~30–90 min (depends on epochs + pathway size)
#   TGAT: ~60–120 min
#   Total: ≤ 24 h on a single GPU with 10 sequential jobs.

set -euo pipefail

cd reactome_biopax_parser
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

REPO="$(pwd)"
GNN_DIR="${REPO}/baselines/GNNs"
RESULTS_ROOT="${REPO}/results/comparison"
DATA_DIR="${REPO}/data/biopax3"

declare -A RHSA
RHSA[immune]="R-HSA-168256"
RHSA[cell_cycle]="R-HSA-1640170"
RHSA[hemostasis]="R-HSA-109582"
RHSA[metabolism]="R-HSA-1430728"
RHSA[gene_expression]="R-HSA-74160"

declare -A NAMES
NAMES[immune]="Immune"
NAMES[cell_cycle]="Cell Cycle"
NAMES[hemostasis]="Hemostasis"
NAMES[metabolism]="Metabolism"
NAMES[gene_expression]="Gene Expression"

SLUGS=(immune cell_cycle hemostasis metabolism gene_expression)

cd "$GNN_DIR"

for SLUG in "${SLUGS[@]}"; do
    RHSA_ID="${RHSA[$SLUG]}"
    NAME="${NAMES[$SLUG]}"
    OUT="${RESULTS_ROOT}/${SLUG}"
    BIOPAX="${DATA_DIR}/${RHSA_ID}.xml"
    SPLIT_CACHE="${OUT}/split_cache.json"

    echo
    echo "════════════════════════════════════════════════════════"
    echo "  MRR eval — STHN  ${NAME}  (${RHSA_ID})"
    echo "════════════════════════════════════════════════════════"

    STHN_PARAMS="${OUT}/best_params_sthn.json"
    if [[ ! -f "$STHN_PARAMS" ]]; then
        echo "  WARNING: $STHN_PARAMS not found, skipping STHN for $NAME"
    else
        python test_sthn.py \
            --biopax  "$BIOPAX" \
            --pathway-name "$NAME" \
            --tune-json "$STHN_PARAMS" \
            --split-cache "$SPLIT_CACHE" \
            --hits \
            --time-ablation \
            --seeds 5 \
            --gpu \
            --out-json "${OUT}/results_sthn.json" \
            --out-csv  "${OUT}/results_sthn.csv"
    fi

    echo
    echo "════════════════════════════════════════════════════════"
    echo "  MRR eval — TGAT  ${NAME}  (${RHSA_ID})"
    echo "════════════════════════════════════════════════════════"

    TGAT_PARAMS="${OUT}/best_params_tgat.json"
    if [[ ! -f "$TGAT_PARAMS" ]]; then
        echo "  WARNING: $TGAT_PARAMS not found, skipping TGAT for $NAME"
    else
        python test_tgat.py \
            --biopax  "$BIOPAX" \
            --pathway-name "$NAME" \
            --tune-json "$TGAT_PARAMS" \
            --split-cache "$SPLIT_CACHE" \
            --hits \
            --time-ablation \
            --seeds 5 \
            --gpu \
            --out-json "${OUT}/results_tgat.json" \
            --out-csv  "${OUT}/results_tgat.csv"
    fi
done

echo
echo "════════════════════════════════════════════════════════"
echo "  All done. Run gen_comparison_tables.py to rebuild LaTeX."
echo "════════════════════════════════════════════════════════"

#!/bin/bash
#SBATCH -J mrr_resume
#SBATCH -o %x.o%j
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

# Continuation of sbatch_mrr_eval.sh.
# Immune (STHN+TGAT) and Cell Cycle STHN are already done.
# This script runs: Cell Cycle TGAT + Hemostasis + Metabolism + Gene Expression.
#
# Uses neg_chunk_size=50000 by default (fish_tgat.py train() default).
# Estimated total: ~16h on a single A100.

set -euo pipefail

cd reactome_biopax_parser
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

REPO="$(pwd)"
GNN_DIR="${REPO}/baselines/GNNs"
DATA_DIR="${REPO}/data/biopax3"

declare -A RHSA
RHSA[cell_cycle]="R-HSA-1640170"
RHSA[hemostasis]="R-HSA-109582"
RHSA[metabolism]="R-HSA-1430728"
RHSA[gene_expression]="R-HSA-74160"

declare -A NAMES
NAMES[cell_cycle]="Cell Cycle"
NAMES[hemostasis]="Hemostasis"
NAMES[metabolism]="Metabolism"
NAMES[gene_expression]="Gene Expression"

# Cell Cycle TGAT first (STHN already done), then full STHN+TGAT for the rest.
SLUGS=(cell_cycle hemostasis metabolism gene_expression)

cd "$GNN_DIR"

for SLUG in "${SLUGS[@]}"; do
    RHSA_ID="${RHSA[$SLUG]}"
    NAME="${NAMES[$SLUG]}"
    OUT="${REPO}/comparison/${SLUG}"
    BIOPAX="${DATA_DIR}/${RHSA_ID}.xml"
    SPLIT_CACHE="${OUT}/split_cache.json"

    # STHN — skip cell_cycle (already done), run the rest.
    if [[ "$SLUG" != "cell_cycle" ]]; then
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
echo "  Done. Run gen_comparison_tables.py to rebuild LaTeX."
echo "════════════════════════════════════════════════════════"

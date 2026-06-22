#!/usr/bin/env bash
# run_benchmarks.sh — tune and/or run FISH GNN baselines (RGCN, TGAT, HGCN).
#
# Default behaviour (no flags): run all three benchmarks without tuning, on
# the Immune pathway, using the project .venv.
#
# Usage
# -----
#   ./run_benchmarks.sh [FLAGS]
#
# Examples
# --------
#   # Tune all models, then benchmark, on GPU
#   ./run_benchmarks.sh --tune --gpu
#
#   # Benchmark only TGAT and HGCN using a pre-featurised pickle
#   ./run_benchmarks.sh --models tgat,hgcn --pickle /data/immune.pkl
#
#   # Quick smoke-test (2 seeds, 50 epochs, no tuning)
#   ./run_benchmarks.sh --epochs 50 --seeds 2
#
#   # Run RGCN with tuned params, compute Hits@K, save to custom dir
#   ./run_benchmarks.sh --models rgcn --tune --hits --out-dir ./results/run1
#
# Flags
# -----
#   --biopax PATH         BioPAX XML file to parse. Default: the Immune pathway
#                         (data/biopax3/R-HSA-168256.xml). Mutually exclusive
#                         with --pickle.
#
#   --pickle PATH         Pre-featurised graph pickle. Skips parsing and node
#                         featurisation (much faster for repeated runs).
#                         Mutually exclusive with --biopax.
#
#   --pathway-name NAME   Human-readable name embedded in output filenames and
#                         summary tables. Default: Immune.
#
#   --models LIST         Comma-separated subset of models to run.
#                         Choices: rgcn, tgat, hgcn, sthn. Default: rgcn,tgat,hgcn.
#
#   --tune                Run Optuna hyperparameter sweeps before benchmarking.
#                         Uses the tuned params for the benchmark run.
#                         Disabled by default.
#
#   --n-trials N          Optuna trials per model (only used when --tune is set).
#                         Default: 50.
#
#   --epochs N            Training epochs per benchmark run. Default: 200.
#
#   --seeds N             Number of random seeds per condition. Default: 5.
#
#   --hidden N            Encoder hidden dimension. Default: 128.
#
#   --n-layers N          Number of encoder layers. Default: 2.
#
#   --lr LR               Learning rate. Default: 0.001.
#
#   --dropout D           Dropout rate. Default: 0.2.
#
#   --n-negatives N       Negative samples per positive (existence head).
#                         Default: 10.
#
#   --order-weight W      Scalar weight applied to the order-task loss on top of
#                         the per-condition CORAL down-weighting. Default: 1.0.
#
#   --time-target MODE    Transform applied to raw step times for the regression
#                         order target. Choices: min_max, log_min_max, rank.
#                         Default: min_max.
#
#   --hits                Compute Hits@K (K=1,10,100) against all-nodes and
#                         type-matched pools. Expensive; off by default.
#
#   --smart-train         Use type-matched negatives during training (instead of
#                         uniform random). Off by default.
#
#   --compartment-emb PATH
#                         Path to a .npz/.pkl/.tsv file of pre-computed
#                         compartment embeddings (e.g. GO2Vec). When omitted,
#                         the +compartment conditions use a one-hot vocabulary.
#
#   --out-dir DIR         Directory for all result files. Created if absent.
#                         Default: ./results.
#
#   --gpu                 Force CUDA. Errors if CUDA is unavailable.
#
#   --cpu                 Force CPU. Useful when MPS auto-detection misbehaves.
#
#   --venv PATH           Path to the Python virtual environment to activate.
#                         Default: .venv (project-local).
#
#   --include-rgat        (RGCN only) Also run RGAT conditions, doubling the
#                         condition count from 12 to 24.
#
#   --n-heads N           (TGAT / HGCN / STHN) Attention heads. Default: 2
#                         (TGAT, STHN), 1 (HGCN). Overrides all three when set
#                         here.
#
#   --time-dim N          (TGAT / STHN) Time2Vec output dimension. Default: 64.
#
#   --edge-feat-dim N     (TGAT / STHN) Edge-type embedding dimension. Default: 32.
#
#   --max-neighbors N     (TGAT only) Most-recent neighbours per node during
#                         encoding. Default: 20.
#
#   --max-edges N         (STHN only) Most-recent training edges per node
#                         (per endpoint) included in each pair's subgraph.
#                         Default: 20.
#
#   --window-size N       (STHN only) Patch/window size for the subgraph
#                         mixer. Must divide 2 * --max-edges. Default: 5.
#
#   --channel-expansion-factor N
#                         (STHN only) Feed-forward expansion factor inside
#                         each mixer layer. Default: 2.
#
#   --use-attention       (HGCN only) Use the attention variant of
#                         HypergraphConv. Off by default.
#
#   --storage URL         Optuna storage URL shared by all tuners.
#                         Default: sqlite:///optuna_studies.db (in --out-dir).

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────

BIOPAX="data/biopax3/R-HSA-168256.xml"
PICKLE=""
PATHWAY_NAME="Immune"
MODELS="rgcn,tgat,hgcn"
TUNE=false
N_TRIALS=50
EPOCHS=200
SEEDS=5
HIDDEN=128
N_LAYERS=2
LR=0.001
DROPOUT=0.2
N_NEGATIVES=10
ORDER_WEIGHT=1.0
TIME_TARGET="min_max"
HITS=false
SMART_TRAIN=false
COMPARTMENT_EMB=""
OUT_DIR="results"
DEVICE_FLAG=""
VENV=".venv"
INCLUDE_RGAT=false
N_HEADS_TGAT=2
N_HEADS_HGCN=1
N_HEADS_STHN=2
TIME_DIM=64
EDGE_FEAT_DIM=32
MAX_NEIGHBORS=20
MAX_EDGES=20
WINDOW_SIZE=5
CHANNEL_EXPANSION_FACTOR=2
USE_ATTENTION=false
STORAGE=""            # resolved to file inside OUT_DIR after parsing

# ── argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --biopax)            BIOPAX="$2";           PICKLE="";      shift 2 ;;
        --pickle)            PICKLE="$2";           BIOPAX="";      shift 2 ;;
        --pathway-name)      PATHWAY_NAME="$2";                     shift 2 ;;
        --models)            MODELS="$2";                           shift 2 ;;
        --tune)              TUNE=true;                             shift   ;;
        --n-trials)          N_TRIALS="$2";                         shift 2 ;;
        --epochs)            EPOCHS="$2";                           shift 2 ;;
        --seeds)             SEEDS="$2";                            shift 2 ;;
        --hidden)            HIDDEN="$2";                           shift 2 ;;
        --n-layers)          N_LAYERS="$2";                         shift 2 ;;
        --lr)                LR="$2";                               shift 2 ;;
        --dropout)           DROPOUT="$2";                          shift 2 ;;
        --n-negatives)       N_NEGATIVES="$2";                      shift 2 ;;
        --order-weight)      ORDER_WEIGHT="$2";                     shift 2 ;;
        --time-target)       TIME_TARGET="$2";                      shift 2 ;;
        --hits)              HITS=true;                             shift   ;;
        --smart-train)       SMART_TRAIN=true;                      shift   ;;
        --compartment-emb)   COMPARTMENT_EMB="$2";                  shift 2 ;;
        --out-dir)           OUT_DIR="$2";                          shift 2 ;;
        --gpu)               DEVICE_FLAG="--gpu";                   shift   ;;
        --cpu)               DEVICE_FLAG="--cpu";                   shift   ;;
        --venv)              VENV="$2";                             shift 2 ;;
        --include-rgat)      INCLUDE_RGAT=true;                     shift   ;;
        --n-heads)           N_HEADS_TGAT="$2"; N_HEADS_HGCN="$2"; N_HEADS_STHN="$2"; shift 2 ;;
        --time-dim)          TIME_DIM="$2";                         shift 2 ;;
        --edge-feat-dim)     EDGE_FEAT_DIM="$2";                    shift 2 ;;
        --max-neighbors)     MAX_NEIGHBORS="$2";                    shift 2 ;;
        --max-edges)         MAX_EDGES="$2";                        shift 2 ;;
        --window-size)       WINDOW_SIZE="$2";                      shift 2 ;;
        --channel-expansion-factor) CHANNEL_EXPANSION_FACTOR="$2";  shift 2 ;;
        --use-attention)     USE_ATTENTION=true;                    shift   ;;
        --storage)           STORAGE="$2";                          shift 2 ;;
        -h|--help)
            grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ── validation ────────────────────────────────────────────────────────────────

if [[ -z "$BIOPAX" && -z "$PICKLE" ]]; then
    echo "Error: no input graph. Provide --biopax or --pickle." >&2
    exit 1
fi

if [[ -n "$BIOPAX" && ! -f "$BIOPAX" ]]; then
    echo "Error: BioPAX file not found: $BIOPAX" >&2
    echo "Download it with:" >&2
    echo "  python -c \"from reactome_graphs import download_single_biopax_file_by_pathway_id; download_single_biopax_file_by_pathway_id('R-HSA-168256', 'data/biopax3/')\"" >&2
    exit 1
fi

if [[ -n "$PICKLE" && ! -f "$PICKLE" ]]; then
    echo "Error: pickle file not found: $PICKLE" >&2
    exit 1
fi

# ── environment ───────────────────────────────────────────────────────────────

# Resolve all paths to absolute before any cd so nothing breaks later.
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="${REPO_ROOT}/baselines/GNNs"

PYTHON="python"
if [[ -d "${REPO_ROOT}/${VENV}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/${VENV}/bin/activate"
    echo "[run_benchmarks] activated venv: ${REPO_ROOT}/${VENV}"
elif [[ -d "$VENV" ]]; then
    source "${VENV}/bin/activate"
    echo "[run_benchmarks] activated venv: $VENV"
else
    echo "[run_benchmarks] warning: venv '$VENV' not found, using system python"
fi

# Resolve OUT_DIR to absolute.
if [[ "$OUT_DIR" = /* ]]; then
    ABS_OUT_DIR="$OUT_DIR"
else
    ABS_OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi
mkdir -p "$ABS_OUT_DIR"

# Optuna storage lives inside ABS_OUT_DIR unless the caller set --storage.
[[ -z "$STORAGE" ]] && STORAGE="sqlite:///${ABS_OUT_DIR}/optuna_studies.db"

# Input graph argument — always absolute.
if [[ -n "$PICKLE" ]]; then
    [[ "$PICKLE" = /* ]] && INPUT_ARG="--pickle $PICKLE" \
                         || INPUT_ARG="--pickle ${REPO_ROOT}/${PICKLE}"
else
    [[ "$BIOPAX" = /* ]] && INPUT_ARG="--biopax $BIOPAX" \
                         || INPUT_ARG="--biopax ${REPO_ROOT}/${BIOPAX}"
fi

cd "$SCRIPT_DIR"

# Optional flags shared by all scripts
COMMON_OPTS="--pathway-name $PATHWAY_NAME"
COMMON_OPTS+=" --epochs $EPOCHS --seeds $SEEDS"
COMMON_OPTS+=" --hidden $HIDDEN --n-layers $N_LAYERS"
COMMON_OPTS+=" --lr $LR --dropout $DROPOUT"
COMMON_OPTS+=" --n-negatives $N_NEGATIVES --order-weight $ORDER_WEIGHT"
COMMON_OPTS+=" --time-target $TIME_TARGET"
[[ -n "$DEVICE_FLAG" ]] && COMMON_OPTS+=" $DEVICE_FLAG"
[[ -n "$COMPARTMENT_EMB" ]] && COMMON_OPTS+=" --compartment-embeddings $COMPARTMENT_EMB"
$HITS        && COMMON_OPTS+=" --hits"
$SMART_TRAIN && COMMON_OPTS+=" --smart-train"

TUNE_OPTS="--pathway-name $PATHWAY_NAME --n-trials $N_TRIALS --storage $STORAGE"
[[ -n "$DEVICE_FLAG" ]] && TUNE_OPTS+=" $DEVICE_FLAG"

log() { echo; echo "════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════"; }

# ── model loop ────────────────────────────────────────────────────────────────

IFS=',' read -ra MODEL_LIST <<< "$MODELS"

for MODEL in "${MODEL_LIST[@]}"; do
    case "$MODEL" in

    # ── RGCN ─────────────────────────────────────────────────────────────────
    rgcn)
        if $TUNE; then
            log "Tuning RGCN ($PATHWAY_NAME, $N_TRIALS trials)"
            $PYTHON tune.py $INPUT_ARG $TUNE_OPTS \
                --out-json "${ABS_OUT_DIR}/best_params_rgcn.json" \
                --out-csv  "${ABS_OUT_DIR}/optuna_history_rgcn.csv"
            BEST_RGCN="${ABS_OUT_DIR}/best_params_rgcn.json"
        fi

        log "Benchmarking RGCN ($PATHWAY_NAME, $SEEDS seeds × $EPOCHS epochs)"
        RGCN_EXTRA=""
        $INCLUDE_RGAT && RGCN_EXTRA+=" --include-rgat"

        # If we just tuned, read the best params and override the defaults.
        if $TUNE && [[ -f "${ABS_OUT_DIR}/best_params_rgcn.json" ]]; then
            BEST="$PYTHON -c \
\"import json,sys; p=json.load(open('${ABS_OUT_DIR}/best_params_rgcn.json'))['best_params']; \
sys.stdout.write(' '.join(['--'+k.replace('_','-')+' '+str(v) for k,v in p.items() if k not in ('epochs','time_target')]))\""
            TUNED_ARGS=$(eval $BEST)
            RGCN_EXTRA+=" $TUNED_ARGS"
        fi

        $PYTHON test.py $INPUT_ARG $COMMON_OPTS $RGCN_EXTRA \
            --out-json "${ABS_OUT_DIR}/results_rgcn.json" \
            --out-csv  "${ABS_OUT_DIR}/results_rgcn.csv"
        ;;

    # ── TGAT ─────────────────────────────────────────────────────────────────
    tgat)
        if $TUNE; then
            log "Tuning TGAT ($PATHWAY_NAME, $N_TRIALS trials)"
            $PYTHON tune_tgat.py $INPUT_ARG $TUNE_OPTS \
                --out-json "${ABS_OUT_DIR}/best_params_tgat.json" \
                --out-csv  "${ABS_OUT_DIR}/optuna_history_tgat.csv"
        fi

        log "Benchmarking TGAT ($PATHWAY_NAME, $SEEDS seeds × $EPOCHS epochs)"
        TGAT_EXTRA="--n-heads $N_HEADS_TGAT --time-dim $TIME_DIM"
        TGAT_EXTRA+=" --edge-feat-dim $EDGE_FEAT_DIM --max-neighbors $MAX_NEIGHBORS"

        if $TUNE && [[ -f "${ABS_OUT_DIR}/best_params_tgat.json" ]]; then
            BEST="$PYTHON -c \
\"import json,sys; p=json.load(open('${ABS_OUT_DIR}/best_params_tgat.json'))['best_params']; \
sys.stdout.write(' '.join(['--'+k.replace('_','-')+' '+str(v) for k,v in p.items() if k not in ('epochs','time_target')]))\""
            TGAT_EXTRA+=" $(eval $BEST)"
        fi

        $PYTHON test_tgat.py $INPUT_ARG $COMMON_OPTS $TGAT_EXTRA \
            --out-json "${ABS_OUT_DIR}/results_tgat.json" \
            --out-csv  "${ABS_OUT_DIR}/results_tgat.csv"
        ;;

    # ── HGCN ─────────────────────────────────────────────────────────────────
    hgcn)
        if $TUNE; then
            log "Tuning HGCN ($PATHWAY_NAME, $N_TRIALS trials)"
            $PYTHON tune_hgcn.py $INPUT_ARG $TUNE_OPTS \
                --out-json "${ABS_OUT_DIR}/best_params_hgcn.json" \
                --out-csv  "${ABS_OUT_DIR}/optuna_history_hgcn.csv"
        fi

        log "Benchmarking HGCN ($PATHWAY_NAME, $SEEDS seeds × $EPOCHS epochs)"
        HGCN_EXTRA="--n-heads $N_HEADS_HGCN"
        $USE_ATTENTION && HGCN_EXTRA+=" --use-attention"

        if $TUNE && [[ -f "${ABS_OUT_DIR}/best_params_hgcn.json" ]]; then
            BEST="$PYTHON -c \
\"import json,sys; p=json.load(open('${ABS_OUT_DIR}/best_params_hgcn.json'))['best_params']; \
sys.stdout.write(' '.join(['--'+k.replace('_','-')+' '+str(v) for k,v in p.items() if k not in ('epochs','time_target','use_attention','use_type_hyperedges')]))\""
            HGCN_EXTRA+=" $(eval $BEST)"
        fi

        $PYTHON test_hgcn.py $INPUT_ARG $COMMON_OPTS $HGCN_EXTRA \
            --out-json "${ABS_OUT_DIR}/results_hgcn.json" \
            --out-csv  "${ABS_OUT_DIR}/results_hgcn.csv"
        ;;

    # ── STHN ─────────────────────────────────────────────────────────────────
    sthn)
        if $TUNE; then
            log "Tuning STHN ($PATHWAY_NAME, $N_TRIALS trials)"
            $PYTHON tune_sthn.py $INPUT_ARG $TUNE_OPTS \
                --out-json "${ABS_OUT_DIR}/best_params_sthn.json" \
                --out-csv  "${ABS_OUT_DIR}/optuna_history_sthn.csv"
        fi

        log "Benchmarking STHN ($PATHWAY_NAME, $SEEDS seeds × $EPOCHS epochs)"
        STHN_EXTRA="--n-heads $N_HEADS_STHN --time-dim $TIME_DIM"
        STHN_EXTRA+=" --edge-feat-dim $EDGE_FEAT_DIM --max-edges $MAX_EDGES"
        STHN_EXTRA+=" --window-size $WINDOW_SIZE"
        STHN_EXTRA+=" --channel-expansion-factor $CHANNEL_EXPANSION_FACTOR"

        if $TUNE && [[ -f "${ABS_OUT_DIR}/best_params_sthn.json" ]]; then
            BEST="$PYTHON -c \
\"import json,sys; p=json.load(open('${ABS_OUT_DIR}/best_params_sthn.json'))['best_params']; \
sys.stdout.write(' '.join(['--'+k.replace('_','-')+' '+str(v) for k,v in p.items() if k not in ('epochs','time_target')]))\""
            STHN_EXTRA+=" $(eval $BEST)"
        fi

        $PYTHON test_sthn.py $INPUT_ARG $COMMON_OPTS $STHN_EXTRA \
            --out-json "${ABS_OUT_DIR}/results_sthn.json" \
            --out-csv  "${ABS_OUT_DIR}/results_sthn.csv"
        ;;

    *)
        echo "Unknown model: $MODEL (choices: rgcn, tgat, hgcn, sthn)" >&2
        exit 1
        ;;
    esac
done

# ── done ─────────────────────────────────────────────────────────────────────

log "All done. Results written to: ${ABS_OUT_DIR}/"
echo
ls -1 "${ABS_OUT_DIR}/"

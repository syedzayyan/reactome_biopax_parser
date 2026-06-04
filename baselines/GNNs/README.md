# GNN Baselines — FISH Benchmark

This directory contains three GNN baselines evaluated on the **FISH** (Functional Interaction Step Hierarchy) benchmark derived from Reactome BioPAX pathways.

All baselines share the same data pipeline, evaluation protocol, and three-task formulation, so their numbers are directly comparable.

---

## What is being benchmarked

Each Reactome pathway is parsed into a directed, heterogeneous, temporal graph where:
- **Nodes** are biological entities (proteins, small molecules, complexes, DNA, RNA).
- **Edges** carry a `type` (reaction, catalysis, complex\_component, translocation, …) and a `time` (the pathway step index at which the interaction first appears).

Three tasks are evaluated simultaneously on a **semi-inductive split** (20 % of nodes held out; test edges bridge seen ↔ unseen endpoints):

| Task | What it predicts | Metric |
|---|---|---|
| **Existence** | Does this (src, dst) edge exist? | AUC-ROC vs random negatives; AUC-ROC vs type-matched partner-excluded negatives |
| **Type** | What kind of edge is it? | Top-1 accuracy; macro AUC |
| **Order** | At what pathway step does it occur? | Spearman ρ; pairwise ordering accuracy |

---

## Models

### RGCN (`fish_rgcn.py` + `test.py`)

Relational Graph Convolutional Network (Schlichtkrull et al. 2018) with optional extensions:

| Flag | Description |
|---|---|
| `architecture rgcn/rgat` | RGCN or Relational GAT encoder |
| `time_encoding` | GraphMixer-style cosine time encoding folded into messages |
| `use_compartment` | Per-node cellular-compartment feature concatenated before projection |
| `order_mode regression/corn/coral` | MSE, CORN ordinal, or CORAL ordinal order head |

Full factorial: RGCN × {time, no-time} × {compartment, no-compartment} × {regression, CORN, CORAL} = **12 conditions** (24 with RGAT via `--include-rgat`).

---

### TGAT (`fish_tgat.py` + `test_tgat.py`)

Temporal Graph Attention Network (Xu et al. 2020) using TGM library's `TemporalAttention` and `Time2Vec` modules.

Key design choices vs RGCN:
- Incoming neighbours gathered by timestamp (most-recent `max_neighbors` per node).
- Time deltas `(ref_time − edge_time)` encoded with Time2Vec before attention.
- Edge type embedded to `edge_feat_dim` dimensions, passed as edge features to `TemporalAttention`.
- Residual skip from initial projected features (TGAT paper convention).
- Same per-type input projections and identical three-head decoder as RGCN.

Conditions: {no-compartment, +compartment} × {regression, CORN, CORAL} = **6 conditions**.

---

### HGCN (`fish_hgcn.py` + `test_hgcn.py`)

Hypergraph Convolutional Network (Bai et al. 2021) using PyG's `HypergraphConv`.

**Why hypergraphs?** Biochemical reactions naturally involve multiple simultaneous participants (multiple substrates → multiple products). A directed pairwise graph decomposes each reaction into individual edges; a hypergraph can represent all participants of one event as a single set, preserving the higher-order grouping structure.

**How heterogeneity is handled** (HypergraphConv is not heterogeneous by design):
1. Per-node-type input projections collapse the native feature spaces into a shared hidden dimension (same as RGCN).
2. A learnable **node-type embedding** is added to the projected features so the uniform hypergraph layers still receive type signals.

**Hyperedge construction** (from the NX graph, no HyperNetX required):

| Hyperedge type | Enabled | Construction |
|---|---|---|
| Time-step hyperedges | always | All src + dst nodes of training edges at the same pathway step → one hyperedge. Captures "all participants in one event". |
| Edge-type hyperedges | `--type-hyperedges` | All nodes connected by the same edge type → one hyperedge. Captures "all nodes in the same biochemical role". |

Conditions: {no-type-he, +type-he} × {no-compartment, +compartment} × {regression, CORN, CORAL} = **12 conditions**.

Optionally use the attention variant of HypergraphConv (`--use-attention --n-heads N`).

---

## Quick start

All scripts accept a BioPAX XML file (parser runs on the fly) or a pre-featurised pickle.

### RGCN

```bash
# From baselines/GNNs/
python test.py \
    --biopax ../../data/biopax3/R-HSA-168256.xml \
    --pathway-name Immune \
    --epochs 200 \
    --seeds 5
```

### TGAT

```bash
python test_tgat.py \
    --biopax ../../data/biopax3/R-HSA-168256.xml \
    --pathway-name Immune \
    --epochs 200 \
    --seeds 5
```

### HGCN

```bash
python test_hgcn.py \
    --biopax ../../data/biopax3/R-HSA-168256.xml \
    --pathway-name Immune \
    --epochs 200 \
    --seeds 5
```

### Using a pre-featurised pickle (faster restarts)

```python
import pickle
from reactome_graphs import ReactomeBioPAX, NodeFeaturiser

parser = ReactomeBioPAX(uniprot_accession_num=True)
G = parser.parse_biopax_into_networkx("data/biopax3/R-HSA-168256.xml")
featuriser = NodeFeaturiser(G, xref_dict={}, parser=parser)
featuriser.download_and_store()
featuriser.featurise()

with open("immune.pkl", "wb") as f:
    pickle.dump(G, f)
```

Then pass `--pickle immune.pkl` instead of `--biopax` to any runner.

---

## Hyperparameter tuning

Each model has a dedicated Optuna sweep targeting the headline condition (+ compartment / CORAL):

```bash
python tune.py      --pickle immune.pkl --n-trials 50   # RGCN
python tune_tgat.py --pickle immune.pkl --n-trials 50   # TGAT
python tune_hgcn.py --pickle immune.pkl --n-trials 50   # HGCN
```

Results land in `best_params.json` / `best_params_tgat.json` / `best_params_hgcn.json`. All three sweeps share the same SQLite study file (`optuna_studies.db`) but use separate study names.

Pass tuned values back via CLI flags (`--hidden`, `--lr`, `--dropout`, etc.).

---

## CLI options

### Common to all three runners

| Option | Default | Description |
|---|---|---|
| `--epochs` | 200 | Training epochs per run |
| `--hidden` | 128 | Encoder hidden dimension |
| `--n-layers` | 2 | Number of encoder layers |
| `--lr` | 1e-3 | Learning rate |
| `--dropout` | 0.2 | Dropout rate |
| `--order-weight` | 1.0 | Extra weight on the order loss |
| `--time-target` | `min_max` | Order regression target (`min_max`, `log_min_max`, `rank`) |
| `--seeds` | 5 | Random seeds to average over |
| `--n-negatives` | 10 | Negative samples per positive (existence head) |
| `--hits` | off | Compute Hits@K (expensive) |
| `--gpu` / `--cpu` | auto | Force device |
| `--compartment-embeddings` | — | Path to `.npz`/`.pkl`/`.tsv` of GO2Vec compartment vectors |

### TGAT-only

| Option | Default | Description |
|---|---|---|
| `--n-heads` | 2 | Attention heads in `TemporalAttention` |
| `--time-dim` | 64 | Time2Vec output dimension |
| `--edge-feat-dim` | 32 | Edge-type embedding dimension |
| `--max-neighbors` | 20 | Most-recent neighbours used per node |

### HGCN-only

| Option | Default | Description |
|---|---|---|
| `--use-attention` | off | Use the attention variant of `HypergraphConv` |
| `--n-heads` | 1 | Attention heads (only with `--use-attention`) |

---

## Output files

| File | Contents |
|---|---|
| `results.json` / `.csv` | RGCN per-run metrics |
| `results_tgat.json` / `.csv` | TGAT per-run metrics |
| `results_hgcn.json` / `.csv` | HGCN per-run metrics |
| `best_params.json` | RGCN best hyperparameters |
| `best_params_tgat.json` | TGAT best hyperparameters |
| `best_params_hgcn.json` | HGCN best hyperparameters |
| `optuna_history.csv` / `_tgat.csv` / `_hgcn.csv` | Per-trial Optuna history |
| `optuna_studies.db` | Shared persistent SQLite study file |

---

## Reported metrics

| Key | Description |
|---|---|
| `exist_auc` | AUC-ROC, uniform-random negative destinations |
| `exist_auc_smart` | AUC-ROC, type-matched partner-excluded negatives |
| `mrr` / `mrr_type` | Mean reciprocal rank (all-nodes / type-matched pool) |
| `hits_at_1/10/100` | Hits@K, all-nodes pool |
| `hits_at_1/10/100_type` | Hits@K, type-matched pool |
| `type_top1` | Top-1 accuracy on edge-type classification |
| `type_macro_auc` | Macro AUC on edge-type classification |
| `order_spearman` | Spearman ρ between predicted and true step order |
| `order_pairwise_acc` | Fraction of randomly sampled pairs correctly ordered |
| `ind_*` | Same metrics on the inductive subset (both endpoints unseen) |

---

## Dependencies

```
torch >= 2.2
torch-geometric >= 2.5          # HypergraphConv, RGCNConv, RGATConv
tgm-lib                         # TGAT: TemporalAttention + Time2Vec
coral-pytorch                   # CORN / CORAL ordinal losses
optuna                          # hyperparameter tuning
```

Install via:

```bash
uv pip install -e ".[gnn]"
uv pip install coral-pytorch optuna git+https://github.com/tgm-team/tgm.git
```

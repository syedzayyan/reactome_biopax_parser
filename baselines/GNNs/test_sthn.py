"""
test_sthn.py — FISH STHN benchmark: 6 conditions x N seeds on semi-inductive split.

Conditions
----------
  {STHN, STHN + compartment} x {regression, CORN, CORAL}

  Pass --time-ablation to also sweep the time-encoding axis (12 conditions):
  {no-time, +time} x {no-compartment, +compartment} x {regression, CORN, CORAL}.
  "+time" keeps STHN's native TimeEncode edge-time encoding (the default);
  "no-time" drops the TimeEncode submodule from the subgraph mixer entirely
  (see fish_sthn.py's use_time_encoding), mirroring fish_rgcn.py's time
  ablation so the two baselines are comparable on this axis. Off by default
  since it doubles runtime on top of an already mixer-heavy baseline.

Usage
-----
    python test_sthn.py \
        --biopax ./data/biopax3/R-HSA-168256.xml \
        --pathway-name Immune \
        --epochs 200 \
        --seeds 5

Outputs
-------
    Per-run results to --out-json (default results_sthn.json) and
    --out-csv (default results_sthn.csv). Summary table printed to stdout.

All non-Hits@K metrics are identical to test.py / test_tgat.py — all
baselines call the same evaluation code so their numbers can be compared
directly. Hits@K/MRR for STHN are a sampled-ranking approximation; see
_hits_at_k_sthn in fish_sthn.py.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from fish_sthn import FISHSTHN, build_dataset, evaluate, train


# ── condition definition ────────────────────────────────────────────────────


@dataclass
class Condition:
    name: str
    use_time: bool
    use_compartment: bool
    order_mode: str

    @property
    def loss_weights(self) -> tuple[float, float, float]:
        if self.order_mode == "coral":
            return (1.0, 1.0, 1.0 / 20)
        return (1.0, 1.0, 1.0)


def _make_conditions(include_time_ablation: bool = False) -> list[Condition]:
    # Factorial over time x compartment x decoder, matching fish_rgcn.py's
    # _make_conditions naming convention ("+t" / "+c" suffixes). Time defaults
    # to always-on (STHN's native TimeEncode) unless the ablation is requested,
    # which adds the no-time variants and doubles the condition count.
    conditions = []
    time_choices = (False, True) if include_time_ablation else (True,)
    for time_enc in time_choices:
        for comp in (False, True):
            for decoder in ("regression", "corn", "coral"):
                name_parts = ["sthn"]
                if time_enc:
                    name_parts.append("+t")
                if comp:
                    name_parts.append("+c")
                name_parts.append(f"/{decoder}")
                conditions.append(Condition(
                    name=" ".join(name_parts).replace(" /", "/"),
                    use_time=time_enc,
                    use_compartment=comp,
                    order_mode=decoder,
                ))
    return conditions


# Conditions are rebuilt at CLI parse time once we know --time-ablation.
CONDITIONS = _make_conditions()


# ── graph loading ────────────────────────────────────────────────────────────


def load_and_featurise(biopax_path: str, reaction_partners: bool, include_complexes: bool):
    from reactome_graphs import NodeFeaturiser, ReactomeBioPAX
    parser = ReactomeBioPAX(uniprot_accession_num=True)
    G = parser.parse_biopax_into_networkx(
        biopax_path,
        reaction_partners=reaction_partners,
        include_complexes=include_complexes,
    )
    featuriser = NodeFeaturiser(G, xref_dict={}, parser=parser)
    featuriser.download_and_store()
    featuriser.featurise()
    return G


def load_pickled_graph(pickle_path: str):
    import pickle
    with open(pickle_path, "rb") as fh:
        return pickle.load(fh)


def load_compartment_embeddings(path: str) -> dict:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        npz = np.load(p, allow_pickle=False)
        if "keys" not in npz or "vectors" not in npz:
            raise ValueError(f"{path}: .npz must contain 'keys' and 'vectors' arrays.")
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in npz["keys"]]
        vectors = npz["vectors"].astype(np.float32)
        return {k: vectors[i] for i, k in enumerate(keys)}
    if suffix in (".pkl", ".pickle"):
        import pickle
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: pickle must contain a dict.")
        return {k: np.asarray(v, dtype=np.float32) for k, v in obj.items()}
    if suffix in (".tsv", ".txt"):
        out = {}
        with open(p) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                key, *vals = parts
                out[key] = np.asarray(vals, dtype=np.float32)
        return out
    raise ValueError(f"Unrecognised compartment-embeddings file type: {suffix}")


# ── seeding ──────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── single run ───────────────────────────────────────────────────────────────


def run_one(
    G,
    condition: Condition,
    data: "FISHData",
    seed: int,
    epochs: int,
    hidden: int,
    n_layers: int,
    n_heads: int,
    lr: float,
    n_order_bins: int,
    time_dim: int,
    edge_feat_dim: int,
    max_edges: int,
    window_size: int,
    channel_expansion_factor: int,
    device: str,
    verbose: bool = False,
    compute_hits: bool = False,
    smart_train: bool = False,
    n_negatives: int = 10,
    time_target: str = "min_max",
    dropout: float = 0.2,
    order_weight: float = 1.0,
) -> dict:
    set_seed(seed)
    model = FISHSTHN(
        data,
        hidden=hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        time_dim=time_dim,
        edge_feat_dim=edge_feat_dim,
        max_edges=max_edges,
        window_size=window_size,
        channel_expansion_factor=channel_expansion_factor,
        order_mode=condition.order_mode,
        n_order_bins=n_order_bins,
        n_negatives=n_negatives,
        use_compartment=condition.use_compartment,
        smart_negatives=smart_train,
        use_time_encoding=condition.use_time,
    )
    base_lw = condition.loss_weights
    effective_lw = (base_lw[0], base_lw[1], base_lw[2] * order_weight)

    start = time.time()
    train(
        model, data,
        epochs=epochs, lr=lr,
        loss_weights=effective_lw,
        device=device,
        log_every=epochs if not verbose else max(epochs // 10, 1),
    )
    metrics = evaluate(model, data, device=device, seed=seed,
                       compute_hits=compute_hits)
    metrics["seconds"] = round(time.time() - start, 1)
    metrics["seed"] = seed
    metrics["condition"] = condition.name
    metrics["use_time_encoding"] = condition.use_time
    metrics["use_compartment"] = condition.use_compartment
    metrics["order_mode"] = condition.order_mode
    metrics["time_target"] = time_target
    metrics["n_negatives"] = n_negatives
    metrics["smart_train"] = smart_train
    return metrics


# ── summary table ────────────────────────────────────────────────────────────


def summary_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    metric_cols = [
        "exist_auc", "exist_auc_smart",
        "mrr", "mrr_type",
        "hits_at_1", "hits_at_10", "hits_at_100",
        "hits_at_1_type", "hits_at_10_type", "hits_at_100_type",
        "type_macro_auc", "type_top1",
        "order_spearman", "order_pairwise_acc",
        "ind_exist_auc", "ind_exist_auc_smart",
        "ind_mrr", "ind_mrr_type",
        "ind_hits_at_1", "ind_hits_at_10", "ind_hits_at_100",
        "ind_hits_at_1_type", "ind_hits_at_10_type", "ind_hits_at_100_type",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    grouped = df.groupby("condition", sort=False)[metric_cols]
    means = grouped.mean()
    stds = grouped.std()
    out = pd.DataFrame(index=means.index)
    for col in metric_cols:
        out[col] = [
            f"{m:+.3f} ± {s:.3f}" if "spearman" in col
            else f"{m:.3f} ± {s:.3f}"
            for m, s in zip(means[col], stds[col])
        ]
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--biopax", help="Path to a .xml/.owl BioPAX Level 3 file")
    src.add_argument("--pickle", help="Path to a pre-parsed and featurised pickle")
    parser.add_argument("--pathway-name", default="Pathway")
    parser.add_argument("--reaction-partners", action="store_true")
    parser.add_argument("--no-complexes", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1,
                        help="Number of Transformer layers in the subgraph mixer.")
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-order-bins", type=int, default=20)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--edge-feat-dim", type=int, default=32,
                        help="Dimension of the edge-type embedding fed to the mixer.")
    parser.add_argument("--max-edges", type=int, default=20,
                        help="Maximum number of (most-recent) training edges "
                             "per node included in the subgraph (per endpoint; "
                             "the mixer sees up to 2x this many).")
    parser.add_argument("--window-size", type=int, default=5,
                        help="Patch/window size for the subgraph mixer. "
                             "Must divide 2 * max-edges.")
    parser.add_argument("--channel-expansion-factor", type=int, default=2,
                        help="Feed-forward expansion factor inside each mixer layer.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--order-weight", type=float, default=1.0)
    parser.add_argument(
        "--time-target", default="min_max",
        choices=["min_max", "log_min_max", "rank"],
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hits", action="store_true",
                        help="Compute Hits@K (sampled approximation, see fish_sthn._hits_at_k_sthn).")
    parser.add_argument("--smart-train", action="store_true")
    parser.add_argument("--n-negatives", type=int, default=10)
    parser.add_argument(
        "--time-ablation", action="store_true",
        help="Also run the no-time-encoding variants (TimeEncode dropped from "
             "the subgraph mixer). Default: time-encoding always on, 6 "
             "conditions. With --time-ablation: 12 conditions, ~2x runtime.",
    )
    parser.add_argument("--out-json", default="results_sthn.json")
    parser.add_argument("--out-csv", default="results_sthn.csv")
    parser.add_argument("--compartment-embeddings", default=None)
    parser.add_argument(
        "--tune-json", default="best_params_sthn.json",
        help="Path to best_params_sthn.json written by a tune_sthn.py sweep. "
             "Loaded automatically if the file exists; CLI flags take precedence.",
    )
    parser.add_argument(
        "--split-cache", default=None,
        help="Path to a JSON file for caching the semi-inductive split per seed. "
             "Shared with RGCN and TGAT to guarantee identical train/test masks.",
    )
    args = parser.parse_args()

    # Auto-load tuned hyperparameters; CLI flags win over JSON values.
    _tune_map = {
        "hidden": "hidden", "n_layers": "n_layers", "n_heads": "n_heads",
        "lr": "lr", "dropout": "dropout", "time_dim": "time_dim",
        "edge_feat_dim": "edge_feat_dim", "max_edges": "max_edges",
        "window_size": "window_size",
        "channel_expansion_factor": "channel_expansion_factor",
        "order_weight": "order_weight", "time_target": "time_target",
        "epochs": "epochs",
    }
    import json as _json
    from pathlib import Path as _Path
    _tune_path = _Path(args.tune_json)
    if _tune_path.exists():
        _tuned = _json.loads(_tune_path.read_text()).get("best_params", {})
        _overridden = []
        for _jkey, _dest in _tune_map.items():
            if _jkey in _tuned and getattr(args, _dest) == parser.get_default(_dest):
                setattr(args, _dest, _tuned[_jkey])
                _overridden.append(f"{_dest}={_tuned[_jkey]}")
        if _overridden:
            print(f"[main] Loaded from {_tune_path}: {', '.join(_overridden)}")
    elif args.tune_json != "best_params_sthn.json":
        parser.error(f"--tune-json file not found: {args.tune_json}")

    if (2 * args.max_edges) % args.window_size != 0:
        parser.error(
            f"2 * --max-edges ({2 * args.max_edges}) must be divisible by "
            f"--window-size ({args.window_size})."
        )

    if args.gpu and args.cpu:
        parser.error("Cannot pass --gpu and --cpu together.")
    if args.device is not None:
        device = args.device
    elif args.gpu:
        if not torch.cuda.is_available():
            parser.error("--gpu requested but CUDA is not available.")
        device = "cuda"
    elif args.cpu:
        device = "cpu"
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[main] Using device={device}")

    # Rebuild conditions list based on whether the time ablation was requested.
    global CONDITIONS
    CONDITIONS = _make_conditions(args.time_ablation)

    print(f"[main] Loading {args.pathway_name}")
    if args.biopax:
        G = load_and_featurise(
            args.biopax,
            reaction_partners=args.reaction_partners,
            include_complexes=not args.no_complexes,
        )
    else:
        G = load_pickled_graph(args.pickle)
    print(f"[main] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    compartment_embeddings = None
    if args.compartment_embeddings:
        print(f"[main] Loading compartment embeddings from {args.compartment_embeddings}")
        compartment_embeddings = load_compartment_embeddings(args.compartment_embeddings)
        emb_dim = len(next(iter(compartment_embeddings.values())))
        print(f"[main]   {len(compartment_embeddings)} terms, dim={emb_dim}")
    else:
        print("[main] No compartment-embeddings file; compartment conditions use "
              "a one-hot vocabulary (learned via the input projection).")

    n_runs = len(CONDITIONS) * args.seeds
    print(f"[main] {len(CONDITIONS)} conditions x {args.seeds} seeds = {n_runs} runs")
    print(f"[main] epochs={args.epochs}  hidden={args.hidden}  n_heads={args.n_heads}  "
          f"max_edges={args.max_edges}  window_size={args.window_size}  lr={args.lr}")

    order_modes = list(dict.fromkeys(c.order_mode for c in CONDITIONS))

    results = []
    run_idx = 0
    for seed in range(args.seeds):
        data_for_mode = {
            mode: build_dataset(
                G,
                split="semi_inductive",
                unseen_node_frac=0.20,
                order_mode=mode,
                n_order_bins=args.n_order_bins,
                compartment_embeddings=compartment_embeddings,
                time_target=args.time_target,
                split_cache=args.split_cache,
                seed=seed,
            )
            for mode in order_modes
        }
        for condition in CONDITIONS:
            run_idx += 1
            print(f"\n[{run_idx}/{n_runs}] {condition.name}  seed={seed}")
            m = run_one(
                G, condition, data_for_mode[condition.order_mode], seed,
                epochs=args.epochs, hidden=args.hidden,
                n_layers=args.n_layers, n_heads=args.n_heads,
                lr=args.lr, n_order_bins=args.n_order_bins,
                time_dim=args.time_dim, edge_feat_dim=args.edge_feat_dim,
                max_edges=args.max_edges, window_size=args.window_size,
                channel_expansion_factor=args.channel_expansion_factor,
                device=device,
                verbose=args.verbose,
                compute_hits=args.hits,
                smart_train=args.smart_train,
                n_negatives=args.n_negatives,
                time_target=args.time_target,
                dropout=args.dropout,
                order_weight=args.order_weight,
            )
            print(
                f"  exist={m['exist_auc']:.3f}/{m.get('exist_auc_smart', float('nan')):.3f}  "
                f"ind_exist={m.get('ind_exist_auc', float('nan')):.3f}/"
                f"{m.get('ind_exist_auc_smart', float('nan')):.3f}  "
                f"type_top1={m['type_top1']:.3f}  "
                f"order_rho={m['order_spearman']:+.3f}  "
                f"pair_acc={m['order_pairwise_acc']:.3f}  "
                f"({m['seconds']}s)"
            )
            results.append(m)

    json_safe = []
    for r in results:
        r_clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in r.items()}
        json_safe.append(r_clean)
    with open(args.out_json, "w") as fh:
        json.dump(json_safe, fh, indent=2)
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print(f"\n[main] Wrote {args.out_json} and {args.out_csv}")

    print("\n" + "=" * 72)
    print(f"SUMMARY: {args.pathway_name}, semi-inductive split, {args.seeds} seeds")
    print("=" * 72)
    summary = summary_table(results)
    with pd.option_context("display.max_colwidth", 32, "display.width", 200):
        print(summary.to_string())


if __name__ == "__main__":
    main()

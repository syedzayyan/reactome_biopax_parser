"""
test_tgat.py — FISH TGAT benchmark: 6 conditions x N seeds on semi-inductive split.

Conditions
----------
  {TGAT, TGAT + compartment} x {regression, CORN, CORAL}

Usage
-----
    python test_tgat.py \
        --biopax ./data/biopax3/R-HSA-168256.xml \
        --pathway-name Immune \
        --epochs 200 \
        --seeds 5

Outputs
-------
    Per-run results to --out-json (default results_tgat.json) and
    --out-csv (default results_tgat.csv). Summary table printed to stdout.

All metrics are identical to test.py (RGCN runner) — both baselines call the
same evaluation code so their numbers can be compared directly.
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

from fish_tgat import FISHTGAT, build_dataset, evaluate, train


# ── condition definition ────────────────────────────────────────────────────


@dataclass
class Condition:
    name: str
    use_compartment: bool
    order_mode: str

    @property
    def loss_weights(self) -> tuple[float, float, float]:
        if self.order_mode == "coral":
            return (1.0, 1.0, 1.0 / 20)
        return (1.0, 1.0, 1.0)


def _make_conditions() -> list[Condition]:
    conditions = []
    for comp in (False, True):
        for decoder in ("regression", "corn", "coral"):
            name_parts = ["tgat"]
            if comp:
                name_parts.append("+c")
            name_parts.append(f"/{decoder}")
            conditions.append(Condition(
                name=" ".join(name_parts).replace(" /", "/"),
                use_compartment=comp,
                order_mode=decoder,
            ))
    return conditions


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
    seed: int,
    epochs: int,
    hidden: int,
    n_layers: int,
    n_heads: int,
    lr: float,
    n_order_bins: int,
    time_dim: int,
    edge_feat_dim: int,
    max_neighbors: int,
    device: str,
    compartment_embeddings: Optional[dict] = None,
    verbose: bool = False,
    compute_hits: bool = False,
    smart_train: bool = False,
    n_negatives: int = 10,
    time_target: str = "min_max",
    dropout: float = 0.2,
    order_weight: float = 1.0,
) -> dict:
    set_seed(seed)
    data = build_dataset(
        G,
        split="semi_inductive",
        unseen_node_frac=0.20,
        order_mode=condition.order_mode,
        n_order_bins=n_order_bins,
        compartment_embeddings=compartment_embeddings if condition.use_compartment else None,
        time_target=time_target,
        seed=seed,
    )
    model = FISHTGAT(
        data,
        hidden=hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        time_dim=time_dim,
        edge_feat_dim=edge_feat_dim,
        max_neighbors=max_neighbors,
        order_mode=condition.order_mode,
        n_order_bins=n_order_bins,
        n_negatives=n_negatives,
        use_compartment=condition.use_compartment,
        smart_negatives=smart_train,
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
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-order-bins", type=int, default=20)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--edge-feat-dim", type=int, default=32,
                        help="Dimension of the edge-type embedding fed to TemporalAttention.")
    parser.add_argument("--max-neighbors", type=int, default=20,
                        help="Maximum number of (most-recent) temporal neighbours "
                             "per node used during encoding.")
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
                        help="Compute Hits@K (expensive).")
    parser.add_argument("--smart-train", action="store_true")
    parser.add_argument("--n-negatives", type=int, default=10)
    parser.add_argument("--out-json", default="results_tgat.json")
    parser.add_argument("--out-csv", default="results_tgat.csv")
    parser.add_argument("--compartment-embeddings", default=None)
    args = parser.parse_args()

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
          f"max_neighbors={args.max_neighbors}  lr={args.lr}")

    results = []
    for ci, condition in enumerate(CONDITIONS, 1):
        for seed in range(args.seeds):
            run_idx = (ci - 1) * args.seeds + seed + 1
            print(f"\n[{run_idx}/{n_runs}] {condition.name}  seed={seed}")
            m = run_one(
                G, condition, seed,
                epochs=args.epochs, hidden=args.hidden,
                n_layers=args.n_layers, n_heads=args.n_heads,
                lr=args.lr, n_order_bins=args.n_order_bins,
                time_dim=args.time_dim, edge_feat_dim=args.edge_feat_dim,
                max_neighbors=args.max_neighbors,
                device=device,
                compartment_embeddings=compartment_embeddings,
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

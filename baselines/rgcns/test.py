"""
main.py — FISH RGCN benchmark: 6 conditions x 5 seeds on Immune semi-inductive.

Conditions
----------
  {RGCN, RGCN + time encoding} x {regression, CORN, CORAL}

Usage
-----
    python main.py \
        --biopax ./data/biopax3/R-HSA-168256.xml \
        --pathway-name Immune \
        --epochs 200 \
        --seeds 5

Outputs
-------
    Prints a mean +/- std summary table to stdout.
    Writes per-run results to results.json and a tidy DataFrame to results.csv.

The runner is deterministic given --seeds (it sweeps 0..N-1). Each seed
re-samples the unseen-node split and re-initialises model weights; the
parsed graph itself is parsed once and shared across all runs.
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

from fish_rgcn import FISHRGCN, build_dataset, evaluate, train

# ── condition definition ────────────────────────────────────────────────────


@dataclass
class Condition:
    name: str
    architecture: str  # 'rgcn' | 'rgat'
    time_encoding: bool
    use_compartment: bool
    order_mode: str  # 'regression' | 'corn' | 'coral'

    @property
    def loss_weights(self) -> tuple[float, float, float]:
        # CORAL loss is on a much larger scale than CORN/MSE; down-weight so
        # the multi-task sum isn't dominated by ordering for CORAL alone.
        if self.order_mode == "coral":
            return (1.0, 1.0, 1.0 / 20)
        return (1.0, 1.0, 1.0)


def _make_conditions(architectures: tuple[str, ...] = ("rgcn",)) -> list[Condition]:
    """
    Factorial over architecture x time x compartment x decoder.

    Default is RGCN-only (12 conditions). Pass architectures=('rgcn', 'rgat')
    to recover the full 24-condition factorial. RGAT is off by default because:
      (a) its own paper acknowledges no distinctive advantage over RGCN, and
      (b) on Immune it runs 2-3x slower with the patched fix, making 5-pathway
          scaling expensive for marginal expected gain.
    """
    conditions = []
    for arch in architectures:
        for time_enc in (False, True):
            for comp in (False, True):
                for decoder in ("regression", "corn", "coral"):
                    name_parts = [arch]
                    if time_enc:
                        name_parts.append("+t")
                    if comp:
                        name_parts.append("+c")
                    name_parts.append("/")
                    name_parts.append(decoder)
                    conditions.append(
                        Condition(
                            name=" ".join(name_parts).replace(" /", " /"),
                            architecture=arch,
                            time_encoding=time_enc,
                            use_compartment=comp,
                            order_mode=decoder,
                        )
                    )
    return conditions


# Conditions are rebuilt at CLI parse time once we know --include-rgat.
CONDITIONS = _make_conditions()


# ── pathway loading ─────────────────────────────────────────────────────────


def load_and_featurise(
    biopax_path: str, reaction_partners: bool, include_complexes: bool
):
    """
    Parse and featurise once. Returns the prepared networkx graph.

    Importing inside the function so the rest of main.py is usable even
    without the reactome_graphs package installed (e.g. running on cached
    pickled graphs in a future iteration).
    """
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
    """Alternative loader if you've already parsed + featurised once."""
    import pickle

    with open(pickle_path, "rb") as fh:
        return pickle.load(fh)


def load_compartment_embeddings(path: str) -> dict:
    """
    Load compartment embeddings (e.g. GO2Vec) from disk.

    Accepts:
      .npz file with arrays 'keys' (1D str/bytes) and 'vectors' (2D float).
      .pkl pickle file containing a dict {key_str -> np.ndarray}.
      .tsv with one row per term: `key\\tv1\\tv2\\t...\\tvD`.

    Keys should be GO term IDs ('GO:0005829') or compartment common names
    ('cytosol') — the build_dataset code tries both orders.
    """
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


# ── seeding ─────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── run one (condition, seed) ───────────────────────────────────────────────


def run_one(
    G,
    condition: Condition,
    seed: int,
    epochs: int,
    hidden: int,
    n_layers: int,
    lr: float,
    n_order_bins: int,
    time_dim: int,
    device: str,
    compartment_embeddings: Optional[dict] = None,
    verbose: bool = False,
    compute_hits: bool = False,
) -> dict:
    set_seed(seed)
    data = build_dataset(
        G,
        split="semi_inductive",
        unseen_node_frac=0.20,
        order_mode=condition.order_mode,
        n_order_bins=n_order_bins,
        compartment_embeddings=compartment_embeddings,
        seed=seed,
    )
    model = FISHRGCN(
        data,
        hidden=hidden,
        n_layers=n_layers,
        order_mode=condition.order_mode,
        n_order_bins=n_order_bins,
        time_encoding=condition.time_encoding,
        time_dim=time_dim,
        architecture=condition.architecture,
        use_compartment=condition.use_compartment,
    )
    start = time.time()
    train(
        model,
        data,
        epochs=epochs,
        lr=lr,
        loss_weights=condition.loss_weights,
        device=device,
        log_every=epochs if not verbose else max(epochs // 10, 1),
    )
    metrics = evaluate(model, data, device=device, seed=seed, compute_hits=compute_hits)
    metrics["seconds"] = round(time.time() - start, 1)
    metrics["seed"] = seed
    metrics["condition"] = condition.name
    metrics["architecture"] = condition.architecture
    metrics["time_encoding"] = condition.time_encoding
    metrics["use_compartment"] = condition.use_compartment
    metrics["order_mode"] = condition.order_mode
    return metrics


# ── summary table ───────────────────────────────────────────────────────────


def summary_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    metric_cols = [
        "exist_auc",  # existence vs random negatives (semi-inductive bridge)
        "exist_auc_smart",  # existence vs smart negatives  (semi-inductive bridge)
        "hits_at_1",
        "hits_at_10",
        "hits_at_100",  # all-nodes pool
        "hits_at_1_type",
        "hits_at_10_type",
        "hits_at_100_type",  # type-matched
        "type_macro_auc",
        "type_top1",
        "order_spearman",
        "order_pairwise_acc",
        "ind_exist_auc",
        "ind_exist_auc_smart",
        "ind_hits_at_1",
        "ind_hits_at_10",
        "ind_hits_at_100",
        "ind_hits_at_1_type",
        "ind_hits_at_10_type",
        "ind_hits_at_100_type",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    grouped = df.groupby("condition", sort=False)[metric_cols]
    means = grouped.mean()
    stds = grouped.std()
    out = pd.DataFrame(index=means.index)
    for col in metric_cols:
        out[col] = [
            f"{m:+.3f} ± {s:.3f}"
            if "spearman" in col or "rho" in col
            else f"{m:.3f} ± {s:.3f}"
            for m, s in zip(means[col], stds[col])
        ]
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--biopax", help="Path to a .xml/.owl BioPAX Level 3 file")
    src.add_argument("--pickle", help="Path to a pre-parsed and featurised pickle")
    parser.add_argument("--pathway-name", default="Pathway")
    parser.add_argument("--reaction-partners", action="store_true")
    parser.add_argument(
        "--no-complexes",
        action="store_true",
        help="Pass include_complexes=False to the parser.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-order-bins", type=int, default=20)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds; sweeps 0..seeds-1."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device: cuda | cuda:0 | cpu | mps. "
        "Default: auto-detect (cuda > mps > cpu).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Shortcut: force CUDA if available, error if not. "
        "Equivalent to --device cuda.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Shortcut: force CPU (useful for debugging or "
        "when MPS auto-detect is misbehaving).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-epoch losses for each run."
    )
    parser.add_argument(
        "--hits",
        action="store_true",
        help="Also compute Hits@K (K=1, 10, 100) against all-nodes and "
        "type-matched candidate pools. Expensive (~minutes per run on "
        "Immune-sized graphs).",
    )
    parser.add_argument(
        "--include-rgat",
        action="store_true",
        help="Also run RGAT conditions. Default: RGCN only (12 conditions). "
        "With --include-rgat: 24 conditions, ~2.5x runtime.",
    )
    parser.add_argument("--out-json", default="results.json")
    parser.add_argument("--out-csv", default="results.csv")
    parser.add_argument(
        "--compartment-embeddings",
        default=None,
        help="Path to a .npz/.pkl/.tsv of compartment embeddings (e.g. GO2Vec). "
        "When provided, conditions with use_compartment=True use these "
        "vectors. When omitted, a one-hot per compartment vocab is used "
        "instead — equivalent to a learned embedding via the input projection.",
    )
    args = parser.parse_args()

    # Resolve device: explicit --device wins; then --gpu / --cpu shortcuts;
    # otherwise auto-detect (CUDA > MPS > CPU).
    if args.gpu and args.cpu:
        parser.error("Cannot pass --gpu and --cpu together.")
    if args.device is not None:
        device = args.device
    elif args.gpu:
        if not torch.cuda.is_available():
            parser.error("--gpu requested but torch.cuda.is_available() is False.")
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

    # Rebuild conditions list based on whether RGAT was requested.
    global CONDITIONS
    archs = ("rgcn", "rgat") if args.include_rgat else ("rgcn",)
    CONDITIONS = _make_conditions(archs)

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

    # Load compartment embeddings once if a path was given. The same dict
    # is passed to every run_one call; build_dataset uses it only when the
    # condition has use_compartment=True (otherwise the dict is unused).
    compartment_embeddings = None
    if args.compartment_embeddings:
        print(
            f"[main] Loading compartment embeddings from {args.compartment_embeddings}"
        )
        compartment_embeddings = load_compartment_embeddings(
            args.compartment_embeddings
        )
        emb_dim = len(next(iter(compartment_embeddings.values())))
        print(f"[main]   {len(compartment_embeddings)} terms, dim={emb_dim}")
    else:
        print(
            "[main] No compartment-embeddings file given; "
            "compartment conditions will use a one-hot vocabulary "
            "(learned compartment embedding via the input projection)."
        )

    n_runs = len(CONDITIONS) * args.seeds
    print(
        f"[main] Running {len(CONDITIONS)} conditions x {args.seeds} seeds "
        f"= {n_runs} runs on device={device}"
    )
    print(
        f"[main] epochs={args.epochs}  hidden={args.hidden}  "
        f"lr={args.lr}  n_order_bins={args.n_order_bins}"
    )

    results = []
    for ci, condition in enumerate(CONDITIONS, 1):
        for seed in range(args.seeds):
            run_idx = (ci - 1) * args.seeds + seed + 1
            print(f"\n[{run_idx}/{n_runs}] {condition.name}  seed={seed}")
            m = run_one(
                G,
                condition,
                seed,
                epochs=args.epochs,
                hidden=args.hidden,
                n_layers=args.n_layers,
                lr=args.lr,
                n_order_bins=args.n_order_bins,
                time_dim=args.time_dim,
                device=device,
                compartment_embeddings=(
                    compartment_embeddings if condition.use_compartment else None
                ),
                verbose=args.verbose,
                compute_hits=args.hits,
            )
            print(
                f"  exist={m['exist_auc']:.3f}/{m.get('exist_auc_smart', float('nan')):.3f}  "
                f"ind_exist={m.get('ind_exist_auc', float('nan')):.3f}/{m.get('ind_exist_auc_smart', float('nan')):.3f}  "
                f"type_top1={m['type_top1']:.3f}  "
                f"order_rho={m['order_spearman']:+.3f}  "
                f"pair_acc={m['order_pairwise_acc']:.3f}  "
                f"({m['seconds']}s)"
            )
            results.append(m)

    # Persist raw + summary.
    json_safe = []
    for r in results:
        r_clean = {
            k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
            for k, v in r.items()
        }
        json_safe.append(r_clean)
    with open(args.out_json, "w") as fh:
        json.dump(json_safe, fh, indent=2)
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print(f"\n[main] Wrote {args.out_json} and {args.out_csv}")

    # Print mean +/- std table.
    print("\n" + "=" * 72)
    print(f"SUMMARY: {args.pathway_name}, semi-inductive split, {args.seeds} seeds")
    print("=" * 72)
    summary = summary_table(results)
    with pd.option_context("display.max_colwidth", 32, "display.width", 200):
        print(summary.to_string())


if __name__ == "__main__":
    main()

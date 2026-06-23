"""
tune_sthn.py — Optuna hyperparameter sweep for the FISH STHN baseline.

Tunes the headline condition (STHN + compartment / coral) on the Immune
semi-inductive split. Same multi-task score objective as tune.py / tune_tgat.py
so results are comparable across RGCN, TGAT and STHN sweeps.

Outputs
-------
  best_params_sthn.json   — best hyperparameters found.
  optuna_history_sthn.csv — per-trial metrics.

Usage
-----
    python tune_sthn.py \
        --pickle ./cached_graphs/immune.pkl \
        --pathway-name Immune \
        --n-trials 50 \
        --epochs 100 \
        --gpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import torch

from fish_sthn import FISHSTHN, build_dataset, evaluate, train


def multi_task_score(metrics: dict) -> float:
    """
    Multi-task score matching tune.py / tune_tgat.py so sweeps are directly
    comparable.

      exist_term  = (exist_auc - 0.5) * 2
      hits_term   = hits_at_10 * 10
      type_term   = (type_top1 - 0.45) / 0.55
      order_term  = (order_pairwise_acc - 0.5) * 2
    """
    exist_term = (metrics.get("exist_auc", 0.5) - 0.5) * 2
    hits_term = metrics.get("hits_at_10", 0.0) * 10
    type_term = (metrics.get("type_top1", 0.45) - 0.45) / 0.55
    order_term = (metrics.get("order_pairwise_acc", 0.5) - 0.5) * 2
    return float((exist_term + hits_term + type_term + order_term) / 4.0)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(args):
    if args.pickle:
        import pickle
        with open(args.pickle, "rb") as fh:
            return pickle.load(fh)
    from reactome_graphs import NodeFeaturiser, ReactomeBioPAX
    parser = ReactomeBioPAX(uniprot_accession_num=True)
    G = parser.parse_biopax_into_networkx(
        args.biopax, reaction_partners=False, include_complexes=True,
    )
    featuriser = NodeFeaturiser(G, xref_dict={}, parser=parser)
    featuriser.download_and_store()
    featuriser.featurise()
    return G


def make_objective(G, args, device):
    fixed_seed = 0

    def objective(trial: optuna.Trial) -> float:
        hidden = trial.suggest_categorical("hidden", [64, 128, 256, 512])
        n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        time_dim = trial.suggest_categorical("time_dim", [16, 32, 64])
        edge_feat_dim = trial.suggest_categorical("edge_feat_dim", [16, 32, 64])
        # max_edges x window_size combinations are all chosen so that
        # 2 * max_edges is divisible by window_size (required by the mixer).
        max_edges = trial.suggest_categorical("max_edges", [10, 20, 30])
        window_size = trial.suggest_categorical("window_size", [2, 5, 10])
        channel_expansion_factor = trial.suggest_categorical(
            "channel_expansion_factor", [1, 2, 4],
        )
        order_weight = trial.suggest_float("order_weight", 0.1, 20.0, log=True)
        time_target = trial.suggest_categorical(
            "time_target", ["min_max", "log_min_max", "rank"],
        )
        trial_epochs = trial.suggest_categorical("epochs", [50, 100, 200, 400])

        set_seed(fixed_seed)
        data = build_dataset(
            G,
            split="semi_inductive",
            unseen_node_frac=0.20,
            order_mode="coral",
            n_order_bins=20,
            time_target=time_target,
            seed=fixed_seed,
        )
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
            order_mode="coral",
            n_order_bins=20,
            n_negatives=20,
            use_compartment=True,
            smart_negatives=False,
            use_time_encoding=True,
        )
        # CORAL loss scale fix (same as RGCN/TGAT tuners).
        loss_weights = (1.0, 1.0, order_weight / 20.0)

        try:
            train(model, data,
                  epochs=trial_epochs, lr=lr,
                  loss_weights=loss_weights,
                  device=device, log_every=trial_epochs)

            m = evaluate(model, data, device=device, seed=fixed_seed,
                         compute_hits=True)
        except torch.cuda.OutOfMemoryError:
            # The mixer encodes every (src, dst) pair + its 20 negatives as a
            # full sequence through a Transformer in one un-batched call, so
            # large hidden/window_size combos can exceed GPU memory on bigger
            # pathways. Report a score worse than any real trial so TPE learns
            # to steer away from this region instead of treating it as a
            # no-information gap (the default behaviour when an exception
            # just propagates up to optuna's catch=(Exception,)).
            print(f"  -> OOM with hidden={hidden}, n_layers={n_layers}, "
                  f"window_size={window_size}, channel_expansion_factor="
                  f"{channel_expansion_factor}; scoring as failed.", flush=True)
            del model, data
            torch.cuda.empty_cache()
            return -1.0

        score = multi_task_score(m)

        print(
            f"  -> score={score:+.3f}  "
            f"exist_auc={m['exist_auc']:.3f}  "
            f"hits_at_10={m.get('hits_at_10', float('nan')):.3f}  "
            f"type_top1={m['type_top1']:.3f}  "
            f"pair_acc={m['order_pairwise_acc']:.3f}",
            flush=True,
        )

        trial.set_user_attr("exist_auc", m["exist_auc"])
        trial.set_user_attr("hits_at_10", m.get("hits_at_10", float("nan")))
        trial.set_user_attr("type_top1", m["type_top1"])
        trial.set_user_attr("order_pairwise_acc", m["order_pairwise_acc"])
        trial.set_user_attr("mrr", m.get("mrr", float("nan")))

        del model, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--biopax", help="Path to BioPAX Level 3 XML")
    src.add_argument("--pickle", help="Path to pre-featurised graph pickle")
    parser.add_argument("--pathway-name", default="Pathway")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-json", default="best_params_sthn.json")
    parser.add_argument("--out-csv", default="optuna_history_sthn.csv")
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--storage", default="sqlite:///optuna_studies.db",
        help="Optuna storage URL. Pass 'none' for in-memory (trials lost on "
             "interruption).",
    )
    args = parser.parse_args()

    if args.storage and args.storage.lower() not in ("none", ""):
        storage = args.storage
    else:
        storage = None
    if args.study_name is None:
        args.study_name = (
            f"fish_{args.pathway_name}_sthn+c_coral"
            .replace(" ", "_").replace("/", "_")
        )
    print(f"[tune] study_name={args.study_name}")
    print(f"[tune] storage={storage if storage else 'in-memory (trials not persisted)'}")

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
        device = ("cuda" if torch.cuda.is_available()
                  else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[tune] device={device}")

    print(f"[tune] Loading {args.pathway_name}")
    G = load_graph(args)
    print(f"[tune] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[tune] Running {args.n_trials} trials. Epochs tuned per trial "
          f"(choices: 50/100/200/400).")

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    n_already = len(study.trials)
    if n_already > 0:
        print(f"[tune] Resuming existing study with {n_already} prior trials.")
    n_remaining = max(0, args.n_trials - n_already)
    if n_remaining == 0:
        print(f"[tune] Study already has {n_already} trials. Reporting current best.")

    objective = make_objective(G, args, device)
    start = time.time()
    if n_remaining > 0:
        study.optimize(
            objective, n_trials=n_remaining, show_progress_bar=False,
            catch=(Exception,),
        )
    print(f"[tune] Total wall-clock: {(time.time() - start) / 60.0:.1f} min")

    print("\n[tune] Best trial:")
    print(f"  value: {study.best_value:.4f}")
    print("  params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print("\n[tune] Best trial user attrs:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"    {k}: {v}")

    with open(args.out_json, "w") as fh:
        json.dump({
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_user_attrs": dict(study.best_trial.user_attrs),
            "n_trials": args.n_trials,
            "pathway": args.pathway_name,
        }, fh, indent=2)

    rows = []
    for t in study.trials:
        row = {"trial": t.number, "value": t.value, **t.params, **t.user_attrs}
        rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\n[tune] Wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()

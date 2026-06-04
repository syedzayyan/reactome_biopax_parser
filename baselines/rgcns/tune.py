"""
tune.py — Optuna hyperparameter sweep for the FISH RGCN baseline.

Tunes a single condition (default: rgcn + time + compartment / coral, the
strongest in the no-tuning baseline) on Immune semi-inductive against a
multi-task average score. Designed to be cheap: 30-50 trials with a single
seed per trial, 100 epochs each.

Outputs
-------
  best_params.json   — best hyperparameters found, ready to feed back into
                       main.py via --hidden / --lr / --dropout flags.
  optuna_history.csv — per-trial metrics.

Usage
-----
    python tune.py \
        --pickle ./cached_graphs/immune.pkl \
        --pathway-name Immune \
        --n-trials 50 \
        --epochs 100 \
        --gpu

The chosen condition for tuning has all encoder augmentations on (time +
compartment) and CORAL decoder. This is justified empirically: it is the
condition that consistently achieves the best multi-task performance, so
tuning it gives results that transfer to the simpler conditions (which
benefit from the same encoder shape).
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

from fish_rgcn import FISHRGCN, build_dataset, evaluate, train


def multi_task_score(metrics: dict) -> float:
    """
    Multi-task score: average of four rescaled metrics, each in roughly [0, 1]
    where 0 = chance and 1 = perfect.

      exist_term  = (exist_auc - 0.5) * 2                    # AUC, chance 0.5
      hits_term   = hits_at_10 * 10                          # weak metric, amplify
      type_term   = (type_top1 - 0.45) / 0.55                # majority baseline ~0.45
      order_term  = (order_pairwise_acc - 0.5) * 2           # pairwise, chance 0.5

    Includes Hits@10 so the search doesn't ignore the weakest metric. The
    type baseline is approximate; if your majority baseline is much higher
    (e.g. 0.55) the type term may go negative — that is fine, Optuna will
    just penalise it heavily.
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
    """Build the Optuna objective closure."""
    fixed_seed = 0  # one seed per trial; we re-seed for reproducibility.

    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space.
        # n_negatives is FIXED at 20 (not tuned) because TPE naturally
        # prefers lower values that converge faster within a fixed epoch
        # budget, regardless of whether they generalise better. Fixing it
        # removes that confound and lets the other knobs do real work.
        n_negatives = 20

        hidden = trial.suggest_categorical("hidden", [128, 256, 512, 1024])
        n_layers = trial.suggest_categorical("n_layers", [2, 3, 4])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        # Order loss is reweighted on the multi-task sum.
        order_weight = trial.suggest_float("order_weight", 0.1, 20.0, log=True)
        # Time-target transform is part of the search.
        time_target = trial.suggest_categorical(
            "time_target", ["min_max", "log_min_max", "rank"],
        )
        # Epochs as a coarse categorical: too short undertrains, too long
        # overfits. Searching at this granularity rather than 1..400.
        trial_epochs = trial.suggest_categorical(
            "epochs", [50, 100, 200, 400],
        )

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
        model = FISHRGCN(
            data,
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
            order_mode="coral",
            n_order_bins=20,
            n_negatives=n_negatives,
            time_encoding=True,
            time_dim=64,
            architecture="rgcn",
            use_compartment=True,
            smart_negatives=False,  # random negatives default
        )
        loss_weights = (1.0, 1.0, order_weight / 20.0)  # CORAL scale fix

        train(model, data,
              epochs=trial_epochs, lr=lr,
              loss_weights=loss_weights,
              device=device, log_every=trial_epochs)

        # Evaluate with Hits@K so the multi-task score has retrieval info.
        m = evaluate(model, data, device=device, seed=fixed_seed,
                     compute_hits=True)
        score = multi_task_score(m)

        # Print a one-line summary in metric space, so the trace is readable.
        # (The Optuna log line below this prints the maximised score, which
        # is a 0-1 rescaled combination of these metrics. Higher is better.)
        print(
            f"  -> score={score:+.3f}  "
            f"exist_auc={m['exist_auc']:.3f}  "
            f"hits_at_10={m.get('hits_at_10', float('nan')):.3f}  "
            f"type_top1={m['type_top1']:.3f}  "
            f"pair_acc={m['order_pairwise_acc']:.3f}",
            flush=True,
        )

        # Record individual metrics on the trial for inspection.
        trial.set_user_attr("exist_auc", m["exist_auc"])
        trial.set_user_attr("hits_at_10", m.get("hits_at_10", float("nan")))
        trial.set_user_attr("type_top1", m["type_top1"])
        trial.set_user_attr("order_pairwise_acc", m["order_pairwise_acc"])
        trial.set_user_attr("mrr", m.get("mrr", float("nan")))

        # Free trial allocations before the next trial starts. Without this
        # CUDA holds onto cached blocks across trials and large-hidden trials
        # near the start make small-hidden trials later OOM by fragmentation.
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
    parser.add_argument("--out-json", default="best_params.json")
    parser.add_argument("--out-csv", default="optuna_history.csv")
    parser.add_argument(
        "--study-name", default=None,
        help="Optuna study name. Default: 'fish_<pathway-name>_<headline-condition>'. "
             "If a study with this name already exists in --storage, trials "
             "resume from where the previous run left off."
    )
    parser.add_argument(
        "--storage", default="sqlite:///optuna_studies.db",
        help="Optuna storage URL. Default: sqlite:///optuna_studies.db "
             "(persistent SQLite file in the current directory). Pass "
             "'none' or empty string to use in-memory storage (trials lost "
             "on interruption)."
    )
    args = parser.parse_args()

    # Resolve study name and storage.
    if args.storage and args.storage.lower() not in ("none", ""):
        storage = args.storage
    else:
        storage = None
    if args.study_name is None:
        # Default name encodes the pathway and the headline condition so
        # different runs don't accidentally collide in the same SQLite file.
        args.study_name = (
            f"fish_{args.pathway_name}_rgcn+t+c_coral"
            .replace(" ", "_").replace("/", "_")
        )
    print(f"[tune] study_name={args.study_name}")
    print(f"[tune] storage={storage if storage else 'in-memory (trials not persisted)'}")

    # Device.
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

    # Load graph once and reuse across trials.
    print(f"[tune] Loading {args.pathway_name}")
    G = load_graph(args)
    print(f"[tune] Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print(f"[tune] Running {args.n_trials} trials. Epochs is tuned per trial "
          f"(choices: 50/100/200/400).")

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
        print(f"[tune] Study already has {n_already} trials (>= --n-trials "
              f"{args.n_trials}). Nothing to do; reporting current best.")

    objective = make_objective(G, args, device)
    start = time.time()
    if n_remaining > 0:
        # catch=(Exception,) so individual trial failures (e.g. OOM on a
        # large hidden-size config) get recorded as failed trials and the
        # sweep continues, instead of aborting the entire study. Optuna's
        # TPE sampler will avoid sampling near failed configurations on
        # subsequent trials.
        study.optimize(
            objective, n_trials=n_remaining, show_progress_bar=False,
            catch=(Exception,),
        )
    print(f"[tune] Total wall-clock: {(time.time() - start) / 60.0:.1f} min")

    # Persist best params and per-trial history.
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

    # Tidy CSV of trial history.
    rows = []
    for t in study.trials:
        row = {"trial": t.number, "value": t.value, **t.params, **t.user_attrs}
        rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\n[tune] Wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()

"""
tune_hgcn.py — Optuna hyperparameter sweep for the FISH HGCN baseline.

Tunes the headline condition (HGCN + type-hyperedges + compartment / coral)
on the Immune semi-inductive split. Same multi-task score as tune.py and
tune_tgat.py so all three sweeps are directly comparable.

Outputs
-------
  best_params_hgcn.json   — best hyperparameters.
  optuna_history_hgcn.csv — per-trial metrics.

Usage
-----
    python tune_hgcn.py \
        --pickle ./cached_graphs/immune.pkl \
        --pathway-name Immune \
        --n-trials 50 \
        --gpu
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import numpy as np
import optuna
import torch

from fish_hgcn import FISHGCN, build_dataset, evaluate, train


def multi_task_score(metrics: dict) -> float:
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
        n_layers = trial.suggest_categorical("n_layers", [1, 2, 3, 4])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        order_weight = trial.suggest_float("order_weight", 0.1, 20.0, log=True)
        time_target = trial.suggest_categorical(
            "time_target", ["min_max", "log_min_max", "rank"],
        )
        use_attention = trial.suggest_categorical("use_attention", [False, True])
        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4]) if use_attention else 1
        use_type_hyperedges = trial.suggest_categorical("use_type_hyperedges", [False, True])
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
        model = FISHGCN(
            data,
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
            use_attention=use_attention,
            n_heads=n_heads,
            use_type_hyperedges=use_type_hyperedges,
            order_mode="coral",
            n_order_bins=20,
            n_negatives=20,
            use_compartment=True,
            smart_negatives=False,
        )
        loss_weights = (1.0, 1.0, order_weight / 20.0)

        train(model, data, epochs=trial_epochs, lr=lr,
              loss_weights=loss_weights, device=device, log_every=trial_epochs)

        m = evaluate(model, data, device=device, seed=fixed_seed, compute_hits=True)
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
    src.add_argument("--biopax")
    src.add_argument("--pickle")
    parser.add_argument("--pathway-name", default="Pathway")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-json", default="best_params_hgcn.json")
    parser.add_argument("--out-csv", default="optuna_history_hgcn.csv")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default="sqlite:///optuna_studies.db")
    args = parser.parse_args()

    storage = args.storage if args.storage.lower() not in ("none", "") else None
    if args.study_name is None:
        args.study_name = (
            f"fish_{args.pathway_name}_hgcn+the+c_coral"
            .replace(" ", "_").replace("/", "_")
        )
    print(f"[tune] study_name={args.study_name}")
    print(f"[tune] storage={storage or 'in-memory'}")

    if args.gpu and args.cpu:
        parser.error("Cannot pass --gpu and --cpu together.")
    if args.device is not None:
        device = args.device
    elif args.gpu:
        device = "cuda"
    elif args.cpu:
        device = "cpu"
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[tune] device={device}")

    G = load_graph(args)
    print(f"[tune] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[tune] {args.n_trials} trials, epochs tuned per trial.")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        study_name=args.study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    n_already = len(study.trials)
    n_remaining = max(0, args.n_trials - n_already)
    if n_already:
        print(f"[tune] Resuming: {n_already} prior trials.")

    start = time.time()
    if n_remaining > 0:
        study.optimize(
            make_objective(G, args, device),
            n_trials=n_remaining,
            show_progress_bar=False,
            catch=(Exception,),
        )
    print(f"[tune] Wall-clock: {(time.time() - start) / 60:.1f} min")

    print(f"\n[tune] Best: value={study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with open(args.out_json, "w") as fh:
        json.dump({
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_user_attrs": dict(study.best_trial.user_attrs),
            "n_trials": args.n_trials,
            "pathway": args.pathway_name,
        }, fh, indent=2)

    rows = [{"trial": t.number, "value": t.value, **t.params, **t.user_attrs}
            for t in study.trials]
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[tune] Wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()

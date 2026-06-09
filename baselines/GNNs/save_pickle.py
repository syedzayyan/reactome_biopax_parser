"""
save_pickle.py — featurise a BioPAX graph once and save it as a pickle.

Run this first on HPC so tune.py and test.py can use --pickle and skip the
expensive ESM-650M embedding step on every job.

Usage
-----
    python save_pickle.py \
        --biopax /path/to/R-HSA-168256.xml \
        --out    immune.pkl \
        --gpu                     # run ESM on GPU
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biopax", required=True, help="BioPAX Level 3 XML file")
    parser.add_argument("--out", required=True, help="Output pickle path")
    parser.add_argument(
        "--reaction-partners", action="store_true",
        help="Include reaction-partner edges (passed to ReactomeBioPAX).",
    )
    parser.add_argument(
        "--no-complexes", action="store_true",
        help="Exclude complex nodes.",
    )
    parser.add_argument(
        "--esm-device", default=None,
        help="Device for ESM embedding: cuda | cpu | mps. Default: auto.",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    import torch
    if args.gpu and args.cpu:
        parser.error("Cannot pass --gpu and --cpu together.")
    if args.esm_device:
        esm_device = args.esm_device
    elif args.gpu:
        if not torch.cuda.is_available():
            parser.error("--gpu requested but CUDA is not available.")
        esm_device = "cuda"
    elif args.cpu:
        esm_device = "cpu"
    else:
        esm_device = (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    print(f"[save_pickle] ESM device: {esm_device}")
    print(f"[save_pickle] Parsing {args.biopax}")

    from reactome_graphs import NodeFeaturiser, ReactomeBioPAX

    parser_bp = ReactomeBioPAX(uniprot_accession_num=True)
    G = parser_bp.parse_biopax_into_networkx(
        args.biopax,
        reaction_partners=args.reaction_partners,
        include_complexes=not args.no_complexes,
    )
    print(f"[save_pickle] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    featuriser = NodeFeaturiser(
        G,
        xref_dict={},
        parser=parser_bp,
        protein_model_device=esm_device,
    )
    featuriser.download_and_store()
    featuriser.featurise()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save_pickle] Saved → {out_path}")


if __name__ == "__main__":
    main()

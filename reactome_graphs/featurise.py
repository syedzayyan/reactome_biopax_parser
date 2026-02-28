"""
reactome_fetch.py
-----------------
Fetch biological sequences (proteins, DNA, RNA) and SMILES (small molecules)
for nodes in a ReactomeBioPAX NetworkX graph.

Supported APIs
--------------
- UniProt REST  → protein sequences
- Ensembl REST  → DNA sequences  (ENSEMBL: prefix)
- miRBase       → RNA sequences  (miRBase: prefix)  [via scrape, no official REST]
- NCBI EFetch   → DNA/RNA fallback (RefSeq: prefix)
- ChEBI API     → SMILES for small molecules
- Guide to Pharmacology API → SMILES for ligands

Cache
-----
Results are written to ``cache_dir`` as JSON files, one per node, keyed by
node ID (sanitised for filesystem). On subsequent calls the cache is checked
first and the API is skipped.

Usage
-----
    from reactome_fetch import ReactomeFetcher

    fetcher = ReactomeFetcher(G, cache_dir="./reactome_cache")
    results = fetcher.fetch_all()
    # results: dict[node_label -> {"type": ..., "id": ..., "data": ..., "error": ...}]
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import requests

log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────


def _sanitise_key(label: str) -> str:
    """Turn a node label into a safe filename stem."""
    return re.sub(r"[^\w\-]", "_", label)[:180]


def _get(
    url: str, params: dict = None, headers: dict = None, timeout: int = 15
) -> requests.Response:
    """GET with basic retry on 429 / 5xx."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5)) * (attempt + 1)
                log.warning("Rate limited by %s — waiting %ds", url, wait)
                time.sleep(wait)
                continue
            return r
        except requests.RequestException as e:
            log.warning("Request error (attempt %d): %s", attempt + 1, e)
            time.sleep(2**attempt)
    raise requests.RequestException(f"Failed after 3 attempts: {url}")


# ── per-database fetchers ──────────────────────────────────────────────────────


def _fetch_uniprot_sequence(accession: str) -> str:
    """Fetch protein sequence from UniProt REST API."""
    # accession may be "UniProt:P12345" or just "P12345"
    acc = accession.split(":")[-1].strip()
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    r = _get(url)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    return "".join(l for l in lines if not l.startswith(">"))


def _fetch_ensembl_sequence(ensembl_id: str) -> str:
    """Fetch DNA/RNA sequence from Ensembl REST API."""
    eid = ensembl_id.split(":")[-1].strip()
    url = f"https://rest.ensembl.org/sequence/id/{eid}"
    r = _get(url, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()["seq"]


def _fetch_ncbi_sequence(accession: str) -> str:
    """Fetch DNA/RNA sequence from NCBI EFetch (RefSeq accessions)."""
    acc = accession.split(":")[-1].strip()
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "nuccore", "id": acc, "rettype": "fasta", "retmode": "text"}
    r = _get(url, params=params)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    return "".join(l for l in lines if not l.startswith(">"))


def _fetch_mirbase_sequence(mirbase_id: str) -> str:
    """
    Fetch RNA sequence from miRBase.
    miRBase has no official REST API — we use their ftp-backed stem-loop
    sequence endpoint.
    """
    mid = mirbase_id.split(":")[-1].strip()
    # miRBase provides a simple text endpoint for mature/stem-loop sequences
    url = f"https://www.mirbase.org/cgi-bin/get_seq.pl?acc={mid}"
    r = _get(url)
    r.raise_for_status()
    # response is plain FASTA
    lines = r.text.strip().splitlines()
    return "".join(l for l in lines if not l.startswith(">"))


def _fetch_chebi_smiles(chebi_id: str) -> str:
    """Fetch SMILES from ChEBI REST API."""
    cid = chebi_id.split(":")[-1].strip()
    url = f"https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?chebiId={cid}"
    r = _get(url, headers={"Accept": "application/json"})
    r.raise_for_status()
    data = r.json()
    # ChEBI wraps response in "return"
    entity = data.get("return", data)
    smiles = entity.get("smiles") or entity.get("inchiString")
    if not smiles:
        raise ValueError(f"No SMILES found for ChEBI:{cid}")
    return smiles


def _fetch_gtop_smiles(ligand_id: str) -> str:
    """Fetch SMILES from Guide to Pharmacology REST API."""
    lid = ligand_id.split(":")[-1].strip()
    url = f"https://www.guidetopharmacology.org/services/ligands/{lid}"
    r = _get(url, headers={"Accept": "application/json"})
    r.raise_for_status()
    data = r.json()
    # GtP returns a list
    if isinstance(data, list):
        data = data[0]
    smiles = data.get("smiles")
    if not smiles:
        raise ValueError(f"No SMILES found for GtP ligand {lid}")
    return smiles


# ── ID routing ─────────────────────────────────────────────────────────────────


def _route_sequence(ref_id: str) -> str:
    """
    Given a ref_id string like 'UniProt:P12345', 'ENSEMBL:ENSG000...',
    'miRBase:MI0000342', 'RefSeq:NM_001234', dispatch to the right fetcher
    and return the sequence.
    """
    if not ref_id:
        raise ValueError("No ref_id available")

    prefix = ref_id.split(":")[0].strip().lower()

    if prefix == "uniprot":
        return _fetch_uniprot_sequence(ref_id)
    elif prefix in ("ensembl", "ensembl_havana", "ensembl gene"):
        return _fetch_ensembl_sequence(ref_id)
    elif prefix == "mirbase":
        return _fetch_mirbase_sequence(ref_id)
    elif prefix in ("refseq", "ncbi"):
        return _fetch_ncbi_sequence(ref_id)
    else:
        raise ValueError(f"Unknown sequence database prefix: '{prefix}' in '{ref_id}'")


def _route_smiles(entity_ref: str) -> str:
    """
    Given an entity_ref string like 'ChEBI:15422' or
    'Guide to Pharmacology:1234', dispatch to the right fetcher.
    """
    if not entity_ref:
        raise ValueError("No entity_ref available")

    prefix = entity_ref.split(":")[0].strip().lower()

    if prefix == "chebi":
        return _fetch_chebi_smiles(entity_ref)
    elif prefix in ("guide to pharmacology", "gtop", "iuphar"):
        return _fetch_gtop_smiles(entity_ref)
    else:
        raise ValueError(
            f"Unknown small molecule database prefix: '{prefix}' in '{entity_ref}'"
        )


# ══════════════════════════════════════════════════════════════════════════════


class ReactomeFetcher:
    """
    Fetch sequences / SMILES for all nodes in a ReactomeBioPAX NetworkX graph.

    Parameters
    ----------
    G : nx.DiGraph
        Graph from ReactomeBioPAX.parse_biopax_into_networkx.
    cache_dir : str | Path
        Directory for caching results. Created if it doesn't exist.
    delay : float
        Seconds to sleep between API calls to be polite.
    """

    def __init__(
        self,
        G: nx.DiGraph,
        cache_dir: str | Path = "./reactome_cache",
        delay: float = 0.25,
    ):
        self.G = G
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay

    # ── cache helpers ──────────────────────────────────────────────────────────

    def _cache_path(self, node_label: str) -> Path:
        return self.cache_dir / f"{_sanitise_key(node_label)}.json"

    def _load_cache(self, node_label: str) -> Optional[dict]:
        p = self._cache_path(node_label)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def _save_cache(self, node_label: str, record: dict):
        with open(self._cache_path(node_label), "w") as f:
            json.dump(record, f, indent=2)

    # ── per-node dispatch ──────────────────────────────────────────────────────

    def _fetch_node(self, node_label: str, node_data: dict) -> dict:
        """Fetch data for a single node, returning a result record."""
        node_type = node_data.get("type", "other")

        record = {
            "node": node_label,
            "type": node_type,
            "ref_id": None,
            "data": None,  # sequence or SMILES
            "error": None,
        }

        try:
            if node_type in ("protein",):
                ref_id = node_data.get("uniprot_id") or node_data.get("ref_id")
                record["ref_id"] = ref_id
                record["data"] = _route_sequence(ref_id)

            elif node_type in ("dna", "rna"):
                ref_id = node_data.get("ref_id")
                record["ref_id"] = ref_id
                record["data"] = _route_sequence(ref_id)

            elif node_type == "small_molecule":
                entity_ref = node_data.get("entityRef")
                record["ref_id"] = entity_ref
                record["data"] = _route_smiles(entity_ref)

            elif node_type == "physical_entity":
                # physical entities are set-level placeholders —
                # try sequence if they somehow have a ref_id, else skip
                ref_id = node_data.get("ref_id")
                if ref_id:
                    record["ref_id"] = ref_id
                    record["data"] = _route_sequence(ref_id)
                else:
                    record["error"] = "physical_entity with no ref_id — skipped"

            else:
                # complex, or_protein, etc. — nothing to fetch directly
                record["error"] = (
                    f"node type '{node_type}' has no direct fetch strategy"
                )

        except Exception as e:
            record["error"] = str(e)
            log.warning("Failed to fetch '%s': %s", node_label, e)

        return record

    # ── main entry point ───────────────────────────────────────────────────────

    def fetch_all(
        self,
        node_types: Optional[list[str]] = None,
        force: bool = False,
    ) -> dict[str, dict]:
        """
        Fetch data for all (or a subset of) nodes in the graph.

        Parameters
        ----------
        node_types : list[str] | None
            If given, only fetch nodes of these types.
            e.g. ["protein", "dna", "rna", "small_molecule"]
            Default: fetch protein, dna, rna, small_molecule.
        force : bool
            If True, re-fetch even if a cache entry exists.

        Returns
        -------
        dict[node_label -> result_record]
        """
        if node_types is None:
            node_types = ["protein", "dna", "rna", "small_molecule"]

        results: dict[str, dict] = {}
        nodes_to_fetch = [
            (n, d) for n, d in self.G.nodes(data=True) if d.get("type") in node_types
        ]

        log.info(
            "Fetching data for %d nodes (types: %s)", len(nodes_to_fetch), node_types
        )

        for i, (node_label, node_data) in enumerate(nodes_to_fetch):
            # check cache first
            if not force:
                cached = self._load_cache(node_label)
                if cached is not None:
                    results[node_label] = cached
                    continue

            record = self._fetch_node(node_label, node_data)
            self._save_cache(node_label, record)
            results[node_label] = record

            if (i + 1) % 50 == 0:
                log.info("  %d / %d done", i + 1, len(nodes_to_fetch))

            time.sleep(self.delay)

        # summary
        ok = sum(1 for r in results.values() if r["data"] is not None)
        err = sum(1 for r in results.values() if r["error"] is not None)
        log.info("Fetch complete: %d succeeded, %d failed/skipped", ok, err)

        return results

    def summary(self, results: dict[str, dict]) -> None:
        """Print a quick breakdown of fetch results."""
        from collections import Counter

        type_ok: Counter = Counter()
        type_err: Counter = Counter()
        for r in results.values():
            t = r["type"]
            if r["data"] is not None:
                type_ok[t] += 1
            else:
                type_err[t] += 1

        print("\nFetch summary")
        print("─" * 42)
        all_types = sorted(set(type_ok) | set(type_err))
        for t in all_types:
            print(f"  {t:<20} ok={type_ok[t]:<6} err={type_err[t]}")
        print("─" * 42)

        # show first few errors per type
        errors_by_type: dict[str, list] = {}
        for r in results.values():
            if r["error"]:
                errors_by_type.setdefault(r["type"], []).append(
                    (r["node"][:40], r["error"][:80])
                )
        for t, errs in errors_by_type.items():
            print(f"\n  Sample errors for '{t}':")
            for node, err in errs[:3]:
                print(f"    {node}: {err}")

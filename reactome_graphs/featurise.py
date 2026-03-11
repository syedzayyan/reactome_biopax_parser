import json
import re
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import requests


DEFAULT_EDGE_TYPES: list[str] | None = None


class NodeFeaturiser:
    def __init__(
        self,
        G: nx.DiGraph,
        xref_dict: dict,
        cache_dir: str = ".cache/node_featuriser",
        fingerprinter="morgan",
        embed_dim: int = 128,
        protein_model_name: str = "facebook/esm2_t6_8M_UR50D",
        protein_model_path=None,
        protein_model_device: str = "cpu",
        edge_types: list[str] | None = None,
        node_types: list[str] | None = None,
        parser = None
    ):
        self.G = G
        self.xref_dict = xref_dict
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprinter = fingerprinter
        self.embed_dim = embed_dim
        self.protein_model_name = protein_model_name
        self.protein_model_path = protein_model_path
        self.protein_model_device = protein_model_device

        self.edge_types = edge_types or DEFAULT_EDGE_TYPES
        self.node_types = node_types

        self.parser = parser

    # ------------------------------------------------------------------ #
    #  CACHE HELPERS                                                       #
    # ------------------------------------------------------------------ #

    def _build_node_type_vocab(self) -> tuple[list[str], dict[str, int]]:
        """
        Infer node type vocabulary from G.nodes[n]['type'], or use self.node_types
        if pinned. Always appends <unk> as the final slot.
        """
        UNK = "<unk>"

        if self.node_types is not None:
            vocab = list(self.node_types)
        else:
            seen: dict[str, None] = {}
            for n in self.G.nodes:
                t = self.G.nodes[n].get("type", "")
                if t:
                    seen[t] = None
            vocab = list(seen.keys())

        if UNK not in vocab:
            vocab = vocab + [UNK]

        return vocab, {t: i for i, t in enumerate(vocab)}

    def _build_edge_type_vocab(self, edges):
        # Build vocab WITHOUT <unk> — it should never appear as a real label
        seen: dict[str, None] = {}
        for _, _, data in edges:
            t = data.get("type", "")
            if t and t != "<unk>":  # ← explicitly exclude
                seen[t] = None
        vocab = (
            list(self.edge_types) if self.edge_types is not None else list(seen.keys())
        )
        vocab = [v for v in vocab if v != "<unk>"]  # strip if caller passed it in
        self.edge_types = vocab
        return vocab, {t: i for i, t in enumerate(vocab)}

    def _cache_path(self, db: str) -> Path:
        safe_name = re.sub(r"[^\w]", "_", db).lower()
        return self.cache_dir / f"{safe_name}.json"

    def _load_cache(self, db: str) -> dict:
        path = self._cache_path(db)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_cache(self, db: str, data: dict) -> None:
        existing = self._load_cache(db)
        existing.update(data)
        with open(self._cache_path(db), "w") as f:
            json.dump(existing, f)

    def _filter_uncached(self, db: str, ids: list[str]) -> tuple[dict, list[str]]:
        """Returns (cached_results, ids_still_needed)."""
        cache = self._load_cache(db)
        cached = {i: cache[i] for i in ids if i in cache}
        missing = [i for i in ids if i not in cache]
        if cached:
            print(f"[{db}] {len(cached)} loaded from cache, {len(missing)} to fetch")
        return cached, missing

    # ------------------------------------------------------------------ #
    #  1. IDENTIFIER EXTRACTION                                            #
    # ------------------------------------------------------------------ #

    def extract(self) -> dict[str, list[str]]:
        """Returns a dictionary of {database: [identifiers]} from graph nodes."""
        result = defaultdict(list)

        for node in self.G.nodes:
            data = self.G.nodes[node]

            identifier = (
                data.get("uniprot_id") or data.get("ref_id") or data.get("entityRef")
            )

            if not identifier and self.xref_dict and data.get("xref"):
                identifier = self._resolve_xrefs(data["xref"], self.xref_dict)
                self.G.nodes[node]["identifier"] = identifier

            if not identifier or not isinstance(identifier, str):
                continue

            chebi_match = re.search(r"\[ChEBI:\s*(\d+)", identifier)
            if chebi_match:
                identifier = f"ChEBI:{chebi_match.group(1)}"

            if ":" in identifier:
                db, value = identifier.split(":", 1)
                result[db.strip()].append(value.strip())

        return dict(result)

    def _resolve_xrefs(self, xref_list: list, uni_xrefs: dict) -> str | None:
        """Resolve xref dicts to identifiers, preferring stable Reactome IDs."""
        reactome_fallback = None

        for xref in xref_list:
            if not isinstance(xref, dict):
                continue

            db = xref.get("DB_NAME", "")
            db_id = xref.get("DB_ID", "")

            if not db or not db_id:
                continue

            if db == "Reactome":
                return f"Reactome:{db_id}"
            elif db.startswith("Reactome Database ID"):
                reactome_fallback = f"Reactome:{db_id}"

        return reactome_fallback

    # ------------------------------------------------------------------ #
    #  2. PER-DATABASE DOWNLOADERS                                         #
    # ------------------------------------------------------------------ #

    def fetch_uniprot(
        self,
        ids: list[str],
        base_url: str = "https://rest.uniprot.org/uniprotkb/accessions",
    ) -> dict[str, str]:
        """Batch fetch protein sequences from UniProt. Returns {id: sequence}."""
        cached, missing = self._filter_uncached("UniProt", ids)
        if not missing:
            return cached

        results = {}
        for chunk in self._chunks(missing, 500):
            try:
                r = requests.get(
                    base_url,
                    params={"accessions": ",".join(chunk), "format": "fasta"},
                    timeout=30,
                )
                r.raise_for_status()
                current_id, seq_lines = None, []
                for line in r.text.splitlines():
                    if line.startswith(">"):
                        if current_id:
                            results[current_id] = "".join(seq_lines)
                        parts = line.split("|")
                        current_id = parts[1] if len(parts) > 1 else line[1:].split()[0]
                        seq_lines = []
                    else:
                        seq_lines.append(line)
                if current_id:
                    results[current_id] = "".join(seq_lines)
            except Exception as e:
                print(f"[UniProt] Batch failed: {e}")

        self._save_cache("UniProt", results)
        return {**cached, **results}

    def fetch_chebi(
        self,
        ids: list[str],
        base_url: str = "https://www.ebi.ac.uk/chebi/backend/api/public/compound/{id}/",
    ) -> dict[str, str]:
        """Fetch SMILES from ChEBI REST API. Returns {id: smiles}."""
        cached, missing = self._filter_uncached("ChEBI", ids)
        if not missing:
            return cached

        results = {}
        for cid in missing:
            try:
                r = requests.get(
                    base_url.format(id=cid),
                    params={
                        "only_ontology_parents": "false",
                        "only_ontology_children": "false",
                    },
                    headers={"accept": "*/*"},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                structure = data.get(
                    "default_structure"
                )  # can be None for minerals etc.
                results[cid] = structure.get("smiles") if structure else None
            except Exception as e:
                print(f"[ChEBI] Failed {cid}: {e}")
                results[cid] = None
            time.sleep(0.1)

        self._save_cache("ChEBI", results)
        return {**cached, **results}

    def fetch_pubchem(
        self,
        ids: list[str],
        base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{ids}/JSON",
    ) -> dict[str, str]:
        """Batch fetch SMILES from PubChem Compound. Returns {id: smiles}."""
        cached, missing = self._filter_uncached("PubChem Substance", ids)
        if not missing:
            return cached

        results = {}
        for chunk in self._chunks(missing, 200):
            url = base_url.format(ids=",".join(chunk))
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                for compound in r.json().get("PC_Compounds", []):
                    cid = str(compound.get("id", {}).get("id", {}).get("cid", ""))
                    smiles = next(
                        (
                            p["value"]["sval"]
                            for p in compound.get("props", [])
                            if p.get("urn", {}).get("label") == "SMILES"
                            and p.get("urn", {}).get("name") == "Absolute"
                        ),
                        None,
                    )
                    if cid:
                        results[cid] = smiles
            except Exception as e:
                print(f"[PubChem] Batch failed: {e}")

        self._save_cache("PubChem Substance", results)
        return {**cached, **results}

    def fetch_ensembl(
        self,
        ids: list[str],
        base_url: str = "https://rest.ensembl.org/sequence/id",
    ) -> dict[str, str]:
        """Batch fetch sequences from Ensembl via POST. Returns {id: sequence}."""
        cached, missing = self._filter_uncached("ENSEMBL", ids)
        if not missing:
            return cached

        results = {}
        for chunk in self._chunks(missing, 50):
            try:
                r = requests.post(
                    base_url,
                    headers={"Content-Type": "application/json"},
                    json={"ids": chunk},
                    timeout=30,
                )
                r.raise_for_status()
                for entry in r.json():
                    eid = entry.get("id")
                    seq = entry.get("seq")
                    if eid:
                        results[eid] = seq
            except Exception as e:
                print(f"[Ensembl] Batch failed: {e}")

        self._save_cache("ENSEMBL", results)
        return {**cached, **results}

    def fetch_ncbi_nucleotide(
        self,
        ids: list[str],
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
    ) -> dict[str, str]:
        """Fetch nucleotide sequences from NCBI. Returns {id: sequence}."""
        cached, missing = self._filter_uncached("NCBI Nucleotide", ids)
        if not missing:
            return cached

        results = {}
        for chunk in self._chunks(missing, 20):
            try:
                r = requests.get(
                    base_url,
                    params={
                        "db": "nucleotide",
                        "id": ",".join(chunk),
                        "rettype": "fasta",
                        "retmode": "text",
                    },
                    timeout=30,
                )
                r.raise_for_status()
                current_id, seq_lines = None, []
                for line in r.text.splitlines():
                    if line.startswith(">"):
                        if current_id:
                            results[current_id] = "".join(seq_lines)
                        current_id = line[1:].split()[0]
                        seq_lines = []
                    else:
                        seq_lines.append(line.strip())
                if current_id:
                    results[current_id] = "".join(seq_lines)
            except Exception as e:
                print(f"[NCBI] Batch failed: {e}")
            time.sleep(0.34)  # max 3 req/sec without API key

        self._save_cache("NCBI Nucleotide", results)
        return {**cached, **results}

    def fetch_guide_to_pharmacology(
        self,
        ids: list[str],
        base_url: str = "https://www.guidetopharmacology.org/services/ligands/{id}",
    ) -> dict[str, str]:
        """Fetch SMILES from Guide to Pharmacology ligands. Returns {id: smiles}."""
        cached, missing = self._filter_uncached("Guide to Pharmacology", ids)
        if not missing:
            return cached

        results = {}
        for gid in missing:
            try:
                r = requests.get(base_url.format(id=gid), timeout=10)
                r.raise_for_status()
                data = r.json()
                entry = data[0] if isinstance(data, list) else data
                smiles = entry.get("smiles") or entry.get("structuralData", {}).get(
                    "smiles"
                )
                results[gid] = smiles
            except Exception as e:
                print(f"[GtP] Failed {gid}: {e}")
                results[gid] = None
            time.sleep(0.1)

        self._save_cache("Guide to Pharmacology", results)
        return {**cached, **results}

    # ------------------------------------------------------------------ #
    #  3. DOWNLOAD ALL + STORE ON GRAPH                                    #
    # ------------------------------------------------------------------ #

    def download_and_store(self) -> None:
        """
        Extracts identifiers, downloads data for each database,
        and stores raw sequences/SMILES as node attributes.
        """
        id_map = self.extract()

        fetchers = {
            "UniProt": (self.fetch_uniprot, "sequence"),
            "ChEBI": (self.fetch_chebi, "smiles"),
            "PubChem Substance": (self.fetch_pubchem, "smiles"),
            "ENSEMBL": (self.fetch_ensembl, "sequence"),
            "NCBI Nucleotide": (self.fetch_ncbi_nucleotide, "sequence"),
            "Guide to Pharmacology": (self.fetch_guide_to_pharmacology, "smiles"),
        }

        db_results = {}
        for db, (fetcher, _) in fetchers.items():
            if db not in id_map:
                continue
            unique_ids = list(set(id_map[db]))
            print(f"[{db}] {len(unique_ids)} unique IDs...")
            db_results[db] = fetcher(unique_ids)

        for node in self.G.nodes:
            data = self.G.nodes[node]

            identifier = (
                data.get("uniprot_id") or data.get("ref_id") or data.get("entityRef")
            )
            if not identifier and self.xref_dict and data.get("xref"):
                identifier = self._resolve_xrefs(data["xref"], self.xref_dict)
            if not identifier:
                continue

            chebi_match = re.search(r"\[ChEBI:\s*(\d+)", identifier)
            if chebi_match:
                identifier = f"ChEBI:{chebi_match.group(1)}"

            if ":" not in identifier:
                continue

            db, value = identifier.split(":", 1)
            db, value = db.strip(), value.strip()

            if db in db_results and value in db_results[db]:
                _, attr_name = fetchers[db]
                self.G.nodes[node][attr_name] = db_results[db][value]

    # ------------------------------------------------------------------ #
    #  4. HELPERS                                                          #
    # ------------------------------------------------------------------ #

    def _chunks(self, lst: list, size: int):
        """Yield successive chunks of a list."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    # ------------------------------------------------------------------ #
    #  5. FEATURISATION                                                    #
    # ------------------------------------------------------------------ #

    def featurise(
        self,
        add_go_embedding: bool = False,
        go_embedding_path: str = "embeddings.pt",
    ) -> None:
        """
        Converts raw sequence/SMILES node attributes to natural-dim feature vectors.
        Each modality outputs its native dimensionality (ESM-2: model hidden size,
        Morgan: 2048, k-mer: 4^k). Features are stored in G.nodes[node]["feature"].

        Requires (optional, imported lazily):
            - rdkit: for Morgan fingerprints (SMILES)
            - torch + transformers: for ESM-2 protein embeddings
        """
        import numpy as np

        G = self.G

        protein_seqs, smiles_nodes, dna_nodes = {}, {}, {}

        for node in G.nodes:
            data = G.nodes[node]
            if data.get("sequence"):
                node_type = self._infer_sequence_type(data["sequence"])
                if node_type == "protein":
                    protein_seqs[node] = data["sequence"]
                else:
                    dna_nodes[node] = data["sequence"]
            elif data.get("smiles"):
                smiles_nodes[node] = data["smiles"]

        print(
            f"[Featurise] {len(protein_seqs)} proteins, {len(smiles_nodes)} molecules, {len(dna_nodes)} DNA/RNA"
        )

        # -- Load feature cache ------------------------------------------------
        feat_cache_path = self.cache_dir / "features.npz"
        feat_cache = {}
        if feat_cache_path.exists():
            loaded = np.load(feat_cache_path, allow_pickle=False)
            feat_cache = {k: loaded[k] for k in loaded.files}
            print(f"[Featurise] {len(feat_cache)} features loaded from cache")

        def _apply_cached(node_dict: dict) -> tuple[dict, dict]:
            hits, misses = {}, {}
            for node, val in node_dict.items():
                if node in feat_cache:
                    hits[node] = feat_cache[node]
                else:
                    misses[node] = val
            return hits, misses

        # -- Proteins: ESM-2 ---------------------------------------------------
        if protein_seqs:
            cached_hits, uncached = _apply_cached(protein_seqs)
            for node, feat in cached_hits.items():
                G.nodes[node]["feature"] = feat
            if uncached:
                esm_feats = self._embed_esm(uncached)
                for node, feat in esm_feats.items():
                    G.nodes[node]["feature"] = feat
                    feat_cache[node] = feat

        # -- SMILES: Morgan fingerprint ----------------------------------------
        if smiles_nodes:
            cached_hits, uncached = _apply_cached(smiles_nodes)
            for node, feat in cached_hits.items():
                G.nodes[node]["feature"] = feat
            if uncached:
                morgan_feats = self._embed_morgan(uncached)
                for node, feat in morgan_feats.items():
                    G.nodes[node]["feature"] = feat
                    feat_cache[node] = feat

        # -- DNA/RNA: k-mer frequencies ----------------------------------------
        if dna_nodes:
            cached_hits, uncached = _apply_cached(dna_nodes)
            for node, feat in cached_hits.items():
                G.nodes[node]["feature"] = feat
            if uncached:
                kmer_feats = self._embed_kmer(uncached)
                for node, feat in kmer_feats.items():
                    G.nodes[node]["feature"] = feat
                    feat_cache[node] = feat


        
        # -- Save updated cache ------------------------------------------------
        if feat_cache:
            np.savez(feat_cache_path, **feat_cache)
            print(
                f"[Featurise] Cache saved ({len(feat_cache)} entries) → {feat_cache_path}"
            )

        # -- Remaining unfeaturised: zero vector (dim inferred from peers) -----
        feat_dim = next(
            (
                G.nodes[n]["feature"].shape[0]
                for n in G.nodes
                if G.nodes[n].get("feature") is not None
            ),
            self.embed_dim,
        )
        zero_count = 0
        for node in G.nodes:
            if G.nodes[node].get("feature") is None:
                G.nodes[node]["feature"] = np.zeros(feat_dim, dtype=np.float32)
                zero_count += 1
        if zero_count:
            print(
                f"[Featurise] {zero_count} nodes have zero vectors (no data available)"
            )

        # -- Complexes: mean pool members --------------------------------------
        self._featurise_complexes(G)
        
        if add_go_embedding:
            self._append_go_embeddings(G, go_embedding_path)

    def _append_go_embeddings(self, G, path: str) -> None:
        """
        Load GO Node2Vec embeddings and concatenate onto existing node features.
        Sources GO terms from:
          1. G.nodes[node]["go_terms"]  — protein-level annotations (list of GO IDs)
          2. G.nodes[node]["cellularLocation"]["DB_ID"]  — compartment from BioPAX parse

        For nodes with no matching GO terms, a zero vector is appended to keep
        feature dimensionality consistent across all nodes.

        Expects embeddings.pt saved as either:
          - {go_id: tensor}  (per-term dict)
          - {"embedding_matrix": tensor, "node_to_idx": dict, "idx_to_node": dict}
        """
        import numpy as np
        import torch
        from pathlib import Path

        if not Path(path).exists():
            print(f"[GO] Embedding file not found: {path} — skipping")
            return

        raw = torch.load(path, map_location="cpu")

        if isinstance(raw, dict) and "embedding_matrix" in raw:
            node_to_idx = raw["node_to_idx"]
            matrix = raw["embedding_matrix"]
            go_embeddings = {
                go_id: matrix[idx].detach().numpy().astype(np.float32)
                for go_id, idx in node_to_idx.items()
            }
        else:
            go_embeddings = {
                go_id: vec.detach().numpy().astype(np.float32)
                for go_id, vec in raw.items()
            }

        if not go_embeddings:
            print("[GO] No embeddings loaded — skipping")
            return

        go_dim = next(iter(go_embeddings.values())).shape[0]
        zero_go = np.zeros(go_dim, dtype=np.float32)

        enriched, missing = 0, 0

        for node in G.nodes:
            existing = G.nodes[node].get("feature")
            if existing is None:
                continue

            # Collect GO IDs from both sources
            go_terms = list(G.nodes[node].get("go_terms", []))

            cell_loc = G.nodes[node].get("cellularLocation")
            if cell_loc and isinstance(cell_loc, dict):
                db_id = cell_loc.get("DB_ID", "")
                if db_id.startswith("GO:"):
                    go_terms.append(db_id)

            # Mean pool over all terms that exist in the embedding vocab
            term_vecs = [go_embeddings[t] for t in go_terms if t in go_embeddings]

            if term_vecs:
                go_vec = np.mean(term_vecs, axis=0).astype(np.float32)
                enriched += 1
            else:
                go_vec = zero_go
                missing += 1

            G.nodes[node]["feature"] = np.concatenate([existing, go_vec])

        print(
            f"[GO] Appended {go_dim}-dim GO embeddings: "
            f"{enriched} nodes enriched, {missing} nodes got zero vector"
        )


    def _infer_sequence_type(self, sequence: str) -> str:
        """Heuristic: if >80% ACGTU characters, treat as DNA/RNA."""
        seq = sequence.upper()
        nucleotide_chars = set("ACGTU")
        ratio = sum(c in nucleotide_chars for c in seq) / max(len(seq), 1)
        return "dna" if ratio > 0.8 else "protein"

    def _embed_esm(self, protein_seqs: dict) -> dict:
        """ESM-2 mean-pooled embeddings. Output dim = model hidden size."""
        import numpy as np

        try:
            import torch
            from transformers import AutoTokenizer, EsmModel
        except ImportError:
            print(
                "[ESM] transformers not installed. pip install transformers. Falling back to one-hot."
            )
            return self._embed_onehot_protein(protein_seqs)

        model_name = self.protein_model_path or self.protein_model_name
        print(f"[ESM] Loading ESM-2 model ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        model.eval()
        device = self.protein_model_device
        model = model.to(device)

        results = {}
        items = list(protein_seqs.items())

        for i in range(0, len(items), 32):
            batch = items[i : i + 32]
            nodes, seqs = zip(*batch)
            seqs_truncated = [s[:1022] for s in seqs]

            tokens = tokenizer(
                list(seqs_truncated),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=True,
            ).to(device)

            with torch.no_grad():
                out = model(**tokens)

            hidden = out.last_hidden_state  # [B, L, D]
            attention_mask = tokens["attention_mask"]  # [B, L]

            for j, node in enumerate(nodes):
                seq_len = attention_mask[j].sum().item()
                feat = hidden[j, 1 : seq_len - 1, :]  # exclude BOS + EOS
                results[node] = feat.mean(dim=0).cpu().numpy().astype(np.float32)

        return results

    def _embed_morgan(self, smiles_nodes: dict, nBits: int = 2048) -> dict:
        """Morgan fingerprints. Output dim = nBits (default 2048)."""
        import numpy as np

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            print("[RDKit] rdkit not installed. pip install rdkit. Using zero vectors.")
            return {node: np.zeros(nBits, dtype=np.float32) for node in smiles_nodes}

        def _mol_from_smiles(smi: str):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return mol
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                return None
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
                Chem.SetAromaticity(mol)
            except Exception:
                pass
            return mol

        results = {}
        for node, smi in smiles_nodes.items():
            if not smi:
                results[node] = np.zeros(nBits, dtype=np.float32)
                continue
            try:
                mol = _mol_from_smiles(smi)
                if mol is None:
                    raise ValueError(f"Could not parse SMILES: {smi}")
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=nBits, useChirality=True
                )
                results[node] = np.array(fp, dtype=np.float32)
            except Exception as e:
                print(f"[Morgan] Failed {node}: {e}")
                results[node] = np.zeros(nBits, dtype=np.float32)

        return results

    def _embed_kmer(self, dna_nodes: dict, k: int = 4) -> dict:
        """K-mer frequency vectors. Output dim = 4^k (256 for k=4)."""
        from itertools import product

        import numpy as np

        alphabet = "ACGT"
        kmers = ["".join(p) for p in product(alphabet, repeat=k)]
        kmer_index = {km: i for i, km in enumerate(kmers)}
        kmer_dim = len(kmers)

        results = {}
        for node, seq in dna_nodes.items():
            seq = seq.upper().replace("U", "T")
            vec = np.zeros(kmer_dim, dtype=np.float32)
            total = 0
            for i in range(len(seq) - k + 1):
                km = seq[i : i + k]
                if km in kmer_index:
                    vec[kmer_index[km]] += 1
                    total += 1
            if total > 0:
                vec /= total
            results[node] = vec

        return results

    def _embed_onehot_protein(self, protein_seqs: dict) -> dict:
        """Fallback: mean one-hot over amino acid alphabet. Output dim = 20."""
        import numpy as np

        AA = "ACDEFGHIKLMNPQRSTVWY"
        aa_index = {a: i for i, a in enumerate(AA)}
        results = {}
        for node, seq in protein_seqs.items():
            vec = np.zeros(len(AA), dtype=np.float32)
            for aa in seq:
                if aa in aa_index:
                    vec[aa_index[aa]] += 1
            if len(seq) > 0:
                vec /= len(seq)
            results[node] = vec
        return results

    def _featurise_complexes(self, G):
        import numpy as np
        
        # Get global featdim from existing leaves
        leaf_feats = [G.nodes[n]['feature'] for n in G.nodes 
                      if G.nodes[n].get('feature') is not None and 
                         G.nodes[n].get('type') not in ('complex', 'Complex')]
        if not leaf_feats:
            print("No leaf features found - skipping complexes")
            return
        featdim = leaf_feats[0].shape[0]
        
        # idtonode map (same as before)
        idtonode = {}
        for store_name in ['proteins', 'molecules', 'dna', 'rna', 'physical_entity', 'complexes']:
            store = getattr(self.parser, store_name, {})
            for entity_id, data in store.items():
                label = self.parser._make_label_for_id(entity_id)
                if label:
                    idtonode[entity_id] = label
        for node in G.nodes:
            data = G.nodes[node]
            for attr in ['reactomedbid', 'refid', 'uniprotid', 'entityRef']:
                val = data.get(attr)
                if val:
                    idtonode[str(val)] = node
        
        def flatten_and_pool(node):
            """Flatten to leaves, resize, mean pool"""
            # Find raw_id for this label
            raw_id = None
            for store in [getattr(self.parser, 'complexes', {})]:
                for eid, data in store.items():
                    if self.parser._make_label_for_id(eid) == node:
                        raw_id = eid
                        break
                if raw_id: break
            
            if not raw_id:
                G.nodes[node]['feature'] = np.zeros(featdim, dtype=np.float32)
                return
            
            def get_leaf_feats(eid, visited=None):
                if visited is None: visited = set()
                if eid in visited: return []
                visited.add(eid)
                
                label = idtonode.get(eid)
                if not label or label not in G:
                    return []
                
                data = G.nodes[label]
                if data.get('type') not in ('complex', 'Complex'):
                    feat = data.get('feature')
                    if feat is not None and feat.shape == (featdim,):
                        return [feat]
                    return []
                
                # Recurse
                feats = []
                if eid in self.parser.complexes:
                    for comp_id in self.parser.complexes[eid].get('components', []):
                        feats.extend(get_leaf_feats(comp_id, visited))
                return feats
            
            all_feats = get_leaf_feats(raw_id)
            if all_feats:
                result = np.mean(all_feats, axis=0).astype(np.float32)
            else:
                result = np.zeros(featdim, dtype=np.float32)
            
            G.nodes[node]['feature'] = result
        
        # Process all complexes
        complexes = [n for n in G.nodes if G.nodes[n].get('type') in ('complex', 'Complex')]
        for c in complexes:
            flatten_and_pool(c)
        
        print(f"Featurised {len(complexes)} complexes")


    def _resize(self, vec, target_dim: int):
        """Resize vector to target_dim by truncating or zero-padding."""
        import numpy as np

        if len(vec) == target_dim:
            return vec
        elif len(vec) > target_dim:
            return vec[:target_dim]
        else:
            return np.pad(vec, (0, target_dim - len(vec)))

    def to_dgbatch(self, edge_types: list[str] | None = None) -> "DGData":
        import numpy as np
        import torch
        from tgm.data import DGData

        G = self.G

        # -- Node ID mapping ---------------------------------------------------
        node_list = list(G.nodes)
        node_to_id = {n: i for i, n in enumerate(node_list)}

        # -- Static node features ----------------------------------------------
        feat_dim = None
        static_feats = []
        for node in node_list:
            feat = G.nodes[node].get("feature")
            if feat is not None:
                static_feats.append(feat)
                feat_dim = len(feat)
            else:
                static_feats.append(None)

        if feat_dim is None:
            raise ValueError(
                "[DGBatch] No node features found — call featurise() first."
            )

        static_feats = [
            f if f is not None else np.zeros(self.embed_dim, dtype=np.float32)
            for f in static_feats
        ]
        target_dim = max(f.shape[0] for f in static_feats)
        padded = [np.pad(f, (0, target_dim - f.shape[0])) for f in static_feats]
        static_node_x = torch.tensor(np.stack(padded), dtype=torch.float32)
        # -- Edges -------------------------------------------------------------
        edges = list(G.edges(data=True))
        if not edges:
            raise ValueError(
                "[DGBatch] Graph has no edges — call download_and_store() first."
            )

        # ✅ Dynamic edge type vocab

        if edge_types is not None:
            self.edge_types = edge_types  # pin caller-supplied vocab before building
        self.edge_types, type_to_idx = self._build_edge_type_vocab(edges)

        src_ids, dst_ids, edge_times, edge_type_ids, edge_feats = [], [], [], [], []

        for src, dst, data in edges:
            src_ids.append(node_to_id[src])
            dst_ids.append(node_to_id[dst])
            t = data.get("time", data.get("timestamp", data.get("t", 0)))
            edge_times.append(float(t))
            raw_type = data.get("type", "")
            unk_idx = -1
            edge_type_ids.append(type_to_idx.get(raw_type, unk_idx))
            ef = data.get("feature", data.get("edge_feature"))
            edge_feats.append(ef)

        order = np.argsort(edge_times, kind="stable")
        edge_times = np.array(edge_times, dtype=np.float32)[order]
        src_ids = np.array(src_ids)[order]
        dst_ids = np.array(dst_ids)[order]
        edge_type_ids = np.array(edge_type_ids)[order]

        edge_index = torch.tensor(
            np.stack([src_ids, dst_ids], axis=1), dtype=torch.long
        )
        edge_time = torch.tensor(edge_times, dtype=torch.int32)

        # ✅ edge_type as a dedicated LongTensor — no one-hot needed
        edge_type = torch.tensor(edge_type_ids, dtype=torch.long)

        # ✅ edge_x carries only continuous features, or None
        has_edge_feats = any(ef is not None for ef in edge_feats)
        if has_edge_feats:
            edge_feat_dim = next(len(ef) for ef in edge_feats if ef is not None)
            edge_x = torch.tensor(
                np.stack(
                    [
                        edge_feats[i]
                        if edge_feats[i] is not None
                        else np.zeros(edge_feat_dim)
                        for i in order
                    ]
                ),
                dtype=torch.float32,
            )
        else:
            edge_x = None

        # ✅ node_type: infer from graph node attribute, fallback to 0
        self.node_types, node_type_to_idx = self._build_node_type_vocab()

        node_type = torch.tensor(
            [
                node_type_to_idx.get(
                    G.nodes[n].get("type", ""), node_type_to_idx["<unk>"]
                )
                for n in node_list
            ],
            dtype=torch.long,
        )

        print(
            f"[DGBatch] {len(node_list)} nodes (feat_dim={static_node_x.shape[1]}), "
            f"{len(edges)} edges (edge_dim={edge_x.shape[1] if edge_x is not None else 0}), "
            f"t=[{edge_times.min():.2f}, {edge_times.max():.2f}]"
        )

        return DGData.from_raw(
            time_delta="r",
            edge_time=edge_time,
            edge_index=edge_index,
            edge_x=edge_x,
            edge_type=edge_type,
            node_type=node_type,
            static_node_x=static_node_x,
            node_y_time=None,
            node_y_nids=None,
            node_y=None,
            node_x_nids=None,
            node_x=None,
            node_x_time=None,
        )

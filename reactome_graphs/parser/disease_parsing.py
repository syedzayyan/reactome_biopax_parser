"""
disease_parsing.py
==================
DiseaseOverlay — overlay Reactome disease BioPAX reactions onto a healthy
pathway graph built by ReactomeBioPAX / _NxGraphMixin.

Design notes
------------
1. Reuses _ParserBase.parse_biopax3_file() for all XML work.
   No parallel XML parser.  A fresh ReactomeBioPAX instance is spun up for
   the disease file and all entity stores come for free.

2. Variant entity detection operates on both the catalyst (GoF / aberrant
   reactions) AND on left-side participants (LoF reactions, which is the
   majority).  An entity is considered a variant if:
     a. It is an OR-group (has bp:memberPhysicalEntity) whose displayName
        contains "variant" / "mutant" / "defect*"  — e.g. "IRAK4 variants"
     b. It is any entity (protein, complex) carrying a bp:feature that
        resolves to a ModificationFeature in the disease file — these are
        disease-causing amino-acid changes (not just PTMs)
     c. Its displayName matches HGVS-like protein-change notation such as
        "IRAK4 E402*", "MyD88 R196C", "TICAM1 R141*"

3. Cartesian product across mutant variants × reaction branches.
   Each [MUT] variant node gets its OWN copy of the reaction sub-graph:

       [MUT_i]  ──catalysis──►  bp_node  ──reaction──►  r_lbl
       [MUT_i]  ──────────────────────────────────────►  [DISEASE]
       r_lbl    ──disease_progression──────────────────► [DISEASE]

   For LoF reactions the cartesian product is across variants × branch nodes:

       bp_node  ──disease_branch──►  [MUT_i]  ──loss_of_function──►  [DISEASE]

4. Branch node resolution:
   a. First pass: match left-side non-variant entities directly against
      healthy graph nodes (reactome_db_id → exact name → fuzzy name).
   b. Second pass (fallback): decompose left-side complexes to their leaf
      protein/molecule components and match those.  This recovers cases
      where the disease file uses a differently-named complex than the
      healthy pathway (e.g. "oligo-MyD88:TIRAP:BTK:activated TLR" → leaves
      match healthy nodes "MyD88 [cytosol]", "MAL [cytosol]", …).

5. Time assignment:
   Disease reactions are sorted by their position in the disease file's own
   PathwayStep chain (dp.step_order) so their time values preserve the
   causal ordering described in the disease pathway, rather than arbitrary
   enumeration order.

6. MONDO / OMIM extraction:
   - OMIM IDs: scanned from bp:comment text ("MIM:XXXXXX") and from
     bp:xref elements with db in {OMIM, MIM}.
   - MONDO IDs: extracted from bp:xref elements with db in {MONDO,
     "Disease Ontology", ORPHA, MedGen} and from bp:disease attributes
     pointing to DiseaseOntologyVocabulary or similar elements.
   - ClinGen: allele IDs stored as annotations on MutantVariant nodes.
   - When no OMIM/MONDO is found the disease label falls back to the
     pathway displayName.

7. Entity label resolution and OR-leaf expansion delegate to the
   _NxGraphMixin helpers already present on the healthy parser:
   _make_label_for_id, _resolve_or_leaves, _make_label.
   The disease parser (a fresh ReactomeBioPAX) has the same methods.
"""

from __future__ import annotations

import logging
import re
import typing
from dataclasses import dataclass, field

import networkx as nx

# ---------------------------------------------------------------------------
# Namespace / ID helpers (mirror _ParserBase)
# ---------------------------------------------------------------------------
_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "biopax": "http://www.biopax.org/release/biopax-level3.owl#",
}
_ID = "{%s}ID" % _NS["rdf"]
_RES = "{%s}resource" % _NS["rdf"]

# Matches HGVS-like protein variant notation in an entity display name.
# Positive: "E402*", "Q293X", "R12C", "E52del", "L230Afs*188", "P554S",
#           "R141*", "S186L", "L93P", "S34Y", "E65del"
# The pattern requires a capital letter + digits + change notation, preceded
# by a word boundary or space so it doesn't match e.g. "p-4Y-MAL" (phospho).
_VARIANT_NAME_RE = re.compile(
    r"(?:^|[\s_-])[A-Z]\d+(?:\*|[A-Z]|del|ins|dup|fs|Ter|TER|X\b)",
    re.IGNORECASE,
)

# Disease xref DB names that carry disease ontology IDs.
_DISEASE_DBS = frozenset(
    {
        "omim",
        "mim",
        "mondo",
        "disease ontology",
        "do",
        "orpha",
        "orphanet",
        "medgen",
        "clinvar",
        "omim disease",
    }
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MutantVariant:
    """A single disease-causing entity resolved from a variant left-side entity
    or an OR-group catalyst."""

    entity_id: str
    display_name: str
    reactome_db_id: str | None
    uniprot_id: str | None
    omim_ids: list[str]
    mutations: list[str]
    clingen_ids: list[str] = field(default_factory=list)


@dataclass
class DiseaseReaction:
    """
    One parsed disease reaction, ready for graph overlay.

    Topology produced by overlay
    ----------------------------
    GoF / aberrant (right_ids non-empty):
        For each mutant variant i:
            bp_node ──[reaction, subtype=disease]──► r_lbl
            [MUT_i] ──[catalysis, subtype=disease_aberrant]──► bp_node
            [MUT_i] ──[aberrant_product]──► [DISEASE]
        r_lbl ──[disease_progression]──► [DISEASE]

    LoF (right_ids empty):
        For each mutant variant i, for each branch node:
            bp_node ──[disease_branch]──► [MUT_i]
            [MUT_i] ──[loss_of_function]──► [DISEASE]
    """

    reaction_id: str
    display_name: str
    left_ids: list[str]       # left-side entity IDs in the disease file
    right_ids: list[str]      # right-side entity IDs (empty → LoF)
    comments: list[str]
    omim_ids: list[str]
    disease_label: str
    mutant_variants: list[MutantVariant] = field(default_factory=list)
    stable_id: str = ""
    # left-side entity IDs that are variant proteins (excluded from branch finding)
    variant_left_ids: list[str] = field(default_factory=list)

    # filled after overlay
    branch_nodes: list[str] = field(default_factory=list)
    disease_node: str = ""

    @property
    def is_loss_of_function(self) -> bool:
        return len(self.right_ids) == 0


@dataclass
class OverlayResult:
    graph: nx.DiGraph
    anchored: list[DiseaseReaction]
    unanchored: list[DiseaseReaction]
    disease_nodes: list[str]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DiseaseOverlay:
    """
    Overlay Reactome disease reactions onto a healthy pathway graph.

    Parameters
    ----------
    parser_cls :
        The ReactomeBioPAX class (or any _ParserBase subclass).
        Used to instantiate a fresh parser for the disease file.
    healthy_parser :
        The already-used parser instance that built the healthy graph.
        Its entity stores (proteins, molecules, complexes, …) are consulted
        when the disease file references IDs that were only defined in the
        healthy file.
    logger :
        Optional stdlib Logger.
    fuzzy_name_match :
        Allow case-insensitive substring matching when exact look-up fails.
    attach_unanchored_to_roots :
        Wire unanchored disease reactions to root nodes with low-confidence flag.
    expand_mutant_variants :
        If True (default) each OR-member becomes its own [MUT] node and the
        reaction sub-graph is replicated per variant (cartesian product).
        If False, a single aggregate [MUT] node is used per reaction.
    decompose_complex_fallback :
        If True (default), when a left-side complex from the disease file
        cannot be matched to the healthy graph directly, its leaf protein/
        molecule components are tried individually as branch nodes.
    """

    def __init__(
        self,
        parser_cls,
        healthy_parser,
        logger: typing.Optional[logging.Logger] = None,
        fuzzy_name_match: bool = True,
        attach_unanchored_to_roots: bool = False,
        expand_mutant_variants: bool = True,
        decompose_complex_fallback: bool = True,
    ):
        self._cls = parser_cls
        self._hp = healthy_parser
        self.logger = logger
        self.fuzzy_name_match = fuzzy_name_match
        self.attach_unanchored = attach_unanchored_to_roots
        self.expand_variants = expand_mutant_variants
        self.decompose_complex_fallback = decompose_complex_fallback

        self._healthy_id_to_label: dict[str, str] = {}
        self._healthy_name_to_label: dict[str, list[str]] = {}
        # Cached set of all ModificationFeature IDs in the current disease file
        self._mod_feature_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        healthy_graph: nx.DiGraph,
        disease_biopax_file: str,
        copy: bool = False,
    ) -> OverlayResult:
        """
        Parse *disease_biopax_file* and merge disease branches into
        *healthy_graph*.
        """
        G = healthy_graph.copy() if copy else healthy_graph

        self._log("info", f"Indexing healthy graph ({G.number_of_nodes()} nodes) …")
        self._build_healthy_index(G)

        # ── Parse the disease file with a fresh parser instance ──────────
        self._log("info", f"Parsing disease BioPAX: {disease_biopax_file}")
        dp = self._cls(
            uniprot_accession_num=getattr(self._hp, "uniprot_accession_num", False),
            logger=self.logger,
        )
        dp.parse_biopax3_file(disease_biopax_file)
        self._dp = dp

        # Pre-compute the set of ModificationFeature IDs for variant detection
        self._mod_feature_ids = self._build_mod_feature_ids(dp)

        # ── Extract disease / OMIM metadata from Pathway elements ────────
        rxn_to_disease, root_label, root_omim = self._mine_pathway_disease(dp)

        # ── Catalysis map: reaction_id → controller entity ID ────────────
        catalysis_map = {
            rxn_id: cat["controller"] for rxn_id, cat in dp.catalysis_dets.items()
        }

        # ── Build DiseaseReaction objects ────────────────────────────────
        disease_reactions = self._collect_all_reactions(
            dp, rxn_to_disease, root_label, root_omim, catalysis_map
        )
        self._log("info", f"Found {len(disease_reactions)} disease reactions")

        # ── Sort by disease file's own PathwayStep ordering ───────────────
        rxn_to_step_key = self._build_rxn_step_order(dp)
        disease_reactions.sort(
            key=lambda dr: rxn_to_step_key.get(dr.reaction_id, ("~", 10**9))
        )

        anchored: list[DiseaseReaction] = []
        unanchored: list[DiseaseReaction] = []
        disease_nodes: list[str] = []
        base_time = self._max_time(G)

        for rank, dr in enumerate(disease_reactions):
            branch_nodes = self._find_branch_nodes(dr)
            dr.branch_nodes = branch_nodes

            if not branch_nodes:
                if self.attach_unanchored:
                    branch_nodes = [n for n in G.nodes if G.in_degree(n) == 0]
                if not branch_nodes:
                    self._log("debug", f"  Unanchored: {dr.display_name!r}")
                    unanchored.append(dr)
                    continue

            t = base_time + 1 + rank
            disease_node_label = self._make_disease_node_label(dr)
            dr.disease_node = disease_node_label

            if disease_node_label not in G:
                G.add_node(
                    disease_node_label,
                    type="disease_phenotype",
                    label=dr.disease_label,
                    display_name=dr.display_name,
                    source_reaction=dr.reaction_id,
                    stable_id=dr.stable_id,
                    omim_ids=dr.omim_ids,
                    is_loss_of_function=dr.is_loss_of_function,
                    synthetic=True,
                    color="purple",
                )
                disease_nodes.append(disease_node_label)

            # ── Build mutant-variant nodes ────────────────────────────────
            if self.expand_variants and dr.mutant_variants:
                mut_labels = []
                for mv in dr.mutant_variants:
                    lbl = f"[MUT] {mv.display_name}"
                    if lbl not in G:
                        G.add_node(
                            lbl,
                            type="mutant_variant",
                            display_name=mv.display_name,
                            reactome_db_id=mv.reactome_db_id,
                            uniprot_id=mv.uniprot_id,
                            omim_ids=mv.omim_ids,
                            mutations=mv.mutations,
                            clingen_ids=mv.clingen_ids,
                            synthetic=False,
                            color="orange",
                        )
                    mut_labels.append(lbl)
            else:
                agg = f"[MUT] {dr.display_name}"
                if agg not in G:
                    G.add_node(
                        agg,
                        type="mutant_variant",
                        display_name=dr.display_name,
                        synthetic=False,
                        color="orange",
                    )
                mut_labels = [agg]

            # ── Resolve right-side products ───────────────────────────────
            right_labels = self._resolve_entity_labels(dr.right_ids)

            if not right_labels:
                # ── Loss-of-Function: cartesian product (variant × branch_node) ──
                for ml in mut_labels:
                    for bp_node in branch_nodes:
                        G.add_edge(
                            bp_node,
                            ml,
                            type="disease_branch",
                            time=t,
                            reaction=dr.reaction_id,
                            disease_label=dr.disease_label,
                            omim_ids=dr.omim_ids,
                            confidence="reactome_curated",
                        )
                    G.add_edge(
                        ml,
                        disease_node_label,
                        type="loss_of_function",
                        time=t + 1,
                        reaction=dr.reaction_id,
                    )
            else:
                # ── GoF / aberrant: cartesian product (variant × branch × right) ──

                # Ensure all right-side product nodes exist
                for r_lbl in right_labels:
                    if r_lbl not in G:
                        G.add_node(
                            r_lbl,
                            type="disease_product",
                            synthetic=True,
                            color="darkorange",
                        )

                # Reaction edges: branch_node → right_product (shared)
                for bp_node in branch_nodes:
                    for r_lbl in right_labels:
                        G.add_edge(
                            bp_node,
                            r_lbl,
                            type="reaction",
                            subtype="disease",
                            time=t,
                            reaction=dr.reaction_id,
                            disease_label=dr.disease_label,
                            omim_ids=dr.omim_ids,
                            confidence="reactome_curated",
                        )

                # Per-variant catalysis and aberrant_product edges
                for ml in mut_labels:
                    for bp_node in branch_nodes:
                        G.add_edge(
                            ml,
                            bp_node,
                            type="catalysis",
                            subtype="disease_aberrant",
                            time=t,
                            reaction=dr.reaction_id,
                        )
                    G.add_edge(
                        ml,
                        disease_node_label,
                        type="aberrant_product",
                        time=t + 1,
                        reaction=dr.reaction_id,
                    )

                # disease_progression: right products → disease phenotype
                for r_lbl in right_labels:
                    G.add_edge(
                        r_lbl,
                        disease_node_label,
                        type="disease_progression",
                        time=t + 2,
                    )

            dr.branch_nodes = branch_nodes
            anchored.append(dr)
            self._log(
                "debug",
                f"  Anchored {dr.disease_label!r} | "
                f"{len(dr.mutant_variants)} variants | "
                f"{len(branch_nodes)} branch point(s): "
                + ", ".join(branch_nodes[:3])
                + ("…" if len(branch_nodes) > 3 else ""),
            )

        self._log(
            "info",
            f"Overlay complete — {len(anchored)} anchored, "
            f"{len(unanchored)} unanchored, "
            f"{len(disease_nodes)} disease nodes added",
        )

        return OverlayResult(
            graph=G,
            anchored=anchored,
            unanchored=unanchored,
            disease_nodes=disease_nodes,
        )

    # ------------------------------------------------------------------
    # Reaction collection — uses dp's already-populated stores
    # ------------------------------------------------------------------

    def _collect_all_reactions(
        self,
        dp,
        rxn_to_disease: dict,
        root_label: str,
        root_omim: list,
        catalysis_map: dict,
    ) -> list[DiseaseReaction]:
        """
        Build DiseaseReaction objects from dp.reactions (populated by
        parse_biopax3_file → __parse_reactions).

        dp.reactions : { rxn_id: (left_ids, right_ids, reaction_type) }
        All reactions in the file are included — no keyword filter needed.

        Variant detection priority:
          1. Catalyst entity (GoF/aberrant reactions with a defined catalyst)
          2. Left-side OR-group entities whose name signals variation
             (e.g. "IRAK4 variants", "MyD88 variants")
          3. Left-side entities that have ModificationFeature elements (disease
             mutations — not just normal PTMs) and/or HGVS-like names
        """
        results = []
        xref_map = self._build_xref_map(dp)

        _RXN_TAGS = (
            "BiochemicalReaction",
            "TemplateReaction",
            "Degradation",
            "Conversion",
            "Transport",
            "TransportWithBiochemicalReaction",
        )

        for rxn_id, (left_ids, right_ids, rxn_type) in dp.reactions.items():
            # Display name & comments: search for the element in dp.tree
            display_name = rxn_id
            comments: list[str] = []
            elem = None
            for tag in _RXN_TAGS:
                elem = dp.tree.find(
                    f"biopax:{tag}[@{{{_NS['rdf']}}}ID='{rxn_id}']", _NS
                )
                if elem is not None:
                    dn_el = elem.find("biopax:displayName", _NS)
                    if dn_el is not None and dn_el.text:
                        display_name = dn_el.text.strip()
                    comments = [
                        c.text.strip()
                        for c in elem.findall("biopax:comment", _NS)
                        if c.text
                    ]
                    break

            # Stable Reactome ID from xref
            stable_id = ""
            if elem is not None:
                for xref_el in elem.findall("biopax:xref", _NS):
                    key = xref_el.get(_RES, "").strip("#")
                    entry = xref_map.get(key, {})
                    xid_val = entry.get("id", "")
                    xdb_val = entry.get("db", "")
                    if "Reactome" in xdb_val and xid_val:
                        stable_id = (
                            f"R-HSA-{xid_val}"
                            if not xid_val.startswith("R-")
                            else xid_val
                        )
                    elif xid_val.startswith("R-HSA-"):
                        stable_id = xid_val
                    if stable_id:
                        break

            # Disease label + OMIM from pathway mapping
            rxn_label, rxn_omim = rxn_to_disease.get(rxn_id, (root_label, root_omim))

            # ── Variant extraction ───────────────────────────────────────
            # Priority 1: catalyst entity (GoF / aberrant reactions)
            catalyst_id = catalysis_map.get(rxn_id)
            mutant_variants: list[MutantVariant] = []
            variant_left_ids: list[str] = []

            if catalyst_id:
                mutant_variants = self._expand_or_to_variants(catalyst_id, rxn_omim, xref_map)

            # Priority 2 & 3: left-side variant entities (covers LoF reactions)
            # Only search the left side when the catalyst gave nothing.
            if not mutant_variants:
                for eid in left_ids:
                    if self._is_variant_entity(eid):
                        variant_left_ids.append(eid)
                        mutant_variants.extend(
                            self._expand_or_to_variants(eid, rxn_omim, xref_map)
                        )

            # OMIM: prefer per-variant IDs, fall back to comment scan, then pathway
            omim_ids = list({oid for mv in mutant_variants for oid in mv.omim_ids})
            if not omim_ids:
                for c in comments:
                    for m in re.finditer(r"MIM:(\d+)", c):
                        if m.group(1) not in omim_ids:
                            omim_ids.append(m.group(1))
            if not omim_ids:
                omim_ids = rxn_omim[:]

            results.append(
                DiseaseReaction(
                    reaction_id=rxn_id,
                    display_name=display_name,
                    left_ids=list(left_ids),
                    right_ids=list(right_ids),
                    comments=comments,
                    omim_ids=omim_ids,
                    disease_label=rxn_label or display_name,
                    stable_id=stable_id,
                    mutant_variants=mutant_variants,
                    variant_left_ids=variant_left_ids,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Variant entity detection
    # ------------------------------------------------------------------

    def _build_mod_feature_ids(self, dp) -> set[str]:
        """Return the set of all ModificationFeature rdf:IDs in the disease XML."""
        root = dp.tree.getroot()
        return {
            el.get(_ID)
            for el in root.findall("biopax:ModificationFeature", _NS)
            if el.get(_ID)
        }

    def _is_variant_entity(self, eid: str) -> bool:
        """
        Return True if the entity at *eid* in the disease parser is a
        disease-causing mutant variant rather than a normal pathway participant.

        Heuristics (any one sufficient):
          a. Entity is an OR-group (has members) whose displayName contains
             "variant", "mutant", or "defect*"
          b. Entity (protein only) has at least one bp:feature that resolves to
             a ModificationFeature (amino-acid change in the disease file)
          c. Entity's displayName matches HGVS-like notation such as
             "IRAK4 E402*", "MyD88 R196C", "TICAM1 R141*", "UNC93B1 L230Afs*"
        """
        dp = self._dp
        for store in (dp.proteins, dp.physical_entity, dp.molecules):
            if eid not in store:
                continue
            data = store[eid]
            name = (data.get("name") or "").lower()

            # (a) Named OR-group
            if data.get("members") and any(
                w in name for w in ("variant", "mutant", "defect")
            ):
                return True

            # (c) HGVS-like name pattern
            raw_name = data.get("name") or ""
            if _VARIANT_NAME_RE.search(raw_name):
                return True
            break

        # (b) ModificationFeature in XML (only meaningful for proteins)
        return self._entity_has_mod_feature(eid)

    def _entity_has_mod_feature(self, eid: str) -> bool:
        """
        Return True if the entity element in the disease XML has any
        bp:feature child pointing to a ModificationFeature.

        ModificationFeatures encode disease mutations (amino-acid changes).
        FragmentFeatures and BindingFeatures are normal structural annotations
        and are intentionally excluded from this check.
        """
        if not self._mod_feature_ids:
            return False
        root = self._dp.tree.getroot()
        for tag in ("biopax:Protein", "biopax:PhysicalEntity", "biopax:Complex"):
            tag_name = tag.split(":")[1]
            elem = root.find(
                f"biopax:{tag_name}[@{{{_NS['rdf']}}}ID='{eid}']", _NS
            )
            if elem is None:
                continue
            for feat in elem.findall("biopax:feature", _NS):
                ref = feat.get(_RES, "").strip("#")
                if ref in self._mod_feature_ids:
                    return True
            return False  # element found but no ModificationFeature
        return False

    # ------------------------------------------------------------------
    # OR-group / variant expansion → MutantVariant list
    # ------------------------------------------------------------------

    def _expand_or_to_variants(
        self,
        catalyst_id: str,
        fallback_omim: list,
        xref_map: dict | None = None,
    ) -> list[MutantVariant]:
        """
        Recursively expand a catalyst / variant entity ID to its leaf
        MutantVariant objects.

        A BioPAX EntitySet (OR-group) has bp:memberPhysicalEntity children.
        This method walks those children recursively until it reaches leaves
        (entities with no members list), producing one MutantVariant per leaf.

        Structural complexes (AND — only bp:component, no members) are
        treated as a single variant leaf.

        Resolution order per entity:
          1. self._dp stores  (disease file entities)
          2. self._hp stores  (healthy file entities — referenced by ID)

        ClinGen allele IDs are extracted from xref_map if provided.
        """
        if xref_map is None:
            xref_map = {}
        dp, hp = self._dp, self._hp
        seen: set = set()
        variants: list[MutantVariant] = []

        def _lookup(eid: str) -> dict | None:
            for store in (
                dp.proteins,
                dp.molecules,
                dp.dna,
                dp.rna,
                dp.physical_entity,
                dp.complexes,
                hp.proteins,
                hp.molecules,
                hp.dna,
                hp.rna,
                hp.physical_entity,
                hp.complexes,
            ):
                if eid in store:
                    return store[eid]
            return None

        def _clingen_ids(eid: str) -> list[str]:
            """Extract ClinGen allele IDs from protein xrefs."""
            root = dp.tree.getroot()
            elem = root.find(
                f"biopax:Protein[@{{{_NS['rdf']}}}ID='{eid}']", _NS
            )
            if elem is None:
                return []
            ids = []
            for xr in elem.findall("biopax:xref", _NS):
                key = xr.get(_RES, "").strip("#")
                entry = xref_map.get(key, {})
                if entry.get("db", "").lower() == "clingen":
                    cg_id = entry.get("id", "")
                    if cg_id:
                        ids.append(cg_id)
            return ids

        def _recurse(eid: str):
            if eid in seen:
                return
            seen.add(eid)
            data = _lookup(eid)
            if data is None:
                return

            # OR-group: has memberPhysicalEntity children → expand each leaf
            members = data.get("members", [])
            if members:
                for mid in members:
                    _recurse(mid)
                return

            # Leaf (either a bare protein/molecule or a structural complex)
            display_name = data.get("name") or eid
            omim_ids = list(data.get("omim_ids") or fallback_omim)
            uniprot_id = data.get("uniprot_id") or data.get("ref_id")

            variants.append(
                MutantVariant(
                    entity_id=eid,
                    display_name=display_name,
                    reactome_db_id=data.get("reactome_db_id"),
                    uniprot_id=uniprot_id,
                    omim_ids=omim_ids,
                    mutations=data.get("mutations") or [],
                    clingen_ids=_clingen_ids(eid),
                )
            )

        _recurse(catalyst_id)
        return variants

    # ------------------------------------------------------------------
    # Entity ID → graph labels
    # ------------------------------------------------------------------

    def _resolve_entity_labels(self, entity_ids: list[str]) -> list[str]:
        """
        Convert entity IDs from the disease file into graph-compatible
        display labels, expanding OR-groups to their leaves.
        """
        dp, hp = self._dp, self._hp
        seen: set = set()
        labels: list[str] = []

        for eid in entity_ids:
            pairs = dp._resolve_or_leaves(eid)
            if not pairs:
                pairs = hp._resolve_or_leaves(eid)
            for _, lbl in pairs:
                if lbl and lbl not in seen:
                    seen.add(lbl)
                    labels.append(lbl)

        return labels

    # ------------------------------------------------------------------
    # Branch-point resolution
    # ------------------------------------------------------------------

    def _find_branch_nodes(self, dr: DiseaseReaction) -> list[str]:
        """
        Map the LEFT substrate entity IDs of the disease reaction to nodes
        that already exist in the healthy graph.

        Variant entities (those in dr.variant_left_ids) are skipped — they
        are disease-causing mutants, not branch points.

        Resolution strategy:
          Pass 1 — direct: resolve each non-variant left entity to healthy
            graph nodes via reactome_db_id → exact name → fuzzy name.
          Pass 2 — complex decomposition fallback: for left complexes that
            did not match in Pass 1, decompose them to their leaf protein/
            molecule components and match those individually.
        """
        variant_set = set(dr.variant_left_ids)
        branch_nodes: list[str] = []
        seen: set = set()

        # Non-variant left entities
        candidate_ids = [eid for eid in dr.left_ids if eid not in variant_set]

        # Pass 1: direct resolution
        for eid in candidate_ids:
            for leaf_lbl in self._resolve_entity_labels([eid]):
                candidates = self._resolve_to_healthy(leaf_lbl, eid)
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        branch_nodes.append(c)

        # Pass 2: complex decomposition (only if Pass 1 produced nothing)
        if not branch_nodes and self.decompose_complex_fallback:
            for eid in candidate_ids:
                for leaf_lbl, leaf_id in self._decompose_entity_to_leaves(eid):
                    candidates = self._resolve_to_healthy(leaf_lbl, leaf_id)
                    for c in candidates:
                        if c not in seen:
                            seen.add(c)
                            branch_nodes.append(c)

        return branch_nodes

    def _decompose_entity_to_leaves(self, eid: str) -> list[tuple[str, str]]:
        """
        Return (label, entity_id) pairs for the leaf protein/molecule
        components of a complex (or the entity itself if not a complex).

        Used as a fallback when a disease complex cannot be name-matched
        to the healthy graph — individual component proteins usually can.
        """
        dp = self._dp
        if eid not in dp.complexes:
            lbl = dp._make_label_for_id(eid)
            return [(lbl, eid)] if lbl else []

        visited: set = set()
        leaves: list[tuple[str, str]] = []

        def _recurse(cid: str):
            if cid in visited:
                return
            visited.add(cid)
            if cid in dp.complexes:
                # Try members (OR-group) first, then components (structural complex)
                members = dp.complexes[cid].get("members", [])
                if members:
                    for mid in members:
                        _recurse(mid)
                else:
                    for comp in dp.complexes[cid].get("components", []):
                        _recurse(comp)
            else:
                lbl = dp._make_label_for_id(cid)
                if lbl:
                    leaves.append((lbl, cid))

        _recurse(eid)
        return leaves

    def _resolve_to_healthy(self, leaf_label: str, eid: str) -> list[str]:
        """Try reactome_db_id → exact name → fuzzy name."""
        dp, hp = self._dp, self._hp

        # 1. reactome_db_id exact match
        for store in (
            dp.proteins,
            dp.molecules,
            dp.dna,
            dp.rna,
            dp.physical_entity,
            dp.complexes,
            hp.proteins,
            hp.molecules,
            hp.dna,
            hp.rna,
            hp.physical_entity,
            hp.complexes,
        ):
            if eid in store:
                db_id = store[eid].get("reactome_db_id")
                if db_id and db_id in self._healthy_id_to_label:
                    return [self._healthy_id_to_label[db_id]]
                break

        # 2. Exact name match (case-insensitive)
        key = leaf_label.strip().lower()
        if key in self._healthy_name_to_label:
            return self._healthy_name_to_label[key]

        # 3. Fuzzy substring match
        if self.fuzzy_name_match:
            matches = [
                label
                for hkey, labels in self._healthy_name_to_label.items()
                if key in hkey or hkey in key
                for label in labels
            ]
            if matches:
                return matches

        return []

    # ------------------------------------------------------------------
    # Healthy graph index
    # ------------------------------------------------------------------

    def _build_healthy_index(self, G: nx.DiGraph):
        self._healthy_id_to_label = {}
        self._healthy_name_to_label = {}
        for node, data in G.nodes(data=True):
            db_id = data.get("reactome_db_id")
            if db_id:
                self._healthy_id_to_label[str(db_id)] = node
            name = data.get("name") or (node if isinstance(node, str) else "")
            if name:
                self._healthy_name_to_label.setdefault(name.strip().lower(), []).append(
                    node
                )
            for alias in data.get("aliases", []) or []:
                if alias:
                    self._healthy_name_to_label.setdefault(
                        alias.strip().lower(), []
                    ).append(node)

    # ------------------------------------------------------------------
    # Disease / OMIM / MONDO metadata from Pathway elements
    # ------------------------------------------------------------------

    def _mine_pathway_disease(self, dp) -> tuple[dict, str, list]:
        """
        Walk all bp:Pathway elements in dp.tree and build:
          rxn_to_disease : { reaction_id → (disease_label, [omim_ids]) }
          root_label     : str  (top pathway label)
          root_omim      : list (top pathway OMIM IDs)

        Supports:
          - OMIM IDs in bp:comment text ("MIM:XXXXXX")
          - OMIM / MONDO / Disease Ontology IDs via bp:xref elements
          - bp:disease attribute pointing to a DiseaseOntologyVocabulary
        """
        root = dp.tree.getroot()
        xref_map = self._build_xref_map(dp)

        # Build disease-vocabulary map for bp:disease attributes
        disease_vocab = self._build_disease_vocab(dp, xref_map)

        all_pathways = root.findall("biopax:Pathway", _NS)

        child_ids: set = set()
        for pw in all_pathways:
            for comp in pw.findall("biopax:pathwayComponent", _NS):
                child_ids.add(comp.get(_RES, "").strip("#"))

        def _mine_one(pw):
            dn_el = pw.find("biopax:displayName", _NS)
            display = dn_el.text.strip() if (dn_el is not None and dn_el.text) else ""
            disease_label, omim_ids = "", []

            # Check bp:disease attribute (newer Reactome files)
            disease_attr = pw.find("biopax:disease", _NS)
            if disease_attr is not None:
                vocab_ref = disease_attr.get(_RES, "").strip("#")
                vocab_entry = disease_vocab.get(vocab_ref, {})
                if vocab_entry.get("label"):
                    disease_label = vocab_entry["label"]
                omim_ids.extend(vocab_entry.get("omim_ids", []))

            # Parse comments for MIM IDs and disease names
            for c in pw.findall("biopax:comment", _NS):
                if not c.text:
                    continue
                for m in re.finditer(r"MIM:(\d+)", c.text, re.IGNORECASE):
                    if m.group(1) not in omim_ids:
                        omim_ids.append(m.group(1))
                # Extract disease names like "(SCID; MIM:102700)"
                for m in re.finditer(
                    r"\(([A-Za-z][^;)]{1,80});\s*MIM:\d+\)", c.text
                ):
                    if not disease_label:
                        disease_label = m.group(1).strip()
                if not disease_label:
                    for m in re.finditer(
                        r"causes\s+([A-Z][A-Z0-9\-]{1,20})\b", c.text
                    ):
                        disease_label = m.group(1).strip()
                        break

            # xref-based OMIM / MONDO IDs on the Pathway element
            for xref_el in pw.findall("biopax:xref", _NS):
                key = xref_el.get(_RES, "").strip("#")
                entry = xref_map.get(key, {})
                db = entry.get("db", "").lower()
                val = entry.get("id", "")
                if db in _DISEASE_DBS and val and val not in omim_ids:
                    omim_ids.append(val)

            if not disease_label:
                disease_label = display
            return disease_label, omim_ids

        # PathwayStep → reaction IDs
        step_to_rxns: dict = {}
        for ps in root.findall("biopax:PathwayStep", _NS):
            ps_id = ps.get(_ID, "")
            step_to_rxns[ps_id] = [
                sp.get(_RES, "").strip("#")
                for sp in ps.findall("biopax:stepProcess", _NS)
            ]
        pw_to_steps: dict = {}
        pw_by_id: dict = {}
        for pw in all_pathways:
            pw_id = pw.get(_ID, "")
            pw_by_id[pw_id] = pw
            pw_to_steps[pw_id] = [
                po.get(_RES, "").strip("#")
                for po in pw.findall("biopax:pathwayOrder", _NS)
            ]

        def _collect_rxns(pw_id, visiting=frozenset()):
            if pw_id in visiting:
                return []
            visiting = visiting | {pw_id}
            pw = pw_by_id.get(pw_id)
            if pw is None:
                return []
            rxns = []
            for comp in pw.findall("biopax:pathwayComponent", _NS):
                ref = comp.get(_RES, "").strip("#")
                if ref in pw_by_id:
                    rxns.extend(_collect_rxns(ref, visiting))
                else:
                    rxns.append(ref)
            for step_id in pw_to_steps.get(pw_id, []):
                for rxn_id in step_to_rxns.get(step_id, []):
                    if rxn_id not in pw_by_id:
                        rxns.append(rxn_id)
            return rxns

        rxn_to_disease: dict = {}
        root_label, root_omim = "", []
        for pw in all_pathways:
            pw_id = pw.get(_ID, "")
            label, omim_ids = _mine_one(pw)
            is_root = pw_id not in child_ids
            if is_root and not root_label:
                root_label, root_omim = label, omim_ids
            for rxn_id in _collect_rxns(pw_id):
                if rxn_id not in rxn_to_disease or not is_root:
                    rxn_to_disease[rxn_id] = (label, omim_ids)

        return rxn_to_disease, root_label, root_omim

    def _build_disease_vocab(self, dp, xref_map: dict) -> dict:
        """
        Build a map from vocabulary ID → {'label': str, 'omim_ids': list}
        for DiseaseOntologyVocabulary and similar elements (used with
        bp:disease attribute on Pathway/Interaction elements).
        """
        root = dp.tree.getroot()
        vocab: dict = {}
        for tag in (
            "biopax:DiseaseOntologyVocabulary",
            "biopax:MedicalOntologyVocabulary",
            "biopax:DiseaseTermVocabulary",
        ):
            tag_name = tag.split(":")[1]
            for el in root.findall(f"biopax:{tag_name}", _NS):
                vid = el.get(_ID, "")
                term_el = el.find("biopax:term", _NS)
                label = term_el.text.strip() if term_el is not None and term_el.text else ""
                omim_ids = []
                for xr in el.findall("biopax:xref", _NS):
                    key = xr.get(_RES, "").strip("#")
                    entry = xref_map.get(key, {})
                    db = entry.get("db", "").lower()
                    val = entry.get("id", "")
                    if db in _DISEASE_DBS and val:
                        omim_ids.append(val)
                if vid:
                    vocab[vid] = {"label": label, "omim_ids": omim_ids}
        return vocab

    # ------------------------------------------------------------------
    # Disease PathwayStep ordering (for time assignment)
    # ------------------------------------------------------------------

    def _build_rxn_step_order(self, dp) -> dict[str, tuple]:
        """
        Return {reaction_id → (pathway_id, local_order)} from the disease
        file's own PathwayStep chain.

        Used to sort disease reactions by their causal position within the
        disease pathway rather than by arbitrary enumeration order.
        """
        result: dict = {}
        for step_id, (step_dets, _) in dp.reactionOrder.items():
            rxn_id = step_dets[0]
            if rxn_id and rxn_id in dp.reactions:
                step_key = dp.step_order.get(step_id)
                if step_key is not None:
                    result[rxn_id] = step_key
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_xref_map(self, dp) -> dict:
        root = dp.tree.getroot()
        xref_map: dict = {}
        for tag in ("biopax:RelationshipXref", "biopax:UnificationXref"):
            for xref in root.findall(tag, _NS):
                xid = xref.get(_ID)
                db_el = xref.find("biopax:db", _NS)
                id_el = xref.find("biopax:id", _NS)
                if xid and db_el is not None and id_el is not None:
                    xref_map[xid] = {
                        "db": (db_el.text or "").strip(),
                        "id": (id_el.text or "").strip(),
                    }
        return xref_map

    @staticmethod
    def _make_disease_node_label(dr: DiseaseReaction) -> str:
        if dr.omim_ids:
            return f"[DISEASE] {dr.disease_label} (OMIM:{dr.omim_ids[0]})"
        return f"[DISEASE] {dr.disease_label}"

    @staticmethod
    def _max_time(G: nx.DiGraph) -> int:
        times = [d.get("time", 0) for _, _, d in G.edges(data=True) if d.get("time")]
        return max(times, default=0)

    def _log(self, level: str, msg: str):
        if self.logger is not None:
            getattr(self.logger, level)(msg)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, result: OverlayResult) -> str:
        lines = [
            "=" * 62,
            "DiseaseOverlay report",
            f"  Anchored   : {len(result.anchored)}",
            f"  Unanchored : {len(result.unanchored)}",
            f"  Disease nodes added : {len(result.disease_nodes)}",
            "-" * 62,
        ]
        for dr in result.anchored:
            lof = "LoF" if dr.is_loss_of_function else "GoF/aberrant"
            lines.append(
                f"  [{lof}] {dr.disease_label!r} "
                f"(OMIM:{dr.omim_ids[0] if dr.omim_ids else '—'}) "
                f"| {len(dr.mutant_variants)} variants "
                f"| branch: {dr.branch_nodes[:2]}"
            )
        if result.unanchored:
            lines += ["", "Unanchored:"]
            for dr in result.unanchored:
                lines.append(f"  {dr.display_name!r}")
        lines.append("=" * 62)
        return "\n".join(lines)

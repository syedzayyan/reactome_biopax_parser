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

2. OR-group catalyst → MutantVariant list is fully recursive.
   A controller entity that is an EntitySet (has bp:memberPhysicalEntity
   children) is walked to its leaves, producing one MutantVariant per leaf.
   Structural complexes (only bp:component, no members) are treated as a
   single leaf variant.

3. Cartesian product across mutant variants × reaction branches.
   Each [MUT] variant node gets its OWN copy of the reaction sub-graph:

       [MUT_i]  ──catalysis──►  bp_node  ──reaction──►  r_lbl
       [MUT_i]  ──────────────────────────────────────►  [DISEASE]
       r_lbl    ──disease_progression──────────────────► [DISEASE]

   For LoF reactions the cartesian product is across variants × branch nodes:

       bp_node  ──disease_branch──►  [MUT_i]  ──loss_of_function──►  [DISEASE]

4. Entity label resolution and OR-leaf expansion delegate to the
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

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MutantVariant:
    """A single disease-causing entity resolved from an OR-group catalyst."""

    entity_id: str
    display_name: str
    reactome_db_id: str | None
    uniprot_id: str | None
    omim_ids: list[str]
    mutations: list[str]


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
    left_ids: list[str]  # left-side entity IDs in the disease file
    right_ids: list[str]  # right-side entity IDs (empty → LoF)
    comments: list[str]
    omim_ids: list[str]
    disease_label: str
    mutant_variants: list[MutantVariant] = field(default_factory=list)
    stable_id: str = ""

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
    """

    def __init__(
        self,
        parser_cls,
        healthy_parser,
        logger: typing.Optional[logging.Logger] = None,
        fuzzy_name_match: bool = True,
        attach_unanchored_to_roots: bool = False,
        expand_mutant_variants: bool = True,
    ):
        self._cls = parser_cls
        self._hp = healthy_parser
        self.logger = logger
        self.fuzzy_name_match = fuzzy_name_match
        self.attach_unanchored = attach_unanchored_to_roots
        self.expand_variants = expand_mutant_variants

        self._healthy_id_to_label: dict[str, str] = {}
        self._healthy_name_to_label: dict[str, list[str]] = {}

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

        anchored: list[DiseaseReaction] = []
        unanchored: list[DiseaseReaction] = []
        disease_nodes: list[str] = []
        base_time = self._max_time(G)

        for idx, dr in enumerate(disease_reactions):
            branch_nodes = self._find_branch_nodes(dr)
            dr.branch_nodes = branch_nodes

            if not branch_nodes:
                if self.attach_unanchored:
                    branch_nodes = [n for n in G.nodes if G.in_degree(n) == 0]
                if not branch_nodes:
                    self._log("debug", f"  Unanchored: {dr.display_name!r}")
                    unanchored.append(dr)
                    continue

            t = base_time + 1 + idx * 3
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
                #
                # Each mutant variant gets its own disease_branch edge FROM
                # every branch node it disrupts, then its own LoF edge to
                # the disease phenotype.  Variants are independent disease
                # branches — they are not alternatives that collapse into a
                # single shared LoF edge.
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
                #
                # Right-side products are shared across all variants (they
                # represent the same aberrant output regardless of which
                # specific mutant drives it).  However EACH variant gets its
                # own catalysis edges pointing to EVERY branch node, and its
                # own aberrant_product edge to the disease node.  This means
                # the disease sub-graph is independently replicated per variant.

                # Ensure all right-side product nodes exist
                for r_lbl in right_labels:
                    if r_lbl not in G:
                        G.add_node(
                            r_lbl,
                            type="disease_product",
                            synthetic=True,
                            color="darkorange",
                        )

                # Reaction edges: branch_node → right_product
                # (shared; independent of which mutant is active)
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

            # Catalyst → expand the full OR-group to MutantVariant leaves.
            # _expand_or_to_variants walks memberPhysicalEntity recursively so
            # a controller that is a collection of mutant proteins produces one
            # MutantVariant per leaf, not just the top-level OR node.
            catalyst_id = catalysis_map.get(rxn_id)
            mutant_variants: list[MutantVariant] = []
            if catalyst_id:
                mutant_variants = self._expand_or_to_variants(catalyst_id, rxn_omim)

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
                )
            )

        return results

    # ------------------------------------------------------------------
    # OR-group expansion → MutantVariant list
    # ------------------------------------------------------------------

    def _expand_or_to_variants(
        self, catalyst_id: str, fallback_omim: list
    ) -> list[MutantVariant]:
        """
        Recursively expand a catalyst entity ID to its leaf MutantVariant
        objects.

        A BioPAX EntitySet (OR-group) has bp:memberPhysicalEntity children.
        This method walks those children recursively until it reaches leaves
        (entities with no members list), producing one MutantVariant per leaf.

        Structural complexes (AND — only bp:component, no members) are
        treated as a single variant leaf.

        Resolution order per entity:
          1. self._dp stores  (disease file entities)
          2. self._hp stores  (healthy file entities — referenced by ID)
        """
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
                )
            )

        _recurse(catalyst_id)
        return variants

    # ------------------------------------------------------------------
    # Entity ID → graph labels
    # Delegates to the disease parser's _resolve_or_leaves + _make_label
    # (same method available on any ReactomeBioPAX / _NxGraphMixin instance).
    # ------------------------------------------------------------------

    def _resolve_entity_labels(self, entity_ids: list[str]) -> list[str]:
        """
        Convert entity IDs from the disease file into graph-compatible
        display labels, expanding OR-groups to their leaves.

        Delegates to dp._resolve_or_leaves (from _NxGraphMixin) which already
        handles arbitrary nesting and all entity stores.  Falls back to hp
        (healthy parser) for IDs not present in the disease file.

        Returns the flat union of all leaf labels — no cartesian product here
        (right-side products are not ANDed together at the label stage).
        """
        dp, hp = self._dp, self._hp
        seen: set = set()
        labels: list[str] = []

        for eid in entity_ids:
            # Try disease parser first
            pairs = dp._resolve_or_leaves(eid)
            if not pairs:
                # Fall back to healthy parser for IDs only defined there
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

        OR-groups on the left side are expanded to leaves first (the healthy
        graph builder has already dissolved OR-nodes via _add_member_nodes).

        Priority:
          1. reactome_db_id exact match
          2. display name / alias exact match (case-insensitive)
          3. Substring fuzzy match (if enabled)
        """
        branch_nodes: list[str] = []
        seen: set = set()

        for eid in dr.left_ids:
            for leaf_lbl in self._resolve_entity_labels([eid]):
                candidates = self._resolve_to_healthy(leaf_lbl, eid)
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        branch_nodes.append(c)

        return branch_nodes

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
    # Disease / OMIM metadata from Pathway elements
    # ------------------------------------------------------------------

    def _mine_pathway_disease(self, dp) -> tuple[dict, str, list]:
        """
        Walk all bp:Pathway elements in dp.tree and build:
          rxn_to_disease : { reaction_id → (disease_label, [omim_ids]) }
          root_label     : str  (top pathway label)
          root_omim      : list (top pathway OMIM IDs)
        """
        root = dp.tree.getroot()
        xref_map = self._build_xref_map(dp)
        all_pathways = root.findall("biopax:Pathway", _NS)

        child_ids: set = set()
        for pw in all_pathways:
            for comp in pw.findall("biopax:pathwayComponent", _NS):
                child_ids.add(comp.get(_RES, "").strip("#"))

        def _mine_one(pw):
            dn_el = pw.find("biopax:displayName", _NS)
            display = dn_el.text.strip() if (dn_el is not None and dn_el.text) else ""
            disease_label, omim_ids = "", []
            for c in pw.findall("biopax:comment", _NS):
                if not c.text:
                    continue
                for m in re.finditer(r"MIM:(\d+)", c.text, re.IGNORECASE):
                    if m.group(1) not in omim_ids:
                        omim_ids.append(m.group(1))
                for m in re.finditer(r"\(([A-Za-z][^;)]{1,80});\s*MIM:\d+\)", c.text):
                    if not disease_label:
                        disease_label = m.group(1).strip()
                if not disease_label:
                    for m in re.finditer(r"causes\s+([A-Z][A-Z0-9\-]{1,20})\b", c.text):
                        disease_label = m.group(1).strip()
                        break
            for xref_el in pw.findall("biopax:xref", _NS):
                key = xref_el.get(_RES, "").strip("#")
                entry = xref_map.get(key, {})
                if entry.get("db", "").upper() in ("OMIM", "MIM"):
                    val = entry.get("id", "")
                    if val and val not in omim_ids:
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

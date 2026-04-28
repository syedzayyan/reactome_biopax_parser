import itertools
import logging
import typing

import networkx as nx

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):
        return iterable


class _NxGraphMixin:
    def parse_biopax_into_networkx(
        self,
        filename: str,
        reaction_partners: bool = False,
        include_complexes: bool = True,
        logger: typing.Optional[logging.Logger] = None,
    ):
        """Parse a BioPAX Level 3 file into a directed NetworkX graph.

        Args:
            filename (str): Path to the BioPAX Level 3 file.
            reaction_partners (bool): If True, add bidirectional edges between
                all left-side co-reactants (left_reactant) and all right-side
                co-products (right_product) in each reaction.  OR-group nodes
                participate in these edges like any other entity; the only
                excluded pairs are an OR node and its own direct members,
                because those members are alternatives for the OR node itself,
                not independent co-participants.
            include_complexes (bool): If True (default), complex entities are
                added as nodes with complex_component edges to their members.
                If False, complexes are not added as nodes — instead,
                bidirectional co-membership edges are added between all leaf
                participants of each complex, and reaction edges reference
                leaf proteins directly.
            logger (logging.Logger, optional): Logger instance.

        Returns:
            networkx.DiGraph: Directed graph of biochemical interactions.

        Edge types
        ----------
        reaction          left entity → right entity          directed
        catalysis         catalyst → left entity              directed
        left_reactant     left ↔ left co-reactant             bidirectional
        right_product     right ↔ right co-product            bidirectional
        translocation     left location → right location      directed
        expression        DNA/gene → each protein product     directed
        complex_component member → complex (or member ↔ member when
                          include_complexes=False)             directed / bidir
        """
        if logger is not None:
            self.logger = logger

        self._log("info", f"Building NetworkX graph from BioPAX file: {filename}")
        self.parse_biopax3_file(filename)

        # Reset caches for this parse
        self._label_cache = {}
        self._name_cache = {}
        self._leaf_cache = {}
        self._member_cache = {}

        G = nx.DiGraph()
        complex_first_step: dict = {}

        # ------------------------------------------------------------------
        # Pass 1: reactions that appear inside a PathwayStep
        # ------------------------------------------------------------------
        # Sort by (pathway_id, local_order) so traversal within a pathway is
        # meaningful; ordering across sibling pathways is arbitrary but
        # deterministic. global_rank gives every edge a monotonic int for
        # time-series plots.
        order_items = sorted(
            self.reactionOrder.items(),
            key=lambda kv: self.step_order.get(kv[0], ("~", 10**9)),
            # "~" sorts after real pathway IDs; unclaimed steps land at the end
        )
        iterator = (
            tqdm(order_items, desc="Building graph", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else order_items
        )

        global_rank = 0
        stepped_reactions: set = set()
        for step_id, step_dets in iterator:
            reaction_id = step_dets[0][0]
            if reaction_id is None or reaction_id not in self.reactions:
                continue
            stepped_reactions.add(reaction_id)

            step_key = self.step_order.get(step_id, (None, None))
            self._build_reaction_edges(
                G,
                reaction_id,
                step_key,
                global_rank,
                complex_first_step,
                reaction_partners,
                include_complexes,
            )
            global_rank += 1

        # ------------------------------------------------------------------
        # Pass 2: reactions that are pathwayComponents but have no PathwayStep
        # ------------------------------------------------------------------
        unstep_count = 0
        for reaction_id in self.reactions:
            if reaction_id in stepped_reactions:
                continue
            self._build_reaction_edges(
                G,
                reaction_id,
                (None, None),  # no pathway / no local_order
                global_rank,
                complex_first_step,
                reaction_partners,
                include_complexes,
            )
            global_rank += 1
            unstep_count += 1

        if unstep_count:
            self._log(
                "debug",
                f"Pass-2 fallback: added {unstep_count} reactions not covered by any PathwayStep",
            )

        self._log(
            "info",
            f"Reaction graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges across {global_rank} reactions "
            f"({len(stepped_reactions)} stepped, {unstep_count} unstepped)",
        )

        if include_complexes:
            self._log("debug", "Building complex hierarchy...")
            self._build_complex_hierarchy(G, complex_first_step)
        else:
            self._log("debug", "Building complex co-membership edges...")
            self._build_complex_comembership(G, complex_first_step)

        self._log("debug", "Dissolving OR-group nodes into members...")
        self._add_member_nodes(G)

        self._log("debug", "Removing self-loops...")
        G.remove_edges_from(list(nx.selfloop_edges(G)))

        self._log("debug", "Annotating node types...")
        self._annotate_nodes(G)

        self._log("info", "Graph construction complete")
        self._debug_cfs = complex_first_step
        return self.finalize_graph(G)

    # -------------------------------------------------------------------------
    # Per-reaction edge builder
    # -------------------------------------------------------------------------

    def _build_reaction_edges(
        self,
        G: nx.DiGraph,
        reaction_id: str,
        step_key: tuple,  # (pathway_id, local_order) — or (None, None)
        global_rank: int,  # monotonic int across all edges for time-series plots
        complex_first_step: dict,
        reaction_partners: bool,
        include_complexes: bool,
    ) -> None:
        """..."""
        left_entities, right_entities, reaction_type = self.reactions[reaction_id]
        pathway_name = self.reaction_to_pathway.get(reaction_id)
        catalysis_exists = self.catalysis_dets.get(reaction_id)
        expanded_catalysts = self._expand_catalyst(catalysis_exists)

        # Unpack once; every edge below uses these three.
        step_pathway, local_order = step_key if step_key else (None, None)

        # Record first step each complex appears in
        for entity_id in itertools.chain(left_entities, right_entities):
            self._record_complex_step(entity_id, global_rank, complex_first_step)

        # ------------------------------------------------------------------
        # Translocation branch
        # ------------------------------------------------------------------
        if reaction_type == "translocation":
            for left, right in zip(left_entities, right_entities):
                if left in self.physical_entity or right in self.physical_entity:
                    continue

                left_ids = self._expand_single_entity(left)
                right_ids = self._expand_single_entity(right)

                right_by_name: dict = {}
                for rid in right_ids:
                    name = self._get_entity_name(rid)
                    if name:
                        right_by_name[name] = rid

                for lid in left_ids:
                    name = self._get_entity_name(lid)
                    if not name or name not in right_by_name:
                        continue
                    rid = right_by_name[name]
                    left_label = self._make_label_for_id(lid)
                    right_label = self._make_label_for_id(rid)
                    if not left_label or not right_label or left_label == right_label:
                        continue

                    G.add_node(left_label)
                    G.add_node(right_label)
                    G.add_edge(
                        left_label,
                        right_label,
                        time=global_rank,
                        local_order=local_order,
                        pathway=pathway_name,
                        type="translocation",
                    )
                    for cat_id in expanded_catalysts:
                        if cat_id is not None:
                            cat_label = self._make_label_for_id(cat_id)
                            if cat_label:
                                G.add_node(cat_label)
                                G.add_edge(
                                    cat_label,
                                    left_label,
                                    time=global_rank,
                                    type="catalysis",
                                    local_order=local_order,
                                    catalysis_type=catalysis_exists["controlType"],
                                    pathway=pathway_name,
                                )

        # ------------------------------------------------------------------
        # Broadcast branch
        # ------------------------------------------------------------------
        elif reaction_type == "broadcast":
            flat_right = self._flatten_or_members(right_entities)

            for left in left_entities:
                left_label = self._make_label_for_id(left)
                if not left_label:
                    continue
                G.add_node(left_label)
                for right in flat_right:
                    right_label = self._make_label_for_id(right)
                    if not right_label:
                        continue
                    G.add_node(right_label)
                    G.add_edge(
                        left_label,
                        right_label,
                        time=global_rank,
                        local_order=local_order,
                        type="expression",
                        pathway=pathway_name,
                    )

        # ------------------------------------------------------------------
        # Normal reaction branch
        # ------------------------------------------------------------------
        else:
            if not include_complexes:
                # Flatten complexes to their leaf members so the graph never
                # contains complex nodes.  This is a flat union — no cartesian
                # product.  NX deduplicates edges natively so any repeated
                # (src, dst) pair from shared sub-complexes is harmless.
                left_entities = self._flatten_to_leaves(left_entities)
                right_entities = self._flatten_to_leaves(right_entities)

            # Resolve labels for both sides up front.
            # We store (label, entity_id) so we can check OR membership later.
            left_resolved: list[tuple[str, str]] = []
            for eid in left_entities:
                lbl = self._make_label_for_id(eid)
                if lbl:
                    G.add_node(lbl)
                    left_resolved.append((lbl, eid))

            right_resolved: list[tuple[str, str]] = []
            for eid in right_entities:
                lbl = self._make_label_for_id(eid)
                if lbl:
                    G.add_node(lbl)
                    right_resolved.append((lbl, eid))

            # Catalysis nodes
            for cat_id in expanded_catalysts:
                cat_label = self._make_label_for_id(cat_id) if cat_id else None
                if cat_label:
                    G.add_node(cat_label)

                # Reaction edges: left → right  (flat O(L×R), no cartesian)
                for left_label, _ in left_resolved:
                    if cat_label is not None:
                        G.add_edge(
                            cat_label,
                            left_label,
                            time=global_rank,
                            local_order=local_order,
                            type="catalysis",
                            catalysis_type=catalysis_exists["controlType"],
                            pathway=pathway_name,
                        )
                    for right_label, _ in right_resolved:
                        G.add_edge(
                            left_label,
                            right_label,
                            time=global_rank,
                            local_order=local_order,
                            type="reaction",
                            pathway=pathway_name,
                        )

            # ------------------------------------------------------------------
            # Reaction-partner edges
            # ------------------------------------------------------------------
            # left_reactant  : bidirectional between every pair of left-side
            #                  participants, including OR-group nodes.
            # right_product  : bidirectional between every pair of right-side
            #                  participants, including OR-group nodes.
            #
            # The only excluded pairs are (OR node, one of its own direct
            # members), because a member is an *alternative* for its OR parent,
            # not an independent co-participant alongside it.  Every other
            # combination — including OR node ↔ regular node, and OR node ↔
            # different OR node — gets partner edges normally.
            if reaction_partners:
                # Build the forbidden set: {frozenset({or_label, member_label})}
                or_member_pairs: set[frozenset] = set()
                all_resolved = left_resolved + right_resolved
                for lbl, eid in all_resolved:
                    direct_members = self._get_direct_members(eid)
                    for mid in direct_members:
                        m_label = self._make_label_for_id(mid)
                        if m_label:
                            or_member_pairs.add(frozenset({lbl, m_label}))

                left_labels = [lbl for lbl, _ in left_resolved]
                right_labels = [lbl for lbl, _ in right_resolved]

                for i, l1 in enumerate(left_labels):
                    for l2 in left_labels[i + 1 :]:
                        if frozenset({l1, l2}) in or_member_pairs:
                            continue
                        G.add_edge(
                            l1,
                            l2,
                            time=global_rank,
                            local_order=local_order,
                            type="left_reactant",
                            pathway=pathway_name,
                        )
                        G.add_edge(
                            l2,
                            l1,
                            time=global_rank,
                            local_order=local_order,
                            type="left_reactant",
                            pathway=pathway_name,
                        )

                for i, r1 in enumerate(right_labels):
                    for r2 in right_labels[i + 1 :]:
                        if frozenset({r1, r2}) in or_member_pairs:
                            continue
                        G.add_edge(
                            r1,
                            r2,
                            time=global_rank,
                            local_order=local_order,
                            type="right_product",
                            pathway=pathway_name,
                        )
                        G.add_edge(
                            r2,
                            r1,
                            time=global_rank,
                            local_order=local_order,
                            type="right_product",
                            pathway=pathway_name,
                        )

    # -------------------------------------------------------------------------
    # Post-construction cleanup
    # -------------------------------------------------------------------------

    def finalize_graph(self, G, verbose=False):
        """
        Post-construction cleanup:
        1. Remove degree-0 nodes (pathway-unreachable OR-expansion orphans)
        2. Report final graph stats
        """
        orphans = [n for n in G.nodes if G.degree(n) == 0]

        if verbose:
            or_orphans = [n for n in orphans if str(n).startswith("[OR]")]
            leaf_orphans = [n for n in orphans if not str(n).startswith("[OR]")]
            print(f"Pruning {len(orphans)} degree-0 nodes:")
            print(f"  Remaining [OR] nodes (should be 0): {len(or_orphans)}")
            print(f"  Leaf nodes with no pathway context: {len(leaf_orphans)}")
            if or_orphans:
                print("  !! Unexpected [OR] survivors:")
                for n in or_orphans[:5]:
                    print(f"    {n!r}")

        G.remove_nodes_from(orphans)

        if verbose:
            print(f"\nFinal graph:")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")

            from collections import Counter

            edge_types = Counter(
                d.get("type", "unknown") for _, _, d in G.edges(data=True)
            )
            print(f"  Edge types:")
            for etype, count in edge_types.most_common():
                print(f"    {etype}: {count}")

            node_types = Counter(
                d.get("type", "unknown") for _, d in G.nodes(data=True)
            )
            print(f"  Node types:")
            for ntype, count in node_types.most_common():
                print(f"    {ntype}: {count}")

        return G

    # -------------------------------------------------------------------------
    # Complex hierarchy (include_complexes=True)
    # -------------------------------------------------------------------------

    def _build_complex_hierarchy(self, G: nx.DiGraph, complex_first_step: dict):
        visited = set()

        def wire(cid: str, inherited_step):
            if cid in visited:
                return
            visited.add(cid)
            cdata = self.complexes[cid]
            parent_label = self._make_label(cdata["name"], cdata)
            step = complex_first_step.get(parent_label, inherited_step)

            if parent_label not in G.nodes:
                G.add_node(parent_label)
                G.nodes[parent_label].update(type="complex", **cdata)

            for component_id in cdata["components"]:
                if component_id in self.complexes:
                    sub_cdata = self.complexes[component_id]
                    sub_label = self._make_label(sub_cdata["name"], sub_cdata)

                    sub_members = sub_cdata.get("members", [])
                    if sub_members:
                        for leaf_id, leaf_label in self._resolve_or_leaves(
                            component_id
                        ):
                            if not leaf_label:
                                continue
                            if leaf_label not in G.nodes:
                                leaf_data, leaf_type = self._find_in_stores(leaf_id)
                                if leaf_data is None and leaf_id in self.complexes:
                                    leaf_data = self.complexes[leaf_id]
                                    leaf_type = "complex"
                                if leaf_data:
                                    G.add_node(leaf_label)
                                    G.nodes[leaf_label].update(
                                        type=leaf_type, **leaf_data
                                    )
                            if leaf_label in G.nodes and not G.has_edge(
                                leaf_label, parent_label
                            ):
                                G.add_edge(
                                    leaf_label,
                                    parent_label,
                                    type="complex_component",
                                    time=step,
                                )
                    else:
                        if sub_label not in G.nodes:
                            G.add_node(sub_label)
                            G.nodes[sub_label].update(type="complex", **sub_cdata)
                        if not G.has_edge(sub_label, parent_label):
                            G.add_edge(
                                sub_label,
                                parent_label,
                                type="complex_component",
                                time=step,
                            )
                        wire(component_id, step)

                else:
                    leaf_pairs = self._resolve_or_leaves(component_id)

                    for leaf_id, leaf_label in leaf_pairs:
                        if not leaf_label:
                            continue
                        if leaf_label not in G.nodes:
                            leaf_data, leaf_type = self._find_in_stores(leaf_id)
                            if leaf_data is None and leaf_id in self.complexes:
                                leaf_data = self.complexes[leaf_id]
                                leaf_type = "complex"
                            if leaf_data:
                                G.add_node(leaf_label)
                                G.nodes[leaf_label].update(type=leaf_type, **leaf_data)
                        if leaf_label in G.nodes and not G.has_edge(
                            leaf_label, parent_label
                        ):
                            G.add_edge(
                                leaf_label,
                                parent_label,
                                type="complex_component",
                                time=step,
                            )

        for cid in self.complexes:
            label = self._make_label(self.complexes[cid]["name"], self.complexes[cid])
            if label in complex_first_step:
                wire(cid, complex_first_step[label])

        fallback_step = max(complex_first_step.values(), default=0)
        for cid in self.complexes:
            label = self._make_label(self.complexes[cid]["name"], self.complexes[cid])
            if label in G.nodes and label not in complex_first_step:
                wire(cid, fallback_step)

    def _resolve_or_leaves(self, eid: str, visiting: frozenset = frozenset()) -> list:
        """
        Recursively resolve an entity to its non-OR leaf (id, label) pairs.
        Handles arbitrary nesting. Cycle-safe via immutable visiting set.
        Checks ALL stores — proteins, molecules, dna, rna,
        physicalentity, complexes.
        """
        if eid in visiting:
            return []
        visiting = visiting | {eid}
        for store in [
            self.proteins,
            self.molecules,
            self.dna,
            self.rna,
            self.physical_entity,
            self.complexes,
        ]:
            if eid in store:
                edata = store[eid]
                members = edata.get("members", [])
                if not members:
                    return [(eid, self._make_label(edata["name"], edata))]
                leaves = []
                for mid in members:
                    leaves.extend(self._resolve_or_leaves(mid, visiting))
                return leaves
        return []

    # -------------------------------------------------------------------------
    # Complex co-membership edges (include_complexes=False)
    # -------------------------------------------------------------------------

    def _build_complex_comembership(self, G: nx.DiGraph, complex_first_step: dict):
        """Add bidirectional co-membership edges between all leaf participants
        of each complex, without adding the complex itself as a node.

        Only complexes reachable from reaction-participating complexes are
        processed.  Leaf resolution is cached so shared sub-complexes are
        only traversed once.

        Performance note: leaf_cache eliminates redundant traversals of shared
        sub-complexes, which is the main source of slowness when
        include_complexes=False.  The inner loop is O(leaves²) per complex but
        leaves are proteins/molecules (already in G), so the set is small in
        practice and NX deduplicates any repeated edges for free.
        """
        leaf_cache: dict = {}

        def _get_leaves(cid: str, visiting: frozenset) -> list:
            if cid in leaf_cache:
                return leaf_cache[cid]
            if cid in visiting:
                return []
            visiting = visiting | {cid}
            leaves = []
            for component_id in self.complexes[cid]["components"]:
                if component_id in self.complexes:
                    leaves.extend(_get_leaves(component_id, visiting))
                else:
                    # OR-group non-complex component: expand to its leaves
                    resolved = self._resolve_or_leaves(component_id)
                    if resolved:
                        leaves.extend(lid for lid, _ in resolved)
                    else:
                        leaves.append(component_id)
            leaf_cache[cid] = leaves
            return leaves

        edges_added = 0
        for cid, cdata in self.complexes.items():
            label = self._make_label(cdata["name"], cdata)
            if label not in complex_first_step:
                continue
            step = complex_first_step[label]
            leaves = _get_leaves(cid, frozenset())

            # Deduplicate and keep only nodes already in G (i.e. reaction
            # participants).  Silently drop leaves that never appeared in any
            # reaction — they have no pathway context and would become orphans.
            seen: set = set()
            leaf_labels: list = []
            for lid in leaves:
                lbl = self._make_label_for_id(lid)
                if lbl and lbl in G.nodes and lbl not in seen:
                    seen.add(lbl)
                    leaf_labels.append(lbl)

            for i, l1 in enumerate(leaf_labels):
                for l2 in leaf_labels[i + 1 :]:
                    if not G.has_edge(l1, l2):
                        G.add_edge(l1, l2, type="complex_component", time=step)
                        edges_added += 1
                    if not G.has_edge(l2, l1):
                        G.add_edge(l2, l1, type="complex_component", time=step)
                        edges_added += 1

        self._log("debug", f"Complex co-membership: {edges_added} edges added")

    # -------------------------------------------------------------------------
    # OR-group dissolution
    # -------------------------------------------------------------------------

    def _add_member_nodes(self, G: nx.DiGraph):
        """
        Dissolve all OR-group nodes into their leaf members.
        Handles arbitrary nesting depth (OR-of-OR-of-OR) via recursive
        leaf resolution. A single pass is sufficient.
        """
        all_stores = [
            (self.proteins, "protein"),
            (self.molecules, "small_molecule"),  # was "molecule"
            (self.dna, "dna"),
            (self.rna, "rna"),
            (self.physical_entity, "physical_entity"),
            (self.complexes, "complex"),
        ]

        def _get_entry(eid):
            for store, stype in all_stores:
                if eid in store:
                    return store[eid], stype
            return None, None

        def _resolve_leaves(eid, visiting=frozenset()):
            if eid in visiting:
                return []
            edata, etype = _get_entry(eid)
            if edata is None:
                return []
            members = edata.get("members", [])
            if not members:
                return [(eid, edata, etype)]
            leaves = []
            for mid in members:
                leaves.extend(_resolve_leaves(mid, visiting | {eid}))
            return leaves

        for store, nodetype in all_stores:
            for entityid, entitydata in store.items():
                members = entitydata.get("members", [])
                if not members:
                    continue

                parent_label = self._make_label(entitydata["name"], entitydata)
                leaf_entities = _resolve_leaves(entityid)

                for leaf_id, leaf_data, leaf_type in leaf_entities:
                    leaf_label = self._make_label(leaf_data["name"], leaf_data)
                    if leaf_label not in G.nodes:
                        G.add_node(leaf_label)
                        G.nodes[leaf_label].update(type=leaf_type, **leaf_data)

                if parent_label not in G.nodes:
                    continue

                out_edges = list(G.out_edges(parent_label, data=True))
                in_edges = list(G.in_edges(parent_label, data=True))
                if any(d.get("type") == "expression" for _, _, d in out_edges):
                    continue

                for leaf_id, leaf_data, leaf_type in leaf_entities:
                    leaf_label = self._make_label(leaf_data["name"], leaf_data)
                    for _, neighbor, edgedata in out_edges:
                        if neighbor != leaf_label:
                            G.add_edge(leaf_label, neighbor, **edgedata)
                    for neighbor, _, edgedata in in_edges:
                        if neighbor != leaf_label:
                            G.add_edge(neighbor, leaf_label, **edgedata)

                G.remove_node(parent_label)

        or_survivors = [
            n
            for n in G.nodes
            if str(n).startswith("[OR]")
            and not any(
                d.get("type") == "expression" for _, _, d in G.out_edges(n, data=True)
            )
        ]
        if or_survivors:
            self._log("debug", f"Sweeping {len(or_survivors)} surviving [OR] nodes")
            G.remove_nodes_from(or_survivors)

    # -------------------------------------------------------------------------
    # Node annotation
    # -------------------------------------------------------------------------

    def _annotate_nodes(self, G: nx.DiGraph):
        """Annotate all graph nodes with type and entity metadata."""
        label_to_annotation: dict = {}
        for store, node_type in [
            (self.proteins, "protein"),
            (self.molecules, "small_molecule"),
            (self.dna, "dna"),
            (self.rna, "rna"),
            (self.physical_entity, "physical_entity"),
            (self.complexes, "complex"),
        ]:
            for entity_id, entity_data in store.items():
                label = self._make_label(entity_data["name"], entity_data)
                if entity_data.get("members"):
                    if label in G.nodes and any(
                        d.get("type") == "expression"
                        for _, _, d in G.out_edges(label, data=True)
                    ):
                        label_to_annotation[label] = (node_type, entity_data)
                    continue
                label_to_annotation[label] = (node_type, entity_data)

        for node in G.nodes:
            if G.nodes[node].get("type"):
                continue
            if node in label_to_annotation:
                node_type, entity_data = label_to_annotation[node]
                G.nodes[node].update({"type": node_type, **entity_data})
            else:
                G.nodes[node]["type"] = "other"

    # -------------------------------------------------------------------------
    # Catalyst expansion
    # -------------------------------------------------------------------------

    def _expand_catalyst(self, catalysis_exists) -> list:
        """Return expanded catalyst IDs, or [None] if no catalysis."""
        if catalysis_exists is None:
            return [None]
        return self._expand_single_entity(catalysis_exists["controller"])

    # -------------------------------------------------------------------------
    # Member / entity expansion
    # -------------------------------------------------------------------------

    def _expand_single_entity(self, entity_id: str) -> list:
        """Return member alternatives for an entity, or [entity_id] if none.
        Results are cached.
        """
        if not hasattr(self, "_member_cache"):
            self._member_cache = {}
        if entity_id in self._member_cache:
            return self._member_cache[entity_id]
        result = [entity_id]
        for store in (
            self.proteins,
            self.molecules,
            self.dna,
            self.rna,
            self.physical_entity,
            self.complexes,
        ):
            if entity_id in store:
                members = store[entity_id].get("members", [])
                if members:
                    result = members
                break
        self._member_cache[entity_id] = result
        return result

    def _flatten_to_leaves(self, entity_list: list) -> list:
        if not hasattr(self, "_leaf_cache"):
            self._leaf_cache = {}
        result = []
        for eid in entity_list:
            if eid in self._leaf_cache:
                result.extend(self._leaf_cache[eid])
                continue
            leaves: list = []
            stack = [eid]
            visiting: set = set()
            while stack:
                current = stack.pop()
                if current in self.complexes:
                    cdata = self.complexes[current]
                    if cdata.get("members"):
                        # OR-group of complexes: expand through members,
                        # not components. Each member may itself be a complex
                        # that needs further flattening.
                        if current in visiting:
                            continue
                        visiting.add(current)
                        stack.extend(cdata["members"])
                    else:
                        # Structural complex: flatten to leaf components
                        if current in visiting:
                            continue
                        visiting.add(current)
                        stack.extend(cdata["components"])
                else:
                    leaves.append(current)
            self._leaf_cache[eid] = leaves
            result.extend(leaves)
        return result

    # -------------------------------------------------------------------------
    # Label helpers
    # -------------------------------------------------------------------------

    def _make_label(self, name: str, entity_data: dict) -> str:
        """Make a unique node label, appending cellular location if present."""
        location = entity_data.get("cellularLocation")
        prefix = "[OR] " if entity_data.get("members") else ""
        if location and isinstance(location, dict):
            loc_name = location.get("common_name")
            if loc_name:
                return f"{prefix}{name} [{loc_name}]"
        return f"{prefix}{name}"

    def _make_label_for_id(self, entity_id: str):
        """Look up an entity ID across all stores and return its display label.
        Results are cached.
        """
        if not hasattr(self, "_label_cache"):
            self._label_cache = {}
        if entity_id in self._label_cache:
            return self._label_cache[entity_id]
        label = None
        for store in (
            self.proteins,
            self.molecules,
            self.dna,
            self.rna,
            self.physical_entity,
            self.complexes,
        ):
            if entity_id in store:
                label = self._make_label(store[entity_id]["name"], store[entity_id])
                break
        self._label_cache[entity_id] = label
        return label

    # -------------------------------------------------------------------------
    # Store helpers
    # -------------------------------------------------------------------------

    def _is_complex(self, entity_id: str) -> bool:
        return entity_id in self.complexes

    def _find_in_stores(self, entity_id: str):
        """Look up an entity across non-complex stores.

        Returns:
            (entity_data, node_type) or (None, None) if not found.
        """
        for store, node_type in [
            (self.proteins, "protein"),
            (self.molecules, "small_molecule"),
            (self.dna, "dna"),
            (self.rna, "rna"),
            (self.physical_entity, "physical_entity"),
        ]:
            if entity_id in store:
                return store[entity_id], node_type
        return None, None

    def _get_entity_name(self, entity_id: str):
        """Return the bare name of an entity without location suffix.
        Results are cached.
        """
        if not hasattr(self, "_name_cache"):
            self._name_cache = {}
        if entity_id in self._name_cache:
            return self._name_cache[entity_id]
        name = None
        for store in (
            self.proteins,
            self.molecules,
            self.dna,
            self.rna,
            self.physical_entity,
        ):
            if entity_id in store:
                name = store[entity_id]["name"]
                break
        self._name_cache[entity_id] = name
        return name

    def _get_direct_members(self, entity_id: str) -> list:
        """Return the direct member IDs of an OR-group entity, or [] if none.

        Used to build the forbidden set for reaction-partner edge generation:
        an OR node and its own direct members should not receive partner edges
        between them, because the members are alternatives for the OR node
        rather than independent co-participants in the reaction.
        """
        for store in (
            self.proteins,
            self.molecules,
            self.dna,
            self.rna,
            self.physical_entity,
            self.complexes,
        ):
            if entity_id in store:
                return store[entity_id].get("members", [])
        return []

    def _record_complex_step(self, entity_id, rank, registry):
        label = self._make_label_for_id(entity_id)
        if label is None:
            return
        existing = registry.get(label)
        if existing is None or rank < existing:
            registry[label] = rank

    def _flatten_or_members(self, entity_list: list) -> list:
        """Like _expand_single_entity but flattens all OR-groups to leaves,
        no cartesian product — just a flat union of all member IDs."""
        result = []
        seen = set()
        for eid in entity_list:
            members = self._expand_single_entity(eid)
            for mid in members:
                if mid not in seen:
                    seen.add(mid)
                    result.append(mid)
        return result

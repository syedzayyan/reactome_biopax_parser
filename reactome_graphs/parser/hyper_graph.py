"""
Hypergraph builder for Reactome BioPAX Level 3 files.

Data model (directed, member-only, reaction-centric)
----------------------------------------------------
Each biochemical reaction becomes *three* hyperedges linked by a shared
``reaction_id`` attribute:

    reactants_{rid}   — tail side of the reaction (inputs)
    products_{rid}    — head side of the reaction (outputs)
    reaction_{rid}    — logical "event" edge carrying reaction metadata;
                        its node set is the union of reactants and products
                        so it's still addressable as a single entity.

Catalysis, when present, adds a fourth edge:

    catalysis_{cid}   — directed: tail = catalyst(s), head = reactants being
                        acted on. ``control_type`` (ACTIVATION/INHIBITION)
                        is stored as an attribute.

Complexes are represented as pure hyperedges with no central "complex" node:

    complex_{cid}     — nodes are the flattened leaf members only.

Because complexes have no node, reactions that reference a complex are
rewritten in terms of its flattened leaf members before becoming
hyperedges. If a complex is never referenced by any reaction it still
appears as a standalone ``complex_*`` hyperedge so its composition is
preserved.

Directedness
------------
HyperNetX 2.x treats hyperedges as undirected sets, so directionality is
encoded by the edge *name* (``reactants_*`` vs ``products_*``, ``catalysis_*``
tail vs head) and by edge attributes. A ``direction`` attribute ("tail" or
"head") tags each directed edge, and every reaction/catalysis edge carries a
``reaction_id`` so you can join them downstream.
"""

import logging
import typing

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


class _HypergraphMixin:
    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def parse_biopax_into_hypergraph(
        self,
        filename: str,
        logger: typing.Optional[logging.Logger] = None,
    ):
        """Parse a BioPAX Level 3 file into a directed HyperNetX hypergraph.

        See the module docstring for the full data model. In short:
          * reactions become three linked hyperedges (reactants / products /
            reaction event), with direction encoded by edge name + attribute
          * catalysis becomes a single directed hyperedge (tail = catalysts,
            head = reactants controlled)
          * complexes are member-only hyperedges; the complex itself is not
            a node

        Parameters
        ----------
        filename : str
            Path to the BioPAX Level 3 (.owl) file.
        logger : logging.Logger, optional
            Logger for debug/info output.

        Returns
        -------
        hypernetx.Hypergraph
            Graph with node / edge properties populated.
        """
        try:
            import hypernetx as hnx
        except ImportError as e:
            raise ImportError(
                "HyperNetX is required for hypergraph parsing. "
                "Install it with: pip install hypernetx"
            ) from e

        if logger is not None:
            self.logger = logger

        self._log("info", f"Building HyperNetX hypergraph from BioPAX file: {filename}")
        self.parse_biopax3_file(filename)

        # Reset caches (same pattern as the NetworkX builder)
        self._label_cache = {}
        self._name_cache = {}
        self._leaf_cache = {}
        self._member_cache = {}

        hyperedges: dict = {}
        edge_attrs: dict = {}
        seen_complexes: set = set()

        # Per-reaction-type counters for readable names
        n_reaction = n_translocation = n_catalysis = n_complex = 0

        # Sort steps by (pathway, local_order) so global_rank lines up with
        # the NetworkX builder. Keeps time values consistent between the two.
        order_items = sorted(
            self.reactionOrder.items(),
            key=lambda kv: self.step_order.get(kv[0], ("~", 10**9)),
        )
        iterator = (
            tqdm(order_items, desc="Building hypergraph", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else order_items
        )

        global_rank = 0
        stepped_reactions: set = set()

        # ---- Pass 1: reactions that live inside a PathwayStep -----------
        for step_id, step_dets in iterator:
            reaction_id = step_dets[0][0]
            if reaction_id is None or reaction_id not in self.reactions:
                continue
            stepped_reactions.add(reaction_id)

            step_key = self.step_order.get(step_id, (None, None))
            counts = self._emit_reaction_hyperedges(
                reaction_id,
                step_key,
                global_rank,
                hyperedges,
                edge_attrs,
                seen_complexes,
            )
            n_reaction += counts["reaction"]
            n_translocation += counts["translocation"]
            n_catalysis += counts["catalysis"]
            n_complex += counts["complex"]
            global_rank += 1

        # ---- Pass 2: reactions not covered by any PathwayStep -----------
        unstepped = 0
        for reaction_id in self.reactions:
            if reaction_id in stepped_reactions:
                continue
            counts = self._emit_reaction_hyperedges(
                reaction_id,
                (None, None),
                global_rank,
                hyperedges,
                edge_attrs,
                seen_complexes,
            )
            n_reaction += counts["reaction"]
            n_translocation += counts["translocation"]
            n_catalysis += counts["catalysis"]
            n_complex += counts["complex"]
            global_rank += 1
            unstepped += 1

        if unstepped:
            self._log(
                "debug",
                f"Pass-2 fallback: {unstepped} reactions had no PathwayStep",
            )

        # ---- Pass 3: orphan complexes (never referenced by any reaction) -
        # Worth keeping so complex composition isn't silently lost.
        orphan_complex_count = 0
        for cid in self.complexes:
            if cid in seen_complexes:
                continue
            seen_complexes.add(cid)
            members = self._flatten_complex_members(cid)
            if not members:
                continue
            edge = f"complex_{cid}"
            hyperedges[edge] = members
            edge_attrs[edge] = {
                "type": "complex",
                "complex_id": cid,
                "members": members,
                "is_hierarchical": any(
                    self._is_complex(c) for c in self.complexes[cid]["components"]
                ),
                "orphan": True,
            }
            n_complex += 1
            orphan_complex_count += 1

        self._log(
            "info",
            f"Created {len(hyperedges)} hyperedges: "
            f"{n_reaction} reactions ({n_reaction // 3 if n_reaction else 0} triples), "
            f"{n_translocation} translocations, {n_catalysis} catalysis, "
            f"{n_complex} complexes ({orphan_complex_count} orphan)",
        )

        # ---- Relabel raw BioPAX IDs to human-readable labels -------------
        label_map = self._build_label_map()
        relabeled_edges, relabeled_attrs = self._relabel(
            hyperedges, edge_attrs, label_map
        )

        # ---- Build node attributes ---------------------------------------
        all_nodes = {n for nodes in relabeled_edges.values() for n in nodes}
        node_attrs = self._build_node_attrs(label_map, all_nodes)

        # ---- Construct Hypergraph ----------------------------------------
        self._log("debug", "Creating HyperNetX hypergraph object...")
        H = hnx.Hypergraph(relabeled_edges)
        self._attach_properties(H, node_attrs, relabeled_attrs)

        self._log(
            "info",
            f"Hypergraph complete: {len(H.nodes)} nodes, {len(H.edges)} hyperedges",
        )
        return H

    # ─────────────────────────────────────────────────────────────────────
    # Emission of a single reaction's hyperedges
    # ─────────────────────────────────────────────────────────────────────

    def _emit_reaction_hyperedges(
        self,
        reaction_id: str,
        step_key: tuple,
        global_rank: int,
        hyperedges: dict,
        edge_attrs: dict,
        seen_complexes: set,
    ) -> dict:
        """Emit the 3–4 hyperedges for one reaction and any newly-seen complexes.

        Returns a dict of counts: {'reaction', 'translocation', 'catalysis', 'complex'}.
        """
        counts = {"reaction": 0, "translocation": 0, "catalysis": 0, "complex": 0}

        left_entities, right_entities, reaction_type = self.reactions[reaction_id]
        pathway_name = self.reaction_to_pathway.get(reaction_id)
        catalysis_meta = self.catalysis_dets.get(reaction_id)
        step_pathway, local_order = step_key if step_key else (None, None)

        # Flatten complexes to leaf members because complexes have no node.
        # Keeps OR-group expansion consistent with the rest of the parser.
        def _flatten(entities: list) -> list:
            out: list = []
            for eid in entities:
                out.extend(self._flatten_to_leaves([eid]))
            # Also expand OR-groups that may survive the flatten
            expanded: list = []
            for eid in out:
                expanded.extend(self._expand_single_entity(eid))
            # dedup while preserving order
            return list(dict.fromkeys(expanded))

        expanded_left = _flatten(left_entities)
        expanded_right = _flatten(right_entities)

        base_attrs = {
            "reaction_id": reaction_id,
            "time": global_rank,
            "pathway": pathway_name,
            "local_order": local_order,
            "reaction_type": reaction_type,
        }

        # ── translocation: a single directed edge (tail ⇒ head) ───────────
        if reaction_type == "translocation":
            # For translocations reactant and product are the "same" entity in
            # different compartments, so emit as one tail edge + one head edge
            # rather than a triple, to avoid the empty middle.
            tail_name = f"translocation_tail_{reaction_id}"
            head_name = f"translocation_head_{reaction_id}"
            if expanded_left:
                hyperedges[tail_name] = expanded_left
                edge_attrs[tail_name] = {
                    **base_attrs,
                    "type": "translocation",
                    "direction": "tail",
                }
            if expanded_right:
                hyperedges[head_name] = expanded_right
                edge_attrs[head_name] = {
                    **base_attrs,
                    "type": "translocation",
                    "direction": "head",
                }
            if expanded_left or expanded_right:
                counts["translocation"] += 1

        else:
            # ── reaction triple: reactants ▸ reaction-event ▸ products ────
            if expanded_left:
                re_name = f"reactants_{reaction_id}"
                hyperedges[re_name] = expanded_left
                edge_attrs[re_name] = {
                    **base_attrs,
                    "type": "reactants",
                    "direction": "tail",
                }
            if expanded_right:
                rp_name = f"products_{reaction_id}"
                hyperedges[rp_name] = expanded_right
                edge_attrs[rp_name] = {
                    **base_attrs,
                    "type": "products",
                    "direction": "head",
                }
            # The reaction-event edge: union of tail + head, acts as the
            # logical "this whole reaction" addressable object.
            union = list(dict.fromkeys(expanded_left + expanded_right))
            if union:
                rxn_name = f"reaction_{reaction_id}"
                hyperedges[rxn_name] = union
                edge_attrs[rxn_name] = {
                    **base_attrs,
                    "type": "reaction",
                    "direction": "event",
                    "reactants": expanded_left,
                    "products": expanded_right,
                }
                counts["reaction"] += 1

            # ── catalysis: one directed hyperedge, tail = catalysts ───────
            if catalysis_meta is not None and expanded_left:
                catalyst_id = catalysis_meta["controller"]
                catalyst_nodes = list(
                    dict.fromkeys(self._expand_single_entity(catalyst_id))
                )
                if catalyst_nodes:
                    cat_tail = f"catalysis_tail_{reaction_id}"
                    cat_head = f"catalysis_head_{reaction_id}"
                    hyperedges[cat_tail] = catalyst_nodes
                    edge_attrs[cat_tail] = {
                        **base_attrs,
                        "type": "catalysis",
                        "direction": "tail",
                        "control_type": catalysis_meta["controlType"],
                    }
                    hyperedges[cat_head] = expanded_left
                    edge_attrs[cat_head] = {
                        **base_attrs,
                        "type": "catalysis",
                        "direction": "head",
                        "control_type": catalysis_meta["controlType"],
                    }
                    counts["catalysis"] += 1

        # ── complex hyperedges for any complex referenced here ────────────
        # Because we flattened above, the reaction's own node-set doesn't
        # mention complexes — but we still want a complex_* hyperedge for
        # each one so its composition is queryable.
        for eid in list(left_entities) + list(right_entities):
            for cid in self._collect_complex_ids(eid):
                if cid in seen_complexes:
                    continue
                seen_complexes.add(cid)
                members = self._flatten_complex_members(cid)
                if not members:
                    continue
                edge = f"complex_{cid}"
                hyperedges[edge] = members
                edge_attrs[edge] = {
                    "type": "complex",
                    "complex_id": cid,
                    "members": members,
                    "time": global_rank,  # earliest-seen rank
                    "pathway": pathway_name,
                    "local_order": local_order,
                    "is_hierarchical": any(
                        self._is_complex(c) for c in self.complexes[cid]["components"]
                    ),
                }
                counts["complex"] += 1

        return counts

    # ─────────────────────────────────────────────────────────────────────
    # Complex helpers
    # ─────────────────────────────────────────────────────────────────────

    def _collect_complex_ids(self, entity_id: str) -> list:
        """Return every complex ID reachable from ``entity_id`` (including itself
        if it's a complex and any nested sub-complexes)."""
        out: list = []
        stack: list = [entity_id]
        seen: set = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if self._is_complex(cur):
                out.append(cur)
                stack.extend(self.complexes[cur]["components"])
        return out

    def _flatten_complex_members(self, complex_id: str) -> list:
        """Flatten a complex to its leaf members, reusing the parser's helper
        where available. Falls back to a local BFS if the helper isn't present
        (so this mixin works even if __flatten_complex_into_protein is renamed)."""
        helper = getattr(self, "_flatten_to_leaves", None) or getattr(
            self, "_ReactomeBioPAX__flatten_complex_into_protein", None
        )
        if helper is not None:
            try:
                result = (
                    helper([complex_id])
                    if helper.__code__.co_argcount >= 2
                    else helper(complex_id)
                )
                return list(dict.fromkeys(result))
            except Exception:
                pass

        # Fallback BFS
        out: list = []
        stack: list = [complex_id]
        seen: set = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if self._is_complex(cur):
                stack.extend(self.complexes[cur]["components"])
            else:
                out.append(cur)
        return list(dict.fromkeys(out))

    # ─────────────────────────────────────────────────────────────────────
    # Label / relabel / attach
    # ─────────────────────────────────────────────────────────────────────

    def _build_label_map(self) -> dict:
        label_map: dict = {}
        for store in (
            self.proteins,
            self.molecules,
            self.physical_entity,
            self.complexes,
            self.dna,
            self.rna,
        ):
            for k, v in store.items():
                label_map[k] = self._make_label(v["name"], v)
        return label_map

    def _relabel(self, hyperedges: dict, edge_attrs: dict, label_map: dict):
        def _lst(ids):
            return [label_map.get(i, i) for i in ids]

        relabeled_edges = {e: _lst(nodes) for e, nodes in hyperedges.items()}
        relabeled_attrs: dict = {}
        for e, attrs in edge_attrs.items():
            a = dict(attrs)
            for field in ("reactants", "products", "members"):
                if field in a and isinstance(a[field], list):
                    a[field] = _lst(a[field])
            for field in ("complex_id", "reaction_id"):
                if field in a and isinstance(a[field], str):
                    a[field] = label_map.get(a[field], a[field])
            relabeled_attrs[e] = a
        return relabeled_edges, relabeled_attrs

    def _build_node_attrs(self, label_map: dict, all_nodes: set) -> dict:
        node_attrs: dict = {}
        for raw_id, label in label_map.items():
            if label not in all_nodes:
                continue
            data, node_type = self._find_in_stores(raw_id)
            if data is not None:
                node_attrs[label] = {"type": node_type, **data}
            elif raw_id in self.complexes:
                # shouldn't appear as a node under members-only semantics,
                # but keep the branch defensively
                node_attrs[label] = {"type": "complex", **self.complexes[raw_id]}
        return node_attrs

    def _attach_properties(self, H, node_attrs: dict, edge_attrs: dict):
        """Best-effort metadata attachment that works across HyperNetX versions."""
        try:
            for node, attrs in node_attrs.items():
                if node in H.nodes:
                    for k, v in attrs.items():
                        H.nodes[node].attrs[k] = v
            for edge, attrs in edge_attrs.items():
                if edge in H.edges:
                    for k, v in attrs.items():
                        H.edges[edge].attrs[k] = v
            self._log("debug", "Metadata attached via per-element attrs")
        except (AttributeError, KeyError):
            self._log(
                "warning",
                "HyperNetX attrs API unavailable — storing as _node_attributes/_edge_attributes",
            )
            H._node_attributes = node_attrs
            H._edge_attributes = edge_attrs

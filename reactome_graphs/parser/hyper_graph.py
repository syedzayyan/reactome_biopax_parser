import logging
import typing

# Conditional tqdm import
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback: tqdm acts as identity function
    def tqdm(iterable, **kwargs):
        return


class _HypergraphMixin:
    def parse_biopax_into_hypergraph(
        self, filename: str, logger: typing.Optional[logging.Logger] = None
    ):
        """Parse a BioPAX Level 3 file into a HyperNetX hypergraph.

        Constructs a hypergraph where:
        - Nodes represent molecular entities (proteins, molecules, complexes, dna, rna)
        - Reaction hyperedges connect all reactants + products in one edge
        - Translocation hyperedges connect the same entity across compartments
        - Catalysis hyperedges connect the catalyst to the entities it controls
        - Complex hyperedges connect a complex to its flattened constituent members

        Args:
            filename (str): Path to the BioPAX Level 3 file.
            logger (logging.Logger, optional): Logger instance for debug/info messages.

        Returns:
            hypernetx.Hypergraph: A hypergraph representing biochemical interactions.
        """
        try:
            import hypernetx as hnx
        except ImportError:
            raise ImportError(
                "HyperNetX is required for hypergraph parsing. "
                "Install it with: pip install hypernetx"
            )

        if logger is not None:
            self.logger = logger

        self._log("info", f"Building HyperNetX hypergraph from BioPAX file: {filename}")
        self.parse_biopax3_file(filename)

        hyperedges = {}  # edge_name  -> list of raw entity IDs
        edge_attributes = {}  # edge_name  -> dict
        seen_complexes = set()

        step = 0
        reaction_count = 0
        translocation_count = 0
        catalysis_count = 0
        complex_count = 0

        order_items = list(self.reactionOrder.items())
        iterator = (
            tqdm(order_items, desc="Building hypergraph", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else order_items
        )

        for _, step_dets in iterator:
            if step_dets[0][0] is None:
                continue

            step += 1
            reaction_id = step_dets[0][0]
            reactions = self.reactions[reaction_id]
            catalysis_exists = self.catalysis_dets.get(reaction_id)

            left_entities, right_entities, reaction_type = reactions

            # expand OR-group members so parents never appear as nodes
            expanded_left = [
                m for eid in left_entities for m in self._expand_single_entity(eid)
            ]
            expanded_right = [
                m for eid in right_entities for m in self._expand_single_entity(eid)
            ]

            # ── translocation hyperedge
            if reaction_type == "translocation":
                edge_name = f"translocation_{translocation_count}_{step}"
                nodes = list(
                    dict.fromkeys(expanded_left + expanded_right)
                )  # dedup, order-stable
                hyperedges[edge_name] = nodes
                edge_attributes[edge_name] = {
                    "type": "translocation",
                    "time": step,
                    "from": expanded_left,
                    "to": expanded_right,
                    "reaction_id": reaction_id,
                }
                translocation_count += 1

            else:
                # ── reaction hyperedge
                edge_name = f"reaction_{reaction_count}_{step}"
                nodes = list(dict.fromkeys(expanded_left + expanded_right))
                hyperedges[edge_name] = nodes
                edge_attributes[edge_name] = {
                    "type": "reaction",
                    "time": step,
                    "reactants": expanded_left,
                    "products": expanded_right,
                    "reaction_id": reaction_id,
                }
                reaction_count += 1

                # ── catalysis hyperedge
                if catalysis_exists is not None:
                    catalyst_id = catalysis_exists["controller"]
                    expanded_catalyst = self._expand_single_entity(catalyst_id)

                    cat_edge_name = f"catalysis_{catalysis_count}_{step}"
                    cat_nodes = list(dict.fromkeys(expanded_catalyst + expanded_left))
                    hyperedges[cat_edge_name] = cat_nodes
                    edge_attributes[cat_edge_name] = {
                        "type": "catalysis",
                        "control_type": catalysis_exists["controlType"],
                        "time": step,
                        "catalysts": expanded_catalyst,
                        "controlled": expanded_left,
                        "reaction_id": reaction_id,
                    }
                    catalysis_count += 1

            # ── complex hyperedges for any complex appearing in this reaction
            for entity in expanded_left + expanded_right:
                if not self._is_complex(entity) or entity in seen_complexes:
                    continue
                seen_complexes.add(entity)

                all_members = self.__flatten_complex_into_protein(entity)
                direct_components = self.complexes[entity]["components"]

                cx_edge_name = f"complex_{entity}"
                hyperedges[cx_edge_name] = [entity] + all_members
                edge_attributes[cx_edge_name] = {
                    "type": "complex",
                    "time": step,
                    "complex_center": entity,
                    "direct_components": direct_components,
                    "all_members": all_members,
                    "is_hierarchical": any(
                        self._is_complex(c) for c in direct_components
                    ),
                }
                complex_count += 1

                # sub-complex hyperedges
                for component in direct_components:
                    if not self._is_complex(component) or component in seen_complexes:
                        continue
                    seen_complexes.add(component)

                    sub_members = self.__flatten_complex_into_protein(component)
                    sub_edge_name = f"complex_{component}"
                    hyperedges[sub_edge_name] = [component] + sub_members
                    edge_attributes[sub_edge_name] = {
                        "type": "complex",
                        "time": step,
                        "complex_center": component,
                        "direct_components": self.complexes[component]["components"],
                        "all_members": sub_members,
                        "is_hierarchical": any(
                            self._is_complex(c)
                            for c in self.complexes[component]["components"]
                        ),
                        "parent_complex": entity,
                    }
                    complex_count += 1

        self._log(
            "info",
            f"Created {len(hyperedges)} hyperedges: "
            f"{reaction_count} reactions, {translocation_count} translocations, "
            f"{catalysis_count} catalysis, {complex_count} complexes",
        )

        # ── build label map using _make_label (location-aware, same as networkx builder)
        label_map = {}
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

        # ── relabel hyperedges
        relabeled_hyperedges = {}
        relabeled_edge_attrs = {}

        def _relabel_list(ids: list) -> list:
            return [label_map.get(i, i) for i in ids]

        for edge_name, nodes in hyperedges.items():
            relabeled_hyperedges[edge_name] = _relabel_list(nodes)

            attrs = edge_attributes[edge_name].copy()
            for field in (
                "from",
                "to",
                "reactants",
                "products",
                "catalysts",
                "controlled",
                "direct_components",
                "all_members",
                "parent_complex",
            ):
                if field in attrs:
                    val = attrs[field]
                    attrs[field] = (
                        _relabel_list(val)
                        if isinstance(val, list)
                        else label_map.get(val, val)
                    )
            if "complex_center" in attrs:
                attrs["complex_center"] = label_map.get(
                    attrs["complex_center"], attrs["complex_center"]
                )

            relabeled_edge_attrs[edge_name] = attrs

        # ── build node attributes from stores
        all_nodes = {n for nodes in relabeled_hyperedges.values() for n in nodes}
        node_attributes = {}

        for raw_id, label in label_map.items():
            if label not in all_nodes:
                continue
            data, node_type = self._find_in_stores(raw_id)
            if data is not None:
                node_attributes[label] = {"type": node_type, **data}
            elif raw_id in self.complexes:
                node_attributes[label] = {"type": "complex", **self.complexes[raw_id]}

        # ── construct hypergraph
        self._log("debug", "Creating HyperNetX hypergraph object...")
        H = hnx.Hypergraph(relabeled_hyperedges)

        try:
            for node, attrs in node_attributes.items():
                if node in H.nodes:
                    for k, v in attrs.items():
                        H.nodes[node].attrs[k] = v
            for edge, attrs in relabeled_edge_attrs.items():
                if edge in H.edges:
                    for k, v in attrs.items():
                        H.edges[edge].attrs[k] = v
            self._log("debug", "Metadata attached successfully")
        except AttributeError:
            self._log(
                "warning",
                "Attribute setting not supported in this HyperNetX version — storing as _attributes",
            )
            H._node_attributes = node_attributes
            H._edge_attributes = relabeled_edge_attrs

        self._log(
            "info",
            f"Hypergraph complete: {len(H.nodes)} nodes, {len(H.edges)} hyperedges",
        )
        return H

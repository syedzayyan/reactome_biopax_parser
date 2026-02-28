import logging
import typing

import networkx as nx

# Conditional tqdm import
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback: tqdm acts as identity function
    def tqdm(iterable, **kwargs):
        return iterable


class _NxGraphMixin:
    def parse_biopax_into_networkx(
        self,
        filename: str,
        reaction_partners: bool = False,
        flatten_complexes: bool = False,
        logger: typing.Optional[logging.Logger] = None,
    ):
        """Parse a BioPAX Level 3 file into a directed NetworkX graph.

        Constructs a directed reaction network where nodes represent molecular
        entities and edges represent biochemical reactions or complex relationships.
        Proteins with member entities (isoforms/variants) are expanded so that each
        reaction is duplicated for each member alternative.

        Args:
            filename (str): Path to the BioPAX Level 3 file.
            complex_proteins_as_node (bool): If True, expand complex proteins as separate nodes.
            logger (logging.Logger, optional): Logger instance for debug/info messages.
                If None, uses the instance logger if available.

        Returns:
            networkx.DiGraph: A directed graph representing biochemical interactions.
        """
        if logger is not None:
            self.logger = logger

        self._log("info", f"Building NetworkX graph from BioPAX file: {filename}")

        self.parse_biopax3_file(filename)
        G = nx.DiGraph()
        step = 0
        reaction_count = 0

        order_items = list(self.reactionOrder.items())
        iterator = (
            tqdm(order_items, desc="Building graph", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else order_items
        )

        for _, step_dets in iterator:
            if step_dets[0][0] is not None:
                step += 1
                reactions = self.reactions[step_dets[0][0]]
                pathway_name = self.reaction_to_pathway.get(step_dets[0][0], None)
                catalysis_exists = self.catalysis_dets.get(step_dets[0][0], None)

                # For transport reactions (same entities, different locations),
                # skip combinatorial expansion to avoid explosion
                left_entities, right_entities, reaction_type = reactions

                # Translocation reactions are already 1-to-1 paired — no expansion needed
                # In parse_biopax_into_networkx, translocation branch:
                if reaction_type == "translocation":
                    for left, right in zip(left_entities, right_entities):
                        if (
                            left in self.physical_entity
                            or right in self.physical_entity
                        ):
                            continue
                        # Expand members but zip (not product) — they're already name-matched 1-to-1
                        left_expansions = self._expand_single_entity(left)
                        right_expansions = self._expand_single_entity(right)
                        for l, r in zip(left_expansions, right_expansions):
                            left_label = self._make_label_for_id(l)
                            right_label = self._make_label_for_id(r)
                            if left_label and right_label and left_label != right_label:
                                G.add_node(l)
                                G.add_node(r)
                                G.add_edge(
                                    l,
                                    r,
                                    time=step,
                                    type="translocation",
                                    pathway=pathway_name,
                                )
                    continue

                # Normal path — member expansion + cartesian product
                expanded_left_combinations = self._expand_members(reactions[0])
                expanded_right_combinations = self._expand_members(reactions[1])

                # Expand catalyst if it has member alternatives
                expanded_catalysts = []
                if catalysis_exists is not None:
                    catalyst_id = catalysis_exists["controller"]
                    if self._is_complex(catalyst_id) and flatten_complexes:
                        catalyst_members = self.__flatten_complex_into_protein(
                            catalyst_id
                        )
                        for cm in catalyst_members:
                            G.add_node(cm, complex_component=True, color="violet")
                        expanded_catalysts = catalyst_members
                    else:
                        expanded_catalysts = self._expand_single_entity(catalyst_id)
                else:
                    expanded_catalysts = [None]

                # Create cartesian product of all combinations
                for catalyst in expanded_catalysts:
                    for left_combo in expanded_left_combinations:
                        for right_combo in expanded_right_combinations:
                            # Add catalyst node if present
                            if catalyst is not None:
                                G.add_node(catalyst)
                                self._log(
                                    "debug",
                                    f"Step {step}: Added catalyst node '{catalyst}'",
                                )

                            for left in left_combo:
                                if self._is_complex(left) and flatten_complexes:
                                    left_members = self.__flatten_complex_into_protein(
                                        left
                                    )
                                    for m in left_members:
                                        G.add_node(
                                            m, complex_component=True, color="violet"
                                        )
                                    if catalyst is not None:
                                        for m in left_members:
                                            G.add_edge(
                                                catalyst,
                                                m,
                                                time=step,
                                                type="catalysis",
                                                catalysis_type=catalysis_exists[
                                                    "controlType"
                                                ],
                                                pathway=pathway_name,
                                            )

                                    for right in right_combo:
                                        if (
                                            self._is_complex(right)
                                            and flatten_complexes
                                        ):
                                            right_members = (
                                                self.__flatten_complex_into_protein(
                                                    right
                                                )
                                            )
                                            for rm in right_members:
                                                G.add_node(
                                                    rm,
                                                    complex_component=True,
                                                    color="violet",
                                                )
                                            for lm in left_members:
                                                for rm in right_members:
                                                    G.add_edge(
                                                        lm,
                                                        rm,
                                                        time=step,
                                                        type="reaction",
                                                        pathway=pathway_name,
                                                    )
                                        else:
                                            G.add_node(right)
                                            for lm in left_members:
                                                G.add_edge(
                                                    lm,
                                                    right,
                                                    time=step,
                                                    type="reaction",
                                                    pathway=pathway_name,
                                                )
                                else:
                                    G.add_node(left)
                                    if catalyst is not None:
                                        G.add_edge(
                                            catalyst,
                                            left,
                                            time=step,
                                            type="catalysis",
                                            catalysis_type=catalysis_exists[
                                                "controlType"
                                            ],
                                            pathway=pathway_name,
                                        )
                                    for right in right_combo:
                                        if (
                                            self._is_complex(right)
                                            and flatten_complexes
                                        ):
                                            right_members = (
                                                self.__flatten_complex_into_protein(
                                                    right
                                                )
                                            )
                                            for rm in right_members:
                                                G.add_node(
                                                    rm,
                                                    complex_component=True,
                                                    color="violet",
                                                )
                                                G.add_edge(
                                                    left,
                                                    rm,
                                                    time=step,
                                                    type="reaction",
                                                    pathway=pathway_name,
                                                )
                                        else:
                                            G.add_node(right)
                                            G.add_edge(
                                                left,
                                                right,
                                                time=step,
                                                type="reaction",
                                                pathway=pathway_name,
                                            )
                            # Peer edges between complex members
                            for entity in left_combo + right_combo:
                                if self._is_complex(entity):
                                    self._add_complex_edges(
                                        G, entity, step, pathway_name, flatten_complexes
                                    )
                        if reaction_partners:
                            for i, l1 in enumerate(left_combo):
                                for l2 in left_combo[i + 1 :]:
                                    G.add_edge(
                                        l1,
                                        l2,
                                        time=step,
                                        type="reaction_partner",
                                        pathway=pathway_name,
                                    )
                                    G.add_edge(
                                        l2,
                                        l1,
                                        time=step,
                                        type="reaction_partner",
                                        pathway=pathway_name,
                                    )

        self._log(
            "info",
            f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges over {step} steps ({reaction_count} reactions)",
        )

        # Node type annotations
        self._log("debug", "Annotating node types...")
        for node in G.nodes:
            if node in self.proteins:
                protein_data = self.proteins[node]
                if protein_data.get("members"):
                    continue
                G.nodes[node].update({"type": "protein", **self.proteins[node]})
            elif node in self.molecules:
                G.nodes[node].update({"type": "small_molecule", **self.molecules[node]})
            elif node in self.physical_entity:
                G.nodes[node].update(
                    {"type": "physical_entity", **self.physical_entity[node]}
                )
            elif node in self.complexes:
                G.nodes[node].update({"type": "complex", **self.complexes[node]})
            elif node in self.dna:
                G.nodes[node].update({"type": "dna", **self.dna[node]})
            elif node in self.rna:
                G.nodes[node].update({"type": "rna", **self.rna[node]})

        # Relabeling mappings
        self._log("debug", "Relabeling nodes with display names...")

        # Replace the existing relabeling maps in parse_biopax_into_networkx:
        protein_map = {
            k: self._make_label(v["name"], v) for k, v in self.proteins.items()
        }
        molecules_map = {
            k: self._make_label(v["name"], v) for k, v in self.molecules.items()
        }
        physical_entity_map = {
            k: self._make_label(v["name"], v) for k, v in self.physical_entity.items()
        }
        complexes_map = {
            k: self._make_label(v["name"], v) for k, v in self.complexes.items()
        }
        dna_map = {k: self._make_label(v["name"], v) for k, v in self.dna.items()}
        rna_map = {k: self._make_label(v["name"], v) for k, v in self.rna.items()}

        G = nx.relabel_nodes(
            G,
            {
                **protein_map,
                **molecules_map,
                **physical_entity_map,
                **complexes_map,
                **dna_map,
                **rna_map,
            },
        )

        self._log("debug", "Removing self-loops...")
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)

        self._add_complex_component_nodes(G)
        self._add_member_nodes(G)

        # Second annotation pass — catches nodes added after relabeling
        label_to_annotation = {}
        for store, node_type in [
            (self.proteins, "protein"),
            (self.molecules, "small_molecule"),
            (self.dna, "dna"),
            (self.rna, "rna"),
            (self.physical_entity, "physical_entity"),
            (self.complexes, "complex"),
        ]:
            for entity_id, entity_data in store.items():
                if entity_data.get("members"):
                    continue
                label = self._make_label(entity_data["name"], entity_data)
                label_to_annotation[label] = (node_type, entity_data)

        for node in G.nodes:
            if not G.nodes[node].get("type") and node in label_to_annotation:
                node_type, entity_data = label_to_annotation[node]
                G.nodes[node].update({"type": node_type, **entity_data})
            elif not G.nodes[node].get("type"):
                G.nodes[node]["type"] = "other"

        self._log("info", "Graph construction complete")
        return G

    def _expand_single_entity(self, entity_id: str) -> list:
        for store in (
            self.proteins,
            self.molecules,
            self.dna,
            self.rna,
            self.physical_entity,
        ):
            if entity_id in store:
                members = store[entity_id].get("members", [])
                if members:
                    return members
        return [entity_id]

    def _expand_members(self, entity_list: list) -> list:
        """Expand a list of entities, handling member alternatives.

        For entities with memberPhysicalEntity (OR alternatives), this creates
        all possible combinations by expanding each entity independently.

        Args:
            entity_list (list): List of entity IDs to expand.

        Returns:
            list of lists: All possible combinations of expanded entities.
        """
        if not entity_list:
            return [[]]

        # Expand each entity in the list
        expanded_per_entity = []
        for entity_id in entity_list:
            expanded = self._expand_single_entity(entity_id)
            expanded_per_entity.append(expanded)

        # Generate cartesian product of all alternatives
        import itertools

        combinations = list(itertools.product(*expanded_per_entity))

        # Convert tuples to lists
        return [list(combo) for combo in combinations]

    def _make_label(self, name, entity_data):
        """Make a unique node label by appending cellular location if present."""
        location = entity_data.get("cellularLocation")
        prefix = "[OR] " if entity_data.get("members") else ""
        if location and isinstance(location, dict):
            loc_name = location.get("common_name")
            if loc_name:
                return f"{prefix}{name} [{loc_name}]"
        return f"{prefix}{name}"

    def _is_complex(self, entity_id: str) -> bool:
        """Check against parsed data, not string matching."""
        return entity_id in self.complexes

    def _find_in_stores(self, entity_id: str):
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

    def _add_member_nodes(self, G: nx.DiGraph):
        """
        Dissolve OR-group parent nodes into their individual members.
        All edges (including translocation) are migrated to each member,
        then the parent is removed. No or_* nodes survive.
        """
        stores = [
            (self.proteins, "protein"),
            (self.molecules, "small_molecule"),
            (self.dna, "dna"),
            (self.rna, "rna"),
            (self.physical_entity, "physical_entity"),
        ]
        for store, node_type in stores:
            for entity_id, entity_data in store.items():
                members = entity_data.get("members", [])
                if not members:
                    continue

                parent_label = self._make_label(entity_data["name"], entity_data)
                if parent_label not in G.nodes:
                    continue

                out_edges = list(G.out_edges(parent_label, data=True))
                in_edges = list(G.in_edges(parent_label, data=True))

                for member_id in members:
                    member_data, member_type = self._find_in_stores(member_id)
                    if member_data is None:
                        continue
                    member_label = self._make_label(member_data["name"], member_data)

                    if member_label not in G.nodes:
                        G.add_node(member_label)
                        G.nodes[member_label].update(
                            {"type": member_type, **member_data}
                        )

                    for _, neighbor, edge_data in out_edges:
                        if neighbor != member_label:  # avoid self-loops
                            G.add_edge(member_label, neighbor, **edge_data)
                    for neighbor, _, edge_data in in_edges:
                        if neighbor != member_label:
                            G.add_edge(neighbor, member_label, **edge_data)

                G.remove_node(parent_label)

    def _add_complex_component_nodes(self, G: nx.DiGraph):
        """Ensure all components of complexes already in the graph are themselves nodes."""

        # Build reverse map: label → (entity_id, store, node_type)
        label_to_id = {}
        for store, node_type in [
            (self.proteins, "protein"),
            (self.molecules, "small_molecule"),
            (self.dna, "dna"),
            (self.rna, "rna"),
            (self.physical_entity, "physical_entity"),
        ]:
            for entity_id, entity_data in store.items():
                label = self._make_label(entity_data["name"], entity_data)
                label_to_id[label] = (entity_id, store, node_type)

        added = 0
        # Iterate over snapshot of nodes (graph may grow)
        for node in list(G.nodes):
            data = G.nodes[node]
            if data.get("type") not in ("complex", "Complex"):
                continue
            # Find the complex id by matching label
            for _, cdata in self.complexes.items():
                if self._make_label(cdata["name"], cdata) != node:
                    continue
                for component_id in cdata["components"]:
                    label = self._make_label_for_id(component_id)
                    if label and label not in G.nodes:
                        cmp_data, cmp_type = self._find_in_stores(component_id)
                        if cmp_data:
                            G.add_node(label)
                            G.nodes[label].update({"type": cmp_type, **cmp_data})
                            added += 1
        or_survivors = [n for n in G.nodes if str(n).startswith("[OR]")]
        G.remove_nodes_from(or_survivors)
        self._log(
            "debug", f"Added {added} complex component nodes (DNA/RNA/protein/molecule)"
        )

    def __flatten_complex_into_protein(self, entity):
        proteins = []
        if entity in self.complexes:
            components = self.complexes[entity]["components"]
            for component_id in components:
                proteins.extend(self.__flatten_complex_into_protein(component_id))
        else:
            # Expand OR-groups instead of returning the parent
            expanded = self._expand_single_entity(entity)
            if len(expanded) == 1 and expanded[0] == entity:
                return [entity]  # not an OR-group
            for member_id in expanded:
                proteins.extend(self.__flatten_complex_into_protein(member_id))
        return proteins

    def _add_complex_edges(
        self, G, complex_node, step, pathway_name, flatten_complexes
    ):
        members = self.__flatten_complex_into_protein(complex_node)

        if flatten_complexes:
            for protein in members:
                G.add_node(protein, complex_component=True, color="violet")
            for i, p1 in enumerate(members):
                for p2 in members[i + 1 :]:
                    G.add_edge(p1, p2, time=step, type="complex", pathway=pathway_name)
                    G.add_edge(p2, p1, time=step, type="complex", pathway=pathway_name)
        else:
            for component in members:
                G.add_node(component, complex_component=True, color="violet")
                G.add_edge(component, complex_node, type="complex_component", time=step)
            G.nodes[complex_node].update(
                {"type": "Complex", **self.complexes[complex_node]}
            )

    def _make_label_for_id(self, entity_id: str) -> str | None:
        """Look up an entity ID across all stores and return its display label."""
        for store in (
            self.proteins,
            self.molecules,
            self.dna,
            self.rna,
            self.physical_entity,
        ):
            if entity_id in store:
                return self._make_label(store[entity_id]["name"], store[entity_id])
        return None

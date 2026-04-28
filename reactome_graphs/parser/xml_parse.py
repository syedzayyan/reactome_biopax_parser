import logging
import re
import typing
import xml.etree.ElementTree as ET

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


namespaces = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xml_schema": "http://www.w3.org/2001/XMLSchema#",
    "biopax": "http://www.biopax.org/release/biopax-level3.owl#",
}

ID_RDF_STRING = "{%s}ID" % namespaces["rdf"]
RESOUCE_RDF_STRING = "{%s}resource" % namespaces["rdf"]


class _ParserBase:
    def __init__(
        self,
        uniprot_accession_num=False,
        logger: typing.Optional[logging.Logger] = None,
    ):
        self.uniprot_accession_num = uniprot_accession_num
        self.logger = logger

    def _log(self, level: str, message: str):
        """Internal logging helper that only logs if logger is available."""
        if self.logger is not None:
            getattr(self.logger, level)(message)

    def parse_biopax3_file(self, filename: str):
        r"""Parse a BioPAX Level 3 XML file and populate internal data structures.

        Args:
            filename (str): Path to the BioPAX Level 3 file.
        """
        self._log("info", f"Starting parse of BioPAX file: {filename}")
        self.tree = ET.parse(filename)
        self._log("debug", "XML tree loaded successfully")

        self._log("debug", "Parsing cellular locations...")
        self.__parse_unixrefs()
        self.__parse_cellular_location()

        self._log("debug", "Parsing Proteins...")
        self.proteinRefs = self.__parse_entity_refs("ProteinReference")
        self.dnaRefs = self.__parse_entity_refs("DnaReference")
        self.rnaRefs = self.__parse_entity_refs("RnaReference")
        self.small_mol_refs = self.__parse_entity_refs("SmallMoleculeReference")
        self.proteins = self.__parse_physical_entities(
            "Protein", color="red", ref_store=self.proteinRefs, desc="Parsing proteins"
        )
        self.dna = self.__parse_physical_entities(
            "Dna", color="yellow", ref_store=self.dnaRefs, desc="Parsing DNA"
        )
        self.rna = self.__parse_physical_entities(
            "Rna", color="orange", ref_store=self.rnaRefs, desc="Parsing RNA"
        )

        self._log("debug", "Parsing molecules...")
        self.__parse_molecules()

        self._log("debug", "Parsing physical entities...")
        self.__physical_entity()

        self._log("debug", "Parsing catalysis...")
        self.__parse_catalysis()

        self._log("debug", "Parsing protein complexes...")
        self.__parse_protein_complexes()

        self._log("debug", "Parsing reactions...")
        self.__parse_reactions()
        self.__parse_pathway_memberships()

        self._log("debug", "Parsing reaction order...")
        self.__parse_reaction_order()

        self._log("debug", "Parsing stoichiometry...")
        self.__parse_stoichometry()

        self._log("info", "BioPAX file parsing complete")

    def __parse_pathway_memberships(self):
        """Map each reaction ID to its most-specific (deepest) parent pathway name.

        In Reactome BioPAX files a large pathway like 'Signalling by GPCR'
        contains many sub-pathways ('GPER1 signaling', 'Beta2 adrenergic', …),
        each owning a distinct set of reactions.  A reaction that is a direct
        component of a sub-pathway should carry *that* sub-pathway name on its
        edges, not the root name — otherwise every edge in the whole GPCR
        super-pathway collapses into a single label.

        Strategy
        --------
        1. Collect every pathway's display name and direct component list.
        2. Compute each pathway's depth (distance from a root pathway).
        3. For each reaction, among all pathways that claim it as a direct
           component, pick the one with the greatest depth — i.e. the most
           specific enclosing context.  Ties are broken arbitrarily (both
           pathways are at equal specificity).
        """
        # --- pass 1: collect names, direct component lists, parent sets ---
        pathway_names: dict[str, str] = {}  # pid -> display name
        id_to_components: dict[str, list] = {}  # pid -> [component ids]
        raw_parents: dict[str, list] = {}  # component_id -> [parent pids]
        id_to_steps: dict[str, int] = {}  # component_id -> [parent pids]

        for pathway in self.tree.findall("biopax:Pathway", namespaces):
            step_ids = [
                s.get(RESOUCE_RDF_STRING, "").strip("#")
                for s in pathway.findall("biopax:pathwayOrder", namespaces)
            ]
            pid = pathway.get(ID_RDF_STRING)

            id_to_steps[pid] = step_ids

            name_el = pathway.find("biopax:displayName", namespaces)
            pathway_names[pid] = name_el.text.strip() if name_el is not None else pid
            components = []
            for component in pathway.findall("biopax:pathwayComponent", namespaces):
                cid = component.get(RESOUCE_RDF_STRING, "").strip("#")
                components.append(cid)
                raw_parents.setdefault(cid, []).append(pid)
            id_to_components[pid] = components

        all_pathway_ids = set(pathway_names)

        # --- pass 2: compute depth of each pathway (memoised) ---
        depth_memo: dict[str, int] = {}

        def _depth(pid: str) -> int:
            if pid in depth_memo:
                return depth_memo[pid]
            pathway_parents = [
                p for p in raw_parents.get(pid, []) if p in all_pathway_ids
            ]
            d = (
                0
                if not pathway_parents
                else 1 + max(_depth(p) for p in pathway_parents)
            )
            depth_memo[pid] = d
            return d

        for pid in all_pathway_ids:
            _depth(pid)

        # --- pass 3: assign each reaction to its deepest (most specific) parent ---
        self.reaction_to_pathway = {}
        for reaction_id in self.reactions:
            parents = [
                p for p in raw_parents.get(reaction_id, []) if p in all_pathway_ids
            ]
            if not parents:
                continue
            deepest = max(parents, key=_depth)
            self.reaction_to_pathway[reaction_id] = pathway_names[deepest]

        self.pathway_to_steps = id_to_steps  # pid -> [stepID, ...]
        self.step_to_pathway = {
            sid: pid for pid, sids in id_to_steps.items() for sid in sids
        }
        self.pathway_names = pathway_names  # expose for downstream use

        self._log(
            "debug",
            f"Mapped {len(self.reaction_to_pathway)} reactions to their most-specific pathway",
        )

    def __parse_entity_refs(self, biopax_type: str) -> dict:
        refs = {}
        for ref in self.tree.findall(f"biopax:{biopax_type}", namespaces):
            ref_id = ref.get(ID_RDF_STRING, None)
            name_el = ref.find("biopax:name", namespaces)
            if name_el is None or name_el.text is None:
                refs[ref_id] = None
                continue
            text = name_el.text.strip()

            # Format 1: "name [DB:ID]" e.g. "ATP(4-) [ChEBI:30616]"
            bracket_match = re.search(r"\[([^:]+):(\S+)\]", text)
            if bracket_match:
                refs[ref_id] = f"{bracket_match.group(1)}:{bracket_match.group(2)}"
                continue

            # Format 2: "DB:ID rest" e.g. "UniProt:P12345"
            colon_match = re.search(r"^(.+?):(\S+)", text)
            if colon_match:
                refs[ref_id] = f"{colon_match.group(1)}:{colon_match.group(2)}"
                continue

            refs[ref_id] = text
        return refs

    def __parse_physical_entities(
        self,
        biopax_type: str,
        color: str,
        ref_store: dict | None = None,
        desc: str = "",
    ) -> dict:
        """
        Generic parser for Protein, Dna, Rna physical entities.

        Args:
            biopax_type: BioPAX element tag (e.g. 'Protein', 'Dna', 'Rna')
            color: Node color string
            ref_store: dict mapping refID → identifier (proteinRefs, dnaRefs, etc.)
                    Pass None to skip ref resolution.
            desc: tqdm description label
        """
        entity_list = list(self.tree.findall(f"biopax:{biopax_type}", namespaces))
        self._log("debug", f"Found {len(entity_list)} {biopax_type} entities")

        results = {}
        iterator = (
            tqdm(entity_list, desc=desc, disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else entity_list
        )

        for entity in iterator:
            eid = entity.get(ID_RDF_STRING, None)

            dis_name_el = entity.find("biopax:displayName", namespaces)
            dis_name = dis_name_el.text if dis_name_el is not None else None
            if dis_name is None:
                name_el = entity.find("biopax:name", namespaces)
                dis_name = name_el.text if name_el is not None else eid

            aliases = [nm.text for nm in entity.findall("biopax:name", namespaces)]

            reactome_db_id = None
            for c in entity.findall("biopax:comment", namespaces):
                if c.text and c.text.startswith("Reactome DB_ID:"):
                    reactome_db_id = c.text.replace("Reactome DB_ID:", "").strip()
                    break

            loc_el = entity.find("biopax:cellularLocation", namespaces)
            cellular_location = (
                self.cellularLocations.get(loc_el.get(RESOUCE_RDF_STRING).strip("#"))
                if loc_el is not None
                else None
            )

            xref = [
                self.uniXrefs.get(
                    xre.get(RESOUCE_RDF_STRING, None).strip("#"),
                    xre.get(RESOUCE_RDF_STRING, None).strip("#"),
                )
                for xre in entity.findall("biopax:xref", namespaces)
            ]

            ref_id = None
            if ref_store is not None:
                entity_ref_el = entity.find("biopax:entityReference", namespaces)
                if entity_ref_el is not None:
                    ref_key = entity_ref_el.get(RESOUCE_RDF_STRING, None).strip("#")
                    ref_id = ref_store.get(ref_key, None)

            record = {
                "name": dis_name,
                "aliases": aliases,
                "reactome_db_id": reactome_db_id,
                "cellularLocation": cellular_location,
                "ref_id": ref_id,
                "xref": xref,
                "color": color,
            }

            members = [
                m.get(RESOUCE_RDF_STRING, None).strip("#")
                for m in entity.findall("biopax:memberPhysicalEntity", namespaces)
            ]
            record["members"] = members
            # Protein-specific extras
            if biopax_type == "Protein":
                record["ref_id"] = None  # proteins use uniprot_id instead
                if self.uniprot_accession_num:
                    record["uniprot_id"] = ref_id

            results[eid] = record

        self._log("info", f"Parsed {len(results)} {biopax_type} entities")
        return results

    def __parse_molecules(self):
        r"""Parse small molecule entities from the BioPAX file.

        Extracts display names, aliases, and comments for each molecule
        and stores them in a dictionary.
        """
        molecules_list = list(self.tree.findall("biopax:SmallMolecule", namespaces))
        self._log("debug", f"Found {len(molecules_list)} small molecules")

        self.molecules = {}
        iterator = (
            tqdm(molecules_list, desc="Parsing molecules", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else molecules_list
        )

        for compounds in iterator:
            moleculesID = compounds.get(ID_RDF_STRING, None)
            disName = compounds.find("biopax:displayName", namespaces).text
            aliases = [nm.text for nm in compounds.findall("biopax:name", namespaces)]
            comment = compounds.find("biopax:comment", namespaces).text
            entityRef = compounds.find("biopax:entityReference", namespaces)

            ref_id = None
            if entityRef is not None:
                ref_key = entityRef.get(RESOUCE_RDF_STRING, None).strip("#")
                ref_id = self.small_mol_refs.get(ref_key, None)

            cellularLocation = (
                compounds.find("biopax:cellularLocation", namespaces)
                .get(RESOUCE_RDF_STRING, None)
                .strip("#")
            )
            memberORs = compounds.findall("biopax:memberPhysicalEntity", namespaces)
            members = []
            if len(memberORs) > 0:
                for member in memberORs:
                    members.append(member.get(RESOUCE_RDF_STRING, None).strip("#"))

            self.molecules[moleculesID] = {
                "name": disName,
                "aliases": aliases,
                "comment": comment,
                "color": "green",
                "cellularLocation": self.cellularLocations.get(cellularLocation),
                "entityRef": ref_id,
                "members": members,
            }

        self._log("info", f"Parsed {len(self.molecules)} small molecules")

    def __parse_cellular_location(self):
        r"""Parse cellular location vocabularies from the BioPAX file.

        Extracts terms and cross-references for cellular locations
        and stores them in a dictionary.
        """
        locations_list = list(
            self.tree.findall("biopax:CellularLocationVocabulary", namespaces)
        )
        self._log("debug", f"Found {len(locations_list)} cellular locations")

        self.cellularLocations = {}
        iterator = (
            tqdm(
                locations_list,
                desc="Parsing cellular locations",
                disable=not TQDM_AVAILABLE,
            )
            if TQDM_AVAILABLE
            else locations_list
        )

        for locations in iterator:
            term = locations.find("biopax:term", namespaces).text
            xref = self.uniXrefs[
                locations.find("biopax:xref", namespaces)
                .get(RESOUCE_RDF_STRING, None)
                .strip("#")
            ]
            self.cellularLocations[locations.get(ID_RDF_STRING, None)] = {
                "common_name": term,
                **xref,
            }

        self._log("info", f"Parsed {len(self.cellularLocations)} cellular locations")

    def __parse_catalysis(self):
        catalysis_list = list(self.tree.findall("biopax:Catalysis", namespaces))
        self._log("debug", f"Found {len(catalysis_list)} catalysis reactions")

        self.catalysis_dets = {}
        iterator = (
            tqdm(catalysis_list, desc="Parsing catalysis", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else catalysis_list
        )

        for catalysis in iterator:
            ct_el = catalysis.find("biopax:controlType", namespaces)
            controlType = ct_el.text.strip("#") if ct_el is not None else "ACTIVATION"

            controller = (
                catalysis.find("biopax:controller", namespaces)
                .get(RESOUCE_RDF_STRING, None)
                .strip("#")
            )
            controlled = (
                catalysis.find("biopax:controlled", namespaces)
                .get(RESOUCE_RDF_STRING, None)
                .strip("#")
            )

            uniXref = [
                xref.get(RESOUCE_RDF_STRING, None)
                for xref in catalysis.findall("biopax:xref", namespaces)
            ]
            self.catalysis_dets[controlled] = {
                "controlType": controlType,
                "controller": controller,
                "uniXref": uniXref,
            }

        self._log("info", f"Parsed {len(self.catalysis_dets)} catalysis reactions")

    def __parse_protein_complexes(self):
        r"""Parse protein complex entities from the BioPAX file.

        Extracts components, stoichiometries, cellular locations,
        cross-references, and Reactome DB IDs for complexes.

        Returns:
            dict: A dictionary mapping complex IDs to their associated metadata.
        """
        complexes_list = list(self.tree.findall("biopax:Complex", namespaces))
        self._log("debug", f"Found {len(complexes_list)} protein complexes")

        self.complexes = {}
        iterator = (
            tqdm(complexes_list, desc="Parsing complexes", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else complexes_list
        )

        for complex in iterator:
            complexId = complex.get(ID_RDF_STRING, None)
            disName = complex.find("biopax:displayName", namespaces).text
            if disName is None:
                disName = complex.find("biopax:name", namespaces).text

            component = [
                cmop.get(RESOUCE_RDF_STRING, None).strip("#")
                for cmop in complex.findall("biopax:component", namespaces)
            ]
            componentStoichiometry = [
                compStoicho.attrib.get(RESOUCE_RDF_STRING, None).strip("#")
                for compStoicho in complex.findall(
                    "biopax:componentStoichiometry", namespaces
                )
            ]

            cellularLocation = (
                complex.find("biopax:cellularLocation", namespaces)
                .get(RESOUCE_RDF_STRING, None)
                .strip("#")
            )
            xref = [
                self.uniXrefs.get(
                    xre.get(RESOUCE_RDF_STRING, None).strip("#"),
                    xre.get(RESOUCE_RDF_STRING, None).strip("#"),
                )
                for xre in complex.findall("biopax:xref", namespaces)
            ]
            dataSource = complex.find("biopax:dataSource", namespaces).text

            comments = complex.findall("biopax:comment", namespaces)

            reactome_db_id = None

            members = [
                m.get(RESOUCE_RDF_STRING, None).strip("#")
                for m in complex.findall("biopax:memberPhysicalEntity", namespaces)
            ]

            for c in comments:
                if c.text and c.text.startswith("Reactome DB_ID:"):
                    reactome_db_id = c.text.replace("Reactome DB_ID:", "").strip()
                    break

            self.complexes[complexId] = {
                "name": disName,
                "components": component,
                "members": members,
                "reactome_db_id": reactome_db_id,
                "cellularLocation": self.cellularLocations.get(cellularLocation),
                "xref": xref,
                "componentStoichiometry": componentStoichiometry,
                "dataSource": dataSource,
                "complex_node": True,
            }

        self._log("info", f"Parsed {len(self.complexes)} protein complexes")

    def __physical_entity(self):
        """Parse physical entities from the BioPAX file.

        Extracts display names, cellular locations, and cross-references
        for physical entities.
        """
        entities_list = list(self.tree.findall("biopax:PhysicalEntity", namespaces))
        self._log("debug", f"Found {len(entities_list)} physical entities")

        self.physical_entity = {}
        iterator = (
            tqdm(
                entities_list,
                desc="Parsing physical entities",
                disable=not TQDM_AVAILABLE,
            )
            if TQDM_AVAILABLE
            else entities_list
        )

        for physEntity in iterator:
            ID = physEntity.get("{%s}ID" % namespaces["rdf"], None)
            disName = physEntity.find("biopax:displayName", namespaces).text
            cellularLocation = (
                physEntity.find("biopax:cellularLocation", namespaces)
                .get(RESOUCE_RDF_STRING, None)
                .strip("#")
            )
            xref = [
                xre.get(RESOUCE_RDF_STRING, None).strip("#")
                for xre in physEntity.findall("biopax:xref", namespaces)
            ]
            members = [
                m.get(RESOUCE_RDF_STRING).strip("#")
                for m in physEntity.findall("biopax:memberPhysicalEntity", namespaces)
            ]
            self.physical_entity[ID] = {
                "name": disName,
                "cellularLocation": self.cellularLocations.get(cellularLocation),
                "xref": xref,
                "members": members,
            }

        self._log("info", f"Parsed {len(self.physical_entity)} physical entities")

    def __parse_stoichometry(self):
        """Parse stoichiometric coefficients for physical entities.

        Extracts coefficients and corresponding entity references from the BioPAX file.
        """
        stoich_list = list(self.tree.findall("biopax:Stoichiometry", namespaces))
        self._log("debug", f"Found {len(stoich_list)} stoichiometry entries")

        self.stoichometry = {}
        iterator = (
            tqdm(stoich_list, desc="Parsing stoichiometry", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else stoich_list
        )

        for stoichometry in iterator:
            ID = stoichometry.get("{%s}ID" % namespaces["rdf"], None)
            stoichiometricCoefficient = stoichometry.find(
                "biopax:stoichiometricCoefficient", namespaces
            ).text
            physicalEntity = stoichometry.find("biopax:physicalEntity", namespaces).get(
                RESOUCE_RDF_STRING
            )
            self.stoichometry[ID] = {
                "stoichiometricCoefficient": stoichiometricCoefficient,
                "physicalEntity": physicalEntity,
            }

        self._log("info", f"Parsed {len(self.stoichometry)} stoichiometry entries")

    def __parse_unixrefs(self):
        """Parse unification cross-references (UniXrefs) from the BioPAX file.

        Extracts database identifiers and names for external references.
        """
        xrefs_list = list(self.tree.findall("biopax:UnificationXref", namespaces))
        self._log("debug", f"Found {len(xrefs_list)} UniXrefs")

        self.uniXrefs = {}
        iterator = (
            tqdm(xrefs_list, desc="Parsing UniXrefs", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else xrefs_list
        )

        for ex_dblink in iterator:
            xrefID = ex_dblink.get(ID_RDF_STRING)
            db_id = ex_dblink.find("biopax:id", namespaces).text
            db_name = ex_dblink.find("biopax:db", namespaces).text
            self.uniXrefs[xrefID] = {"DB_NAME": db_name, "DB_ID": db_id}

        self._log("info", f"Parsed {len(self.uniXrefs)} UniXrefs")

    def __parse_reaction_order(self):
        r"""Parse the order of biochemical pathway steps.
        Extracts relationships between pathway steps, including
        which biochemical reactions occur and which steps follow.
        """
        steps_list = list(self.tree.findall("biopax:PathwayStep", namespaces))
        self._log("debug", f"Found {len(steps_list)} pathway steps")
        reaction_steps = {}
        iterator = (
            tqdm(steps_list, desc="Parsing pathway steps", disable=not TQDM_AVAILABLE)
            if TQDM_AVAILABLE
            else steps_list
        )
        for pathways in iterator:
            stepID = pathways.get("{%s}ID" % namespaces["rdf"])
            stepDetails = [None, []]
            for stepProcess in pathways.findall("biopax:stepProcess", namespaces):
                stepText = stepProcess.get(RESOUCE_RDF_STRING, None).strip("#")
                if "BiochemicalReaction" in stepText:
                    stepDetails[0] = stepText
                else:
                    stepDetails[1].append(stepText)
            nextStep = [
                steps.get(RESOUCE_RDF_STRING, None).strip("#")
                for steps in pathways.findall("biopax:nextStep", namespaces)
            ]
            reaction_steps[stepID] = (stepDetails, nextStep)
        # ← parse loop ends here

        self.reactionOrder = reaction_steps
        self._log("info", f"Parsed {len(self.reactionOrder)} pathway steps")

        # ---- per-pathway traversal: assign (pathway_id, local_order) to each step ----
        from collections import deque

        self.step_order = {}  # stepID -> (pathway_id, local_index)

        for pid, step_ids in self.pathway_to_steps.items():
            step_set = set(step_ids)
            if not step_set:
                continue

            # Find roots: steps in this pathway that no other step in this pathway
            # points to via nextStep.
            incoming = set()
            for sid in step_ids:
                if sid not in self.reactionOrder:
                    continue
                _, next_steps = self.reactionOrder[sid]
                for ns in next_steps:
                    if ns in step_set:
                        incoming.add(ns)
            roots = [s for s in step_ids if s not in incoming]

            # BFS from roots, assign local order in traversal sequence.
            visited = set()
            order_idx = 0
            queue = deque(roots)
            while queue:
                sid = queue.popleft()
                if sid in visited or sid not in step_set:
                    continue
                visited.add(sid)
                self.step_order[sid] = (pid, order_idx)
                order_idx += 1
                if sid in self.reactionOrder:
                    _, next_steps = self.reactionOrder[sid]
                    for ns in next_steps:
                        if ns in step_set and ns not in visited:
                            queue.append(ns)

            # Cycle safety: steps in cycles with no roots won't be reached by BFS.
            for sid in step_ids:
                if sid in step_set and sid not in visited:
                    visited.add(sid)
                    self.step_order[sid] = (pid, order_idx)
                    order_idx += 1

    def __parse_reactions(self):
        reactions_list = list(
            self.tree.findall("biopax:BiochemicalReaction", namespaces)
        )
        self._log("debug", f"Found {len(reactions_list)} biochemical reactions")

        self.reactions = {}
        skipped = 0

        for biochemreac in reactions_list:
            if (
                biochemreac.find("biopax:conversionDirection", namespaces).text
                != "LEFT-TO-RIGHT"
            ):
                skipped += 1
                continue

            left = biochemreac.findall("biopax:left", namespaces)
            right = biochemreac.findall("biopax:right", namespaces)
            lefties = [e.get(RESOUCE_RDF_STRING, None).strip("#") for e in left]
            righties = [e.get(RESOUCE_RDF_STRING, None).strip("#") for e in right]

            ID = biochemreac.get("{%s}ID" % namespaces["rdf"])

            # display_name_el = biochemreac.find("biopax:displayName", namespaces)

            # display_name = (
            #     display_name_el.text.strip().lower()
            #     if display_name_el is not None and display_name_el.text
            #     else ""
            # )

            is_translocation = self._is_translocation(lefties, righties)
            is_broadcast = self._is_broadcast_reaction(lefties, righties)

            if is_translocation:
                paired_left, paired_right = self._pair_translocation(lefties, righties)
                self.reactions[ID] = (paired_left, paired_right, "translocation")
            elif is_broadcast:
                self.reactions[ID] = (lefties, righties, "broadcast")
            else:
                self.reactions[ID] = (lefties, righties, "reaction")

        self._log("info", f"Parsed {len(self.reactions)} reactions ({skipped} skipped)")

    def _is_broadcast_reaction(self, lefties: list, righties: list) -> bool:
        """
        A broadcast reaction produces each right-side entity independently
        from the left — not a single combined product.

        True transcription/expression events have:
        - A small left side (1–3 entities) dominated by DNA
        - A large right side (mostly proteins/complexes) that dwarfs the left

        Returns True if this should emit L→each(R) edges rather than
        cartesian(L) × cartesian(R).
        """
        if len(lefties) > 3:
            return False

        def _is_dna_entity(eid: str) -> bool:
            if eid in self.dna:
                return True
            if eid in self.physical_entity:
                members = self.physical_entity[eid].get("members", [])
                return bool(members) and all(mid in self.dna for mid in members)
            return False

        dna_count = sum(1 for eid in lefties if _is_dna_entity(eid))
        if dna_count / max(len(lefties), 1) <= 0.5:
            return False

        right_protein_count = sum(
            1 for eid in righties if eid in self.proteins or eid in self.complexes
        )
        right_mostly_proteins = right_protein_count / max(len(righties), 1) > 0.8

        # Right side must massively outnumber left (gene → many products)
        fan_out = len(righties) >= len(lefties) * 5 and len(righties) > 5

        return right_mostly_proteins and fan_out

    def _pair_translocation(self, lefties: list, righties: list):
        """
        For bulk secretion/exocytosis, pair each left entity with its
        name-matched right entity. PhysicalEntity placeholders (no name match)
        are dropped entirely.
        """

        def name_of(eid):
            for store in (self.proteins, self.molecules, self.dna, self.rna):
                if eid in store:
                    return store[eid]["name"]
            return None  # PhysicalEntity placeholder — no match possible

        right_by_name = {}
        for e in righties:
            n = name_of(e)
            if n:
                right_by_name[n] = e

        paired_left, paired_right = [], []
        for left_entity in lefties:
            name = name_of(left_entity)
            if name and name in right_by_name:
                paired_left.append(left_entity)
                paired_right.append(right_by_name[name])

        return paired_left, paired_right

    def _is_translocation(self, lefties: list, righties: list) -> bool:
        all_stores = (self.proteins, self.molecules, self.dna, self.rna)

        def name_and_loc(eid):
            for store in all_stores:
                if eid in store:
                    return store[eid]["name"], store[eid].get("cellularLocation")
            return None, None

        left_by_name = {}
        for l in lefties:
            name, loc = name_and_loc(l)
            if name:
                left_by_name[name] = loc

        if not left_by_name:
            return False

        translocation_count = 0
        matched_count = 0
        for r in righties:
            name, r_loc = name_and_loc(r)
            if name and name in left_by_name:
                matched_count += 1
                if left_by_name[name] != r_loc:
                    translocation_count += 1

        if matched_count == 0:
            return False

        # Majority of name-matched right entities must come from a different location
        return (translocation_count / matched_count) > 0.5

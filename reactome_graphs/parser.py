import xml.etree.ElementTree as ET
import networkx as nx

namespaces = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xml_schema": "http://www.w3.org/2001/XMLSchema#",
    "biopax": "http://www.biopax.org/release/biopax-level3.owl#",
}

ID_RDF_STRING = "{%s}ID" % namespaces["rdf"]
RESOUCE_RDF_STRING = "{%s}resource" % namespaces["rdf"]


class ReactomeBioPAX:
    def __init__(self):
        self.tree = None

    def parse_biopax3_file(self, filename: str):
        r"""Parse a BioPAX Level 3 XML file and populate internal data structures.

        Args:
            filename (str): Path to the BioPAX Level 3 file.
        """
        self.tree = ET.parse(filename)

        self.__parse_unixrefs()
        self.__parse_cellular_location()

        self.__parse_molecules()
        self.__parse_proteins()
        self.__parse_protein_complexes()

        self.__parse_reactions()
        self.__parse_reaction_order()
        self.__physical_entity()

        self.__parse_stoichometry()

    def __parse_molecules(self):
        r"""Parse small molecule entities from the BioPAX file.

        Extracts display names, aliases, and comments for each molecule
        and stores them in a dictionary.
        """
        self.molecules = {}
        for compounds in self.tree.findall("biopax:SmallMolecule", namespaces):
            moleculesID = compounds.get(ID_RDF_STRING, None)
            disName = compounds.find("biopax:displayName", namespaces).text
            aliases = [nm.text for nm in compounds.findall("biopax:name", namespaces)]
            comment = compounds.find("biopax:comment", namespaces).text
            self.molecules[moleculesID] = {
                "name": disName,
                "aliases": aliases,
                "comment": comment,
                "color": "green",
            }

    def __parse_cellular_location(self):
        r"""Parse cellular location vocabularies from the BioPAX file.

        Extracts terms and cross-references for cellular locations
        and stores them in a dictionary.
        """
        self.cellularLocations = {}
        for locations in self.tree.findall(
            "biopax:CellularLocationVocabulary", namespaces
        ):
            term = locations.find("biopax:term", namespaces).text
            xref = self.uniXrefs[
                locations.find("biopax:xref", namespaces)
                .get(RESOUCE_RDF_STRING, None)
                .strip("#")
            ]
            self.cellularLocations[locations.get(ID_RDF_STRING, None)] = (term, xref)

    def __parse_proteins(self):
        r"""Parse protein entities from the BioPAX file.

        Extracts names, aliases, cellular locations, complex membership,
        and Reactome DB IDs for each protein.
        """
        self.proteins = {}
        for proteins in self.tree.findall("biopax:Protein", namespaces):
            proteinID = proteins.get(ID_RDF_STRING, None)
            complexity = proteins.findall("biopax:memberPhysicalEntity", namespaces)
            disName = proteins.find("biopax:displayName", namespaces).text
            aliases = [nm.text for nm in proteins.findall("biopax:name", namespaces)]
            comment = proteins.find("biopax:comment", namespaces).text
            reactome_db_id = (
                comment.strip("Reactome DB_ID: ")
                if "Reactome DB_ID: " in comment
                else None
            )
            cellularLocation = (
                proteins.find("biopax:cellularLocation", namespaces)
                .get(RESOUCE_RDF_STRING, None)
                .strip("#")
            )
            complexes = []
            if len(complexity) > 0:
                for complex in complexity:
                    complexes.append(complex.get(RESOUCE_RDF_STRING, None))

            self.proteins[proteinID] = {
                "name": disName,
                "aliases": aliases,
                "reactome_db_id": reactome_db_id,
                "comment": comment,
                "cellularLocation": cellularLocation,
                "complexes": complexes,
                "color": "red",
            }

    def __parse_protein_complexes(self):
        r"""Parse protein complex entities from the BioPAX file.

        Extracts components, stoichiometries, cellular locations,
        cross-references, and Reactome DB IDs for complexes.

        Returns:
            dict: A dictionary mapping complex IDs to their associated metadata.
        """
        self.complexes = {}
        for complex in self.tree.findall("biopax:Complex", namespaces):
            complexId = complex.get(ID_RDF_STRING, None)
            disName = complex.find("biopax:displayName", namespaces).text

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
            disName = complex.find("biopax:displayName", namespaces).text
            xref = [
                xre.get(RESOUCE_RDF_STRING, None).strip("#")
                for xre in complex.findall("biopax:xref", namespaces)
            ]
            dataSource = complex.find("biopax:dataSource", namespaces).text
            comment = complex.find("biopax:comment", namespaces).text
            reactome_db_id = (
                comment.strip("Reactome DB_ID: ")
                if "Reactome DB_ID: " in comment
                else None
            )

            self.complexes[complexId] = {
                "name": disName,
                "components": component,
                "reactome_db_id": reactome_db_id,
                "cellularLocation": cellularLocation,
                "xref": xref,
                "componentStoichiometry": componentStoichiometry,
                "dataSource": dataSource,
            }

        return self.complexes

    def __physical_entity(self):
        """Parse physical entities from the BioPAX file.

        Extracts display names, cellular locations, and cross-references
        for physical entities.
        """
        self.physical_entity = {}
        for physEntity in self.tree.findall("biopax:PhysicalEntity", namespaces):
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
            self.physical_entity[ID] = {
                "name": disName,
                "cellularLocation": cellularLocation,
                "xref": xref,
            }

    def __parse_stoichometry(self):
        """Parse stoichiometric coefficients for physical entities.

        Extracts coefficients and corresponding entity references from the BioPAX file.
        """
        self.stoichometry = {}
        for stoichometry in self.tree.findall("biopax:Stoichiometry", namespaces):
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

    def __parse_unixrefs(self):
        """Parse unification cross-references (UniXrefs) from the BioPAX file.

        Extracts database identifiers and names for external references.
        """
        self.uniXrefs = {}
        for ex_dblink in self.tree.findall("biopax:UnificationXref", namespaces):
            xrefID = ex_dblink.get(ID_RDF_STRING)
            db_id = ex_dblink.find("biopax:id", namespaces).text
            db_name = ex_dblink.find("biopax:db", namespaces).text
            self.uniXrefs[xrefID] = (db_id, db_name)

    def __parse_reactions(self):
        """Parse biochemical reactions and their directional relationships.

        Extracts left-to-right reactants and products and filters
        out non-directional reactions.
        """
        self.reactions = {}
        for biochemreac in self.tree.findall("biopax:BiochemicalReaction", namespaces):
            left = biochemreac.findall("biopax:left", namespaces)
            right = biochemreac.findall("biopax:right", namespaces)

            lefties = []
            for entities in left:
                lefties.append(entities.get(RESOUCE_RDF_STRING, None).strip("#"))
            righties = []
            for r in right:
                righties.append(r.get(RESOUCE_RDF_STRING, None).strip("#"))

            if (
                biochemreac.find("biopax:conversionDirection", namespaces).text
                == "LEFT-TO-RIGHT"
            ):
                ID = biochemreac.get("{%s}ID" % namespaces["rdf"])
                self.reactions[ID] = (lefties, righties)
            else:
                Warning(
                    f"Skipping this reaction as it is not Left to Right and is {biochemreac.find('biopax:conversionDirection', namespaces).text}"
                )

    def __parse_reaction_order(self):
        r"""Parse the order of biochemical pathway steps.

        Extracts relationships between pathway steps, including
        which biochemical reactions occur and which steps follow.
        """
        reaction_steps = {}
        for pathways in self.tree.findall("biopax:PathwayStep", namespaces):
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
        self.reactionOrder = reaction_steps

    def __flatten_complex_into_protein(self, entity):
        """Recursively expand a complex entity into its constituent proteins.

        Args:
            entity (str): The ID of a complex or protein entity.

        Returns:
            list[str]: A flat list of protein entity IDs contained in the complex.
        """
        proteins = []
        if entity in self.complexes:
            _, components, _ = self.complexes[entity]
            for component_id in components:
                proteins.extend(self.__flatten_complex_into_protein(component_id))
        else:
            return [entity]
        return proteins

    def parse_biopax_into_networkx(self, filename: str):
        """Parse a BioPAX Level 3 file into a directed NetworkX graph.

        Constructs a directed reaction network where nodes represent molecular
        entities and edges represent biochemical reactions or complex relationships.

        Args:
            filename (str): Path to the BioPAX Level 3 file.

        Returns:
            networkx.DiGraph: A directed graph representing biochemical interactions.
        """
        self.parse_biopax3_file(filename)
        G = nx.DiGraph()
        step = 0
        for _, step_dets in self.reactionOrder.items():
            if step_dets[0][0] is not None:
                step += 1
                reactions = self.reactions[step_dets[0][0]]
                for left in reactions[0]:
                    for right in reactions[1]:
                        G.add_edge(left, right, time=step, is_a="reaction")

        top_level_complexes = [node for node in G.nodes if "Complex" in node]

        for node in top_level_complexes:
            for component in self.__flatten_complex_into_protein(node):
                if "Protein" in component:
                    G.add_node(component, complex_component=True, color="violet")
                    G.add_edge(component, node, complex_relationship=True)

        for node in G.nodes:
            if "Protein" in node:
                G.nodes[node].update(self.proteins[node])
            elif "SmallMolecule" in node:
                G.nodes[node].update(self.molecules[node])
            elif "PhysicalEntity" in node:
                G.nodes[node].update(self.physical_entity[node])
            else:
                pass

        protein_map = dict(
            (k, self.proteins[k]["name"]) for k, v in self.proteins.items()
        )
        molecules_map = dict(
            (k, self.molecules[k]["name"]) for k, v in self.molecules.items()
        )
        physical_entity_map = dict(
            (k, self.physical_entity[k]["name"])
            for k, v in self.physical_entity.items()
        )
        complexes_map = dict(
            (k, self.complexes[k][0]) for k, v in self.complexes.items()
        )

        G = nx.relabel_nodes(
            G, {**protein_map, **molecules_map, **physical_entity_map, **complexes_map}
        )
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    def parse_biopax_into_ubergraphs(self, filename: str):
        raise NotImplementedError("What Can I Say?")

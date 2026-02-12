import xml.etree.ElementTree as ET
import networkx as nx
import pandas as pd
import typing
import logging

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
    def __init__(self, uniprot_accession_num = False):
        self.tree = None
        self.uniprot_accession_num = uniprot_accession_num

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
        # self.__parse_entity_references()
        self.__parse_catalysis()
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
            self.cellularLocations[locations.get(ID_RDF_STRING, None)] = {"common_name": term, **xref}

    def __parse_proteins(self):
        r"""Parse protein entities from the BioPAX file.

        Extracts names, aliases, cellular locations, complex membership,
        and Reactome DB IDs for each protein.
        """
        if self.uniprot_accession_num:
            df_uni_reactome = pd.read_csv(
                "/Users/bty644/Desktop/phd/code/collate_data/reactome/data/UniProt2Reactome_PE_All_Levels.txt",
                sep="\t",
                header=None,  # no header row in file
                names=["source_db", "stable_id", "entity_name", "pathway_id", "url", "event_name", "evidence_code", "species"],
            )
        self.proteins = {}
        for proteins in self.tree.findall("biopax:Protein", namespaces):
            proteinID = proteins.get(ID_RDF_STRING, None)
            complexity = proteins.findall("biopax:memberPhysicalEntity", namespaces)
            disName = proteins.find("biopax:displayName", namespaces).text
            if disName is None:
                disName = proteins.find("biopax:name", namespaces).text

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
            xref = [
                self.uniXrefs.get(xre.get(RESOUCE_RDF_STRING, None).strip("#"), xre.get(RESOUCE_RDF_STRING, None).strip("#"))
                for xre in proteins.findall("biopax:xref", namespaces)
            ]

            if self.uniprot_accession_num:
                for ref in xref: 
                    if ref["DB_NAME"] == "Reactome":
                        try:
                            db_id = ref["DB_ID"]
                            uni_id = df_uni_reactome[df_uni_reactome['stable_id'] == db_id].iloc[0]['source_db']
                        except Exception:
                            uni_id = "not_available"

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
                "xref": xref,
                "color": "red",
            }
            if self.uniprot_accession_num:
                self.proteins[proteinID]["uniprot_id"] = uni_id

    def __parse_catalysis(self):
        self.catalysis_dets = {}

        for catalysis in self.tree.findall("biopax:Catalysis", namespaces):
            controlType = catalysis.find("biopax:controlType", namespaces).text.strip("#")
            controller = catalysis.find("biopax:controller", namespaces).get(RESOUCE_RDF_STRING, None).strip("#")
            controlled = catalysis.find("biopax:controlled", namespaces).get(RESOUCE_RDF_STRING, None).strip("#")

            uniXref = [xref.get(RESOUCE_RDF_STRING, None) for xref in catalysis.findall("biopax:xref", namespaces)]
            self.catalysis_dets[controlled] = {
                "controlType" : controlType,
                "controller" : controller,
                "uniXref" : uniXref
            }

    def __parse_entity_references(self):
        self.ProteinReference = {}
        for entityReference in self.tree.findall("biopax:ProteinReference", namespaces):
            entityReference = entityReference.find("biopax:displayName", namespaces).text
        

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
                self.uniXrefs.get(xre.get(RESOUCE_RDF_STRING, None).strip("#"), xre.get(RESOUCE_RDF_STRING, None).strip("#"))
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
                "complex_node" : True
            }

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
            self.uniXrefs[xrefID] = {
                "DB_NAME" : db_name,
                "DB_ID" : db_id
            }

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
            components = self.complexes[entity]["components"]
            for component_id in components:
                proteins.extend(self.__flatten_complex_into_protein(component_id))
        else:
            return [entity]
        return proteins

    def parse_biopax_into_networkx(
        self, 
        filename: str, 
        complex_proteins_as_node: bool = False,
        logger: typing.Optional[logging.Logger] = None
    ):
        """Parse a BioPAX Level 3 file into a directed NetworkX graph.

        Constructs a directed reaction network where nodes represent molecular
        entities and edges represent biochemical reactions or complex relationships.

        Args:
            filename (str): Path to the BioPAX Level 3 file.
            complex_proteins_as_node (bool): If True, expand complex proteins as separate nodes.
            logger (logging.Logger, optional): Logger instance for debug/info messages. 
                If None, no logging occurs.

        Returns:
            networkx.DiGraph: A directed graph representing biochemical interactions.
        """
        if logger is not None:
            logger.info(f"Parsing BioPAX file: {filename}")
        
        self.parse_biopax3_file(filename)
        G = nx.DiGraph()
        step = 0
        reaction_count = 0
        
        for _, step_dets in self.reactionOrder.items():
            if step_dets[0][0] is not None:
                step += 1
                reactions = self.reactions[step_dets[0][0]]
                catalysis_exists = self.catalysis_dets.get(step_dets[0][0], None)
                
                if catalysis_exists is not None:
                    G.add_node(catalysis_exists["controller"], time=step)
                    if logger is not None:
                        logger.debug(f"Step {step}: Added catalyst node '{catalysis_exists['controller']}'")
                
                for left in reactions[0]:
                    G.add_node(left, time=step)

                    if "Complex" in left:
                        protein_in_complexes = {}
                        for component in self.__flatten_complex_into_protein(left):
                            if complex_proteins_as_node:
                                G.add_node(component, complex_component=True, color="violet")
                                G.add_edge(component, left, is_a="complex_component", time=step)
                            else:
                                if "Protein" in component:
                                    protein_in_complexes[component] = self.proteins[component]  

                        G.nodes[left].update({"constituents": protein_in_complexes})  
                        G.nodes[left].update({"type": "Complex", **self.complexes[left]})
                    
                    if catalysis_exists is not None:
                        G.add_edge(catalysis_exists["controller"], left, time=step, is_a="catalysis", catalysis_type = catalysis_exists["controlType"])
                    
                    for right in reactions[1]:
                        G.add_node(right, time=step)
                        G.add_edge(left, right, time=step, is_a="reaction")
                        reaction_count += 1

                        if "Complex" in right:
                            protein_in_complexes = {}
                            for component in self.__flatten_complex_into_protein(right):
                                if complex_proteins_as_node:
                                    G.add_node(component, complex_component=True, color="violet")
                                    G.add_edge(component, right, is_a="complex_component", time=step)
                                else:
                                    if "Protein" in component:
                                        protein_in_complexes[component] = self.proteins[component]  

                            G.nodes[right].update({"constituents": protein_in_complexes})  
                            G.nodes[right].update({"type": "Complex", **self.complexes[right]})

        if logger is not None:
            logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges over {step} steps ({reaction_count} reactions)")

        # Node type annotations
        for node in G.nodes:
            if "Protein" in node:
                G.nodes[node].update({"type": "protein", **self.proteins[node]})
            elif "SmallMolecule" in node:
                G.nodes[node].update({"type": "small_molecule", **self.molecules[node]})
            elif "PhysicalEntity" in node:
                G.nodes[node].update({"type": "physical_entity", **self.physical_entity[node]})

        # Relabeling mappings
        protein_map = dict((k, self.proteins[k]["name"]) for k, v in self.proteins.items())
        molecules_map = dict((k, self.molecules[k]["name"]) for k, v in self.molecules.items())
        physical_entity_map = dict((k, self.physical_entity[k]["name"]) for k, v in self.physical_entity.items())
        complexes_map = dict((k, self.complexes[k]["name"]) for k, v in self.complexes.items())

        G = nx.relabel_nodes(G, {**protein_map, **molecules_map, **physical_entity_map, **complexes_map})
        G.remove_edges_from(nx.selfloop_edges(G))
        
        if logger is not None:
            logger.info("Graph relabeling complete. Self-loops removed.")
        
        return G

    def parse_biopax_into_ubergraphs(self, filename: str):
        raise NotImplementedError("What Can I Say?")

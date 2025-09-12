import argparse
import os.path
import re
import xml.etree.ElementTree as ET
import networkx as nx
import requests 
import pandas as pd

namespaces = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'owl': 'http://www.w3.org/2002/07/owl#',
    'xml_schema': 'http://www.w3.org/2001/XMLSchema#',
    'biopax': 'http://www.biopax.org/release/biopax-level3.owl#',
}

ID_RDF_STRING = "{%s}ID" % namespaces['rdf']
RESOUCE_RDF_STRING = "{%s}resource" % namespaces['rdf']

class ReactomeBioPAX():
    def __init__(self):
        self.tree = None
        self.hypergraph = False
        self.molecules = None
        self.proteins = None
        self.reactions = None
        self.reactionOrder = None

    def download_list_of_pathways(save_dir: str = "./"):
        res = requests.get('https://reactome.org/download/current/ReactomePathways.txt')
        if res.ok:
            with open(f"{save_dir}list_of_pathways.txt", mode="wb") as file:
                file.write(res.content)

    def download_biopax_file(save_dir: str = "./"):
        list_of_pathways = pd.read_csv(f"./data/list_of_pathways.txt", delimiter="\t", names = ["id", "name", "species"])

    def parse_biopax3_file(self, filename: str, hypergraph: bool = False):
        self.tree = ET.parse(filename)
        self.hypergraph = hypergraph

        self.uniXrefs = self.__parse_unixrefs()
        self.cellularLocations = self.__parse_cellular_location()

        self.molecules = self.__parse_molecules()
        self.proteins = self.__parse_proteins()
        self.complexes = self.__parse_protein_complexes()

        self.reactions = self.__parse_reactions()
        self.reactionOrder = self.__parse_reaction_order()

    def __parse_molecules(self):
        molecules = {}
        for compounds in self.tree.findall("biopax:SmallMolecule", namespaces):
            moleculesID = compounds.get(ID_RDF_STRING)
            disName = compounds.find("biopax:displayName", namespaces).text
            aliases = [nm.text for nm in compounds.findall("biopax:name", namespaces)]
            comment = compounds.find("biopax:comment", namespaces).text
            molecules[moleculesID] = (disName, aliases, comment)
        return molecules
        
    def __parse_cellular_location(self):
        cellularLocations = {}
        for locations in self.tree.findall("biopax:CellularLocationVocabulary", namespaces):
            term = locations.find("biopax:term", namespaces).text
            xref = self.uniXrefs[locations.find("biopax:xref", namespaces).get(RESOUCE_RDF_STRING).strip("#")]
            cellularLocations[locations.get(ID_RDF_STRING)] = (term, xref)
        return cellularLocations

    def __parse_proteins(self):
        protein_dict = {}
        for proteins in self.tree.findall("biopax:Protein", namespaces):
            proteinID = proteins.get(ID_RDF_STRING)
            complexity = proteins.findall("biopax:memberPhysicalEntity", namespaces)
            disName = proteins.find("biopax:displayName", namespaces).text
            aliases = [nm.text for nm in proteins.findall("biopax:name", namespaces)]
            comment = proteins.find("biopax:comment", namespaces).text
            cellularLocation = proteins.find("biopax:cellularLocation", namespaces).get(RESOUCE_RDF_STRING).strip("#")
            complexes = []
            if len(complexity) > 0:
                for complex in complexity:  
                    complexes.append(complex.get(RESOUCE_RDF_STRING))

            if self.hypergraph:
                protein_dict[proteinID] = (disName, aliases, comment, cellularLocation, complexes)
            else:
                protein_dict[proteinID] = (disName, aliases, comment, self.cellularLocations.get(cellularLocation), complexes)
            
        return protein_dict

    def __parse_protein_complexes(self):
        complex_dict = {}
        for complex in self.tree.findall("biopax:Complex", namespaces):
            complexId = complex.get(ID_RDF_STRING)
            disName = complex.find("biopax:displayName", namespaces).text

            component = [cmop.get(RESOUCE_RDF_STRING).strip("#") for cmop in complex.findall("biopax:component", namespaces)]
            componentStoichiometry = [compStoicho.text for compStoicho in complex.findall("biopax:componentStoichiometry", namespaces)]

            cellularLocation = complex.find("biopax:cellularLocation", namespaces).get(RESOUCE_RDF_STRING).strip("#")
            disName = complex.find("biopax:displayName", namespaces).text
            xref = complex.findall("biopax:xref", namespaces)
            dataSource = complex.findall("biopax:dataSource", namespaces)
            comment = complex.findall("biopax:comment", namespaces)

            complex_dict[complexId] = (disName, component, cellularLocation)

        return complex_dict
    
    def __physical_entity(self):
        self.physical_entity = {}
        for physEntity in self.tree.findall("biopax:PhysicalEntity", namespaces):
            ID = physEntity.get("{%s}ID" % namespaces['rdf'])
            disName = physEntity.find("biopax:displayName", namespaces).text
            cellularLocation = complex.find("biopax:cellularLocation", namespaces).get(RESOUCE_RDF_STRING).strip("#")
            xref = [xre.get(RESOUCE_RDF_STRING).strip("#") for xre in complex.findall("biopax:xref", namespaces)]
            self.physical_entity[ID] = (disName, cellularLocation, xref)
        


    def __parse_stoichometry(self):
        for stoichometry in self.tree.findall("biopax:Stoichiometry", namespaces):
            component = stoichometry.findall("biopax:component", namespaces)

    def __parse_unixrefs(self):
        uniXrefs = {}
        for ex_dblink in self.tree.findall("biopax:UnificationXref", namespaces):
            xrefID = ex_dblink.get(ID_RDF_STRING)
            db_id = ex_dblink.find("biopax:id", namespaces).text
            db_name = ex_dblink.find("biopax:db", namespaces).text
            uniXrefs[xrefID] = (db_id, db_name)
        return uniXrefs

    def __parse_reactions(self):
        reactions = {}
        for biochemreac in self.tree.findall("biopax:BiochemicalReaction", namespaces):
            left = biochemreac.findall("biopax:left", namespaces)
            right = biochemreac.findall("biopax:right", namespaces)
            
            lefties = []
            for l in left:
                lefties.append(l.get(RESOUCE_RDF_STRING).strip("#"))
            righties = []   
            for r in right:
                righties.append(r.get(RESOUCE_RDF_STRING).strip("#"))

            if biochemreac.find("biopax:conversionDirection", namespaces).text == "LEFT-TO-RIGHT":
                ID = biochemreac.get("{%s}ID" % namespaces['rdf'])
                reactions[ID] = (lefties, righties)
        return reactions

    def __parse_reaction_order(self):
        reaction_steps = {}
        for pathways in self.tree.findall("biopax:PathwayStep", namespaces):
            stepID = int(pathways.get("{%s}ID" % namespaces['rdf']).strip("PathwayStep"))
            stepDetails = [None, []]
            for stepProcess in pathways.findall("biopax:stepProcess", namespaces):
                stepText = stepProcess.get(RESOUCE_RDF_STRING).strip("#")
                if "BiochemicalReaction" in stepText:
                    stepDetails[0] = stepText
                else:
                    stepDetails[1].append(stepText)

            nextStep = [steps.get(RESOUCE_RDF_STRING).strip("#") for steps in pathways.findall("biopax:nextStep", namespaces)]
            reaction_steps[stepID] = (stepDetails, nextStep)
        reaction_steps = dict(sorted(reaction_steps.items()))
        return [*reaction_steps.values()]

    def __flatten_complex_into_protein(self, entity):
        proteins = []
        if entity in self.complexes:
            _, components, _ = self.complexes[entity]
            for component_id in components:
                proteins.extend(self.__flatten_complex_into_protein(component_id))
        else:
            return [entity]
        return proteins
    
    def __complex_undirected_edges(self, G: nx.DiGraph, side):
        if type(side) is list:
            last_node = None
            for entities in side:
                if last_node and last_node != entities:
                    G.add_edge(last_node, entities, is_a = "complex")
                    G.add_edge(entities, last_node, is_a = "complex")
                last_node = entities

    def parse_biopax_into_networkx(self, filename: str):
        self.parse_biopax3_file(filename)
        G = nx.DiGraph()
        for _, reactions in self.reactions.items():
            left = [self.__flatten_complex_into_protein(r) if "Complex" in r else r for r in reactions[0]]
            right = [self.__flatten_complex_into_protein(r) if "Complex" in r else r for r in reactions[1]]
            for l in left:
                self.__complex_undirected_edges(G, l)
                for r in right:
                    self.__complex_undirected_edges(G, l)
                    if type(l) is list:
                        for en1 in l:
                            if type(r) is list:
                                for en2 in r:
                                    G.add_edge(en1, en2, is_a = "reaction")
                            else:
                                G.add_edge(en1, r, is_a = "reaction")
                    elif type(r) is list:
                        for rr in r:
                            G.add_edge(l, rr, is_a = "reaction")
                    else:
                        G.add_edge(l, r, is_a = "reaction")
        
        for node in G.nodes:
            if "Protein" in node:
                G.nodes[node].update({
                    "aliases" : self.proteins[node][1],
                    "reactome_id" : self.proteins[node][2],
                    "cellular_location" : self.proteins[node][3],
                    "complex_membership" : self.proteins[node][4],
                })
        protein_map = dict((k,self.proteins[k][0]) for k,v in self.proteins.items())
        molecules_map = dict((k,self.proteins[k][0]) for k,v in self.molecules.items())
        physical_entity_map = dict((k,self.proteins[k][0]) for k,v in self.physical_entity.items())
        G = nx.relabel_nodes(G, protein_map)
        return G
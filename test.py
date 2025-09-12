from reactome_parser import ReactomeBioPAX
from pyvis.network import Network

parser = ReactomeBioPAX()
G = parser.parse_biopax_into_networkx("./data/biopax3/R-HSA-68689.xml")

nt = Network('1000px', '1000px', notebook=False, directed=True)
nt.from_nx(G)
nt.save_graph('./nx.html')
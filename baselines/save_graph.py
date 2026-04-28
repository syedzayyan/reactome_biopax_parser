import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from reactome_graphs import NodeFeaturiser, ReactomeBioPAX, ReactomeViz

plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = ReactomeBioPAX(uniprot_accession_num=True)
G = parser.parse_biopax_into_networkx(
    "../data/biopax3/R-HSA-168256.xml", reaction_partners=True, include_complexes=True
)
node_feats = NodeFeaturiser(G, xref_dict=parser.uniXrefs)
node_feats.download_and_store()
node_feats.featurise(
    add_go_embedding=True, go_embedding_path="../go_node2vec_matrix.pt"
)
viz = ReactomeViz(G)
viz.print_stats()


def safe(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    if isinstance(x, (list, dict)):
        return json.dumps(x)
    return x


# --- PCA on node features -------------------------------------------------
K_PCA = 4

# collect features as (node, vector) pairs, keeping only the modal dim
raw = [(n, G.nodes[n].get("feature")) for n in G.nodes]
raw = [(n, v) for n, v in raw if v is not None and hasattr(v, "shape")]
dims = [v.shape[0] for _, v in raw]
if dims:
    from collections import Counter

    modal_dim = Counter(dims).most_common(1)[0][0]
    raw = [(n, v) for n, v in raw if v.shape[0] == modal_dim]
    print(f"[PCA] {len(raw)} nodes with dim={modal_dim}")

    X = np.stack([v for _, v in raw]).astype(np.float32)
    # drop constant columns to avoid NaN
    keep = X.var(axis=0) > 1e-12
    X = X[:, keep]

    pca = PCA(n_components=K_PCA)
    scores = pca.fit_transform(X)
    print(f"[PCA] explained variance: {pca.explained_variance_ratio_.round(3)}")

    for (node, _), s in zip(raw, scores):
        for j in range(K_PCA):
            G.nodes[node][f"pc{j + 1}"] = float(s[j])

    # nodes with no valid features get zeros so the CSV column is numeric
    for n in G.nodes:
        for j in range(K_PCA):
            key = f"pc{j + 1}"
            if key not in G.nodes[n]:
                G.nodes[n][key] = 0.0
else:
    print("[PCA] no usable features found")


# --- edges ---
edges_df = nx.to_pandas_edgelist(G)
edges_df.to_csv(".cache/edges_partners_se.csv", index=False)

# --- nodes (drop the heavy `feature` column — R only needs pc1..pcK + type) ---
rows = []
for n, attr in G.nodes(data=True):
    d = {"node": n}
    for k, v in attr.items():
        if k == "feature":
            continue  # R doesn't need the full vector anymore
        d[k] = safe(v)
    rows.append(d)
pd.DataFrame(rows).to_csv(".cache/nodes_partners_se.csv", index=False)

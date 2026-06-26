"""
fish_rgcn.py — RGCN multi-task baseline for the FISH benchmark.

Single shared encoder, three task heads:
  * existence — binary, BCE with K negative samples per positive edge
  * type      — k-way classification over edge types, cross-entropy on
                positive edges only
  * order     — either regression on normalised step (default), or CORN
                ordinal classification over `order_bins` quantile buckets

Heterogeneous handling
----------------------
Node features arrive at native modality dimension (ESM-2 320 for proteins,
Morgan 2048 for small molecules, k-mer 256 for nucleic acids, padded for
complexes, zero for `other`). A per-node-type linear projects each into the
shared hidden dimension before message passing — no zero-padding to a
common dim, no PCA bottleneck.

Edge heterogeneity is handled by RGCNConv: each of the K edge types gets its
own message-passing weight matrix.

Splits
------
Temporal: sort edges by `time`, hold out the latest fraction (default 20%).
For type and order, only positive edges are used. For existence, K random
negatives are sampled per positive against the training-node vocabulary.

Usage
-----
    from fish_rgcn import build_dataset, FISHRGCN, train, evaluate

    data = build_dataset(G, order_mode="regression", n_order_bins=20)
    model = FISHRGCN(data, hidden=128, n_layers=2,
                     order_mode="regression", n_order_bins=20)
    history = train(model, data, epochs=200, lr=1e-3)
    metrics = evaluate(model, data)
    print(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, RGATConv, RGCNConv

# coral-pytorch provides reference CORN and CORAL implementations from the
# authors of those papers (Raschka et al.). Used here in preference to a
# handrolled loss so the comparison is canonical and not algorithm-confounded.
from coral_pytorch.losses import corn_loss as _corn_loss
from coral_pytorch.losses import coral_loss as _coral_loss
from coral_pytorch.dataset import corn_label_from_logits as _corn_label_from_logits

# Shared fixed-frequency cosine time encoder (GraphMixer-style), also used by
# fish_tgat.py and fish_sthn.py so "time encoding" is the same module
# everywhere it's ablated.
from time_encoding import TimeEncode


# ── relational message passing with edge-time features ─────────────────────


class RGCNTimeConv(MessagePassing):
    """
    RGCN with per-edge time encoding folded into the message.

    For each edge (u -> v, relation r, time t):
        m = W_r * h_u + W_t * phi(t)
    Messages are summed over incoming neighbours, then projected and combined
    with a self-loop term (matching RGCNConv's basis behaviour, simplified).

    This is the (A) variant from the design choices: time enters at message
    passing, so the encoder is genuinely temporal — embeddings at every layer
    incorporate when each interaction happened, not just what.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, time_dim: int):
        super().__init__(aggr="mean")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # One source weight per relation, plus a self-loop weight.
        self.W_rel = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_time = nn.Linear(time_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_rel)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_time.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        # Relational projection. The naive per-edge gather (rel_W = W_rel[
        # edge_type], shape (E, in_dim, out_dim)) materialises a tensor that
        # is E * in_dim * out_dim floats — 58 GB at E=14k, hidden=1024.
        # Instead, project ALL source features under every relation matrix
        # once -> (N, R, out_dim) -> then gather by edge_type for each edge.
        # Memory drops from O(E * H^2) to O(N * R * H) + O(E * H).
        all_proj = torch.einsum("ni,rio->nro", x, self.W_rel)   # (N, R, out_dim)
        # Gather: for edge e, take row src[e] and relation rel[e].
        src_ids = edge_index[0]                                  # (E,)
        msg_rel = all_proj[src_ids, edge_type]                   # (E, out_dim)

        # Add time term per edge.
        msg = msg_rel + self.W_time(edge_time_feat)              # (E, out_dim)

        # Aggregate by destination. Pass `size` so destination tensor has
        # full N rows (otherwise propagate omits nodes with no incoming edges).
        N = x.size(0)
        out = self.propagate(edge_index, x=x, msg=msg, size=(N, N))
        out = out + self.W_self(x) + self.bias
        return out

    def message(self, msg):
        return msg



# ── data container ──────────────────────────────────────────────────────────


@dataclass
class FISHData:
    """Tensors and bookkeeping for one parsed FISH graph."""

    # Per-node-type feature arrays, each shape (n_nodes_of_type, type_dim).
    feats_by_type: dict[int, torch.Tensor]
    # node_id -> node_type id (int)
    node_type: torch.Tensor                       # (N,) long
    # For each node, its row index inside its type's feature matrix.
    node_intratype_idx: torch.Tensor              # (N,) long
    # Edge index over the full node set (training-edges only by default).
    edge_index: torch.Tensor                      # (2, E_train) long
    edge_type: torch.Tensor                       # (E_train,) long
    edge_time: torch.Tensor                       # (E,) float (all edges)

    # All edges including test (used by the heads and for evaluation):
    all_src: torch.Tensor                         # (E,) long
    all_dst: torch.Tensor                         # (E,) long
    all_type: torch.Tensor                        # (E,) long
    all_time: torch.Tensor                        # (E,) float
    train_mask: torch.Tensor                      # (E,) bool
    test_mask: torch.Tensor                       # (E,) bool
    # Inductive test mask: edges with neither endpoint seen during training.
    # Populated only by the semi-inductive split; otherwise all-False. Lets
    # these otherwise-discarded edges contribute a separate "feature-only"
    # prediction metric where the model has no graph context for either
    # endpoint and must score from features alone.
    inductive_mask: Optional[torch.Tensor] = None

    # Ordinal targets (only used if order_mode='corn' or 'coral').
    order_bin: Optional[torch.Tensor] = None      # (E,) long in [0, n_bins-1]

    # Regression target for the order head, transformed from raw step times
    # according to the chosen ``time_target`` mode (see build_dataset).
    # Stored on the dataset so the loss reads from here directly, rather
    # than recomputing the normalisation inside the training loop.
    order_target: Optional[torch.Tensor] = None   # (E,) float in [0, 1]

    # Per-node compartment feature, shape (N, compartment_dim). Zero matrix
    # when compartment features were not requested (handled by use_compartment
    # check on the model side).
    compartment_feat: Optional[torch.Tensor] = None
    compartment_dim: int = 0
    compartment_vocab: list[str] = field(default_factory=list)

    # Vocabularies for reference / decoding.
    node_type_names: list[str] = field(default_factory=list)
    edge_type_names: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.node_type.numel()

    @property
    def n_node_types(self) -> int:
        return len(self.node_type_names)

    @property
    def n_edge_types(self) -> int:
        return len(self.edge_type_names)


# ── dataset construction ────────────────────────────────────────────────────


def build_dataset(
    G,
    split: str = "temporal",
    test_frac: float = 0.20,
    unseen_node_frac: float = 0.20,
    order_mode: str = "regression",
    n_order_bins: int = 20,
    embed_dim_fallback: int = 64,
    compartment_embeddings: Optional[dict] = None,
    compartment_dim: int = 32,
    force_unseen: Optional[list] = None,
    time_target: str = "min_max",
    seed: int = 0,
) -> FISHData:
    """
    Convert a featurised FISH graph to tensors.

    Parameters
    ----------
    G : networkx.DiGraph
        Output of ``parse_biopax_into_networkx`` after ``NodeFeaturiser.featurise``.
        Each node must have a ``type`` attribute and a ``feature`` ndarray.
    split : 'temporal' | 'semi_inductive'
        See top-level docstring for the two split semantics.
    test_frac : float
        Used by 'temporal'. Fraction of edges (by time) held out.
    unseen_node_frac : float
        Used by 'semi_inductive'. Fraction of nodes marked unseen.
    order_mode : 'regression' | 'corn' | 'coral'
    n_order_bins : int
    embed_dim_fallback : int
        Used only if a node type has no real features.
    compartment_embeddings : dict, optional
        See ``compartment_dim`` below. When provided, a per-node compartment
        feature matrix is built and stored on FISHData.
    compartment_dim : int
        Dimensionality of the compartment feature.
    force_unseen : list of node names, optional
        Used by 'semi_inductive'. Node names that MUST be in the unseen set,
        unioned with the random sample. Used by case-study code that needs a
        specific receptor to be held out.
    seed : int
        For reproducible node sampling in the semi-inductive split.
    """
    if split not in {"temporal", "semi_inductive"}:
        raise ValueError(f"split must be 'temporal' or 'semi_inductive', got {split!r}")
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_to_id = {n: i for i, n in enumerate(nodes)}

    # Node-type vocab (preserve insertion order).
    seen_types: dict[str, None] = {}
    for n in nodes:
        seen_types[G.nodes[n].get("type", "other")] = None
    node_type_names = list(seen_types.keys())
    type_to_idx = {t: i for i, t in enumerate(node_type_names)}

    # Per-type feature matrices, with per-node intra-type row indexing.
    feats_by_type: dict[int, list[np.ndarray]] = {i: [] for i in range(len(node_type_names))}
    intratype_idx = np.zeros(n_nodes, dtype=np.int64)
    node_type_arr = np.zeros(n_nodes, dtype=np.int64)
    for nid, name in enumerate(nodes):
        t = G.nodes[name].get("type", "other")
        ti = type_to_idx[t]
        node_type_arr[nid] = ti
        feat = G.nodes[name].get("feature")
        if feat is None:
            feat = np.zeros(embed_dim_fallback, dtype=np.float32)
        intratype_idx[nid] = len(feats_by_type[ti])
        feats_by_type[ti].append(np.asarray(feat, dtype=np.float32))

    feats_tensors: dict[int, torch.Tensor] = {}
    for ti, lst in feats_by_type.items():
        if not lst:
            # No nodes of this type — empty tensor with sane dim.
            feats_tensors[ti] = torch.zeros((0, embed_dim_fallback), dtype=torch.float32)
            continue
        # All vectors of one type should already share a dim (e.g. all
        # proteins are ESM-2 320). If not, pad to the type's max within type.
        dims = {f.shape[0] for f in lst}
        if len(dims) > 1:
            max_d = max(dims)
            lst = [np.pad(f, (0, max_d - f.shape[0])) for f in lst]
        feats_tensors[ti] = torch.from_numpy(np.stack(lst))

    # Edge tensors.
    seen_etypes: dict[str, None] = {}
    for _, _, d in G.edges(data=True):
        seen_etypes[d.get("type", "other")] = None
    edge_type_names = list(seen_etypes.keys())
    etype_to_idx = {t: i for i, t in enumerate(edge_type_names)}

    src, dst, etypes, times = [], [], [], []
    for u, v, d in G.edges(data=True):
        src.append(node_to_id[u])
        dst.append(node_to_id[v])
        etypes.append(etype_to_idx[d.get("type", "other")])
        t = d.get("time")
        # Some edges lack a step — bucket those to the maximum so they end
        # up in the test split (or drop them — here we drop, since edges
        # without time can't be ordered).
        times.append(float(t) if isinstance(t, (int, float)) else float("nan"))

    src = np.asarray(src); dst = np.asarray(dst)
    etypes = np.asarray(etypes); times = np.asarray(times, dtype=np.float64)
    keep = ~np.isnan(times)
    if not keep.all():
        print(f"[build_dataset] dropping {(~keep).sum()} edges with no `time`")
    src, dst, etypes, times = src[keep], dst[keep], etypes[keep], times[keep]

    # Build train/test masks based on the chosen split.
    if split == "temporal":
        sorted_steps = np.sort(np.unique(times))
        target = 1.0 - test_frac
        cut_idx = int(np.searchsorted(
            np.array([(times <= s).mean() for s in sorted_steps]), target
        ))
        cut_idx = min(cut_idx, len(sorted_steps) - 1)
        cut_time = sorted_steps[cut_idx]
        train_mask_np = times <= cut_time
        test_mask_np = ~train_mask_np
        inductive_mask_np = np.zeros_like(train_mask_np)  # unused
    else:  # 'semi_inductive'
        # Hold out a random fraction of nodes as unseen. Training edges are
        # those with both endpoints seen. Test edges are bridge edges with
        # exactly one seen endpoint; edges with neither endpoint seen are
        # discarded since they cannot be scored against any training context.
        n_total_nodes = len(nodes)
        n_unseen = int(round(unseen_node_frac * n_total_nodes))
        perm = rng.permutation(n_total_nodes)
        unseen_ids = set(perm[:n_unseen].tolist())

        # Union in any explicitly-requested nodes (e.g. case-study targets).
        # If a name doesn't resolve, raise — silently dropping a case-study
        # target would be a hard-to-detect bug.
        if force_unseen:
            missing = [n for n in force_unseen if n not in node_to_id]
            if missing:
                raise KeyError(
                    f"force_unseen nodes not found in graph: {missing[:5]}"
                    + (f" (and {len(missing) - 5} more)" if len(missing) > 5 else "")
                )
            for n in force_unseen:
                unseen_ids.add(node_to_id[n])

        src_seen = np.array([s not in unseen_ids for s in src])
        dst_seen = np.array([d not in unseen_ids for d in dst])
        both_seen = src_seen & dst_seen
        one_seen = src_seen ^ dst_seen
        neither_seen = (~src_seen) & (~dst_seen)
        # Train = both endpoints seen; semi-inductive test = exactly one
        # endpoint unseen; inductive test = neither endpoint seen (these
        # edges are scored from node features alone — no graph context).
        train_mask_np = both_seen
        test_mask_np = one_seen
        inductive_mask_np = neither_seen
        if neither_seen.any():
            print(f"[build_dataset] semi-inductive: {int(neither_seen.sum())} "
                  f"edges with both endpoints unseen are promoted to the "
                  f"inductive test set (no graph context — feature-only).")

    # Ordinal bins (always computed; the head may or may not use them).
    bins = None
    if order_mode in ("corn", "coral"):
        # Quantile binning on training-time values, then assign all edges.
        qs = np.linspace(0, 1, n_order_bins + 1)[1:-1]
        edges_q = np.quantile(times[train_mask_np], qs)
        bins = np.digitize(times, edges_q)        # (E,) in [0, n_bins-1]

    # Regression target for the order head. Three transforms, all map raw
    # pathway-step times into [0, 1]; the regression head trains under MSE
    # against this target.
    #   min_max     — linear: (t - t_min) / (t_max - t_min). Default. Heavy
    #                 tails (late steps with few events) get most of the
    #                 weight on the high end.
    #   log_min_max — apply log(1+t) first, then min-max. Compresses the
    #                 tail; early dense region gets more weight.
    #   rank        — rank-based: target = rank(t) / (N-1). Marginal
    #                 distribution is uniform regardless of original shape.
    #                 Useful if step-count distribution is highly skewed.
    train_times = times[train_mask_np]
    if time_target == "min_max":
        t_min, t_max = float(times.min()), float(times.max())
        span = max(t_max - t_min, 1.0)
        order_target_np = ((times - t_min) / span).astype(np.float32)
    elif time_target == "log_min_max":
        log_t = np.log1p(times - times.min())  # shift to >= 0 first
        t_min, t_max = float(log_t.min()), float(log_t.max())
        span = max(t_max - t_min, 1.0)
        order_target_np = ((log_t - t_min) / span).astype(np.float32)
    elif time_target == "rank":
        # rank(t) / (N-1), using the *all-edges* time ranking (so train and
        # test see the same global ranking). argsort gives the inverse.
        order_ranks = np.argsort(np.argsort(times))
        denom = max(len(times) - 1, 1)
        order_target_np = (order_ranks / denom).astype(np.float32)
    else:
        raise ValueError(
            f"time_target must be 'min_max' | 'log_min_max' | 'rank', "
            f"got {time_target!r}"
        )

    # Build per-node compartment feature matrix from the parser-written
    # `cellularLocation` attribute. The attribute is a dict like
    # {"common_name": "cytosol", "xref": "GO:0005829", ...} or None.
    # Resolution order for lookup key: GO ID (precise, GO2Vec-compatible)
    # -> common name (fallback) -> '_unknown_' (catch-all bucket).
    compartment_feat_tensor = None
    compartment_feat_dim = 0
    compartment_vocab: list[str] = []
    if compartment_embeddings is not None:
        # Pre-supplied vector embeddings (e.g. GO2Vec). Build a per-node
        # matrix from them.
        emb_dim = len(next(iter(compartment_embeddings.values())))
        unknown_vec = np.zeros(emb_dim, dtype=np.float32)
        per_node = np.zeros((len(nodes), emb_dim), dtype=np.float32)
        for nid, name in enumerate(nodes):
            loc = G.nodes[name].get("cellularLocation")
            if isinstance(loc, dict):
                # Try common GO-style xref fields, falling back to name.
                key = (
                    loc.get("xref")
                    or loc.get("GO")
                    or loc.get("go")
                    or loc.get("common_name")
                )
            else:
                key = None
            if key in compartment_embeddings:
                per_node[nid] = compartment_embeddings[key]
            else:
                per_node[nid] = unknown_vec
        compartment_feat_tensor = torch.from_numpy(per_node)
        compartment_feat_dim = emb_dim
    else:
        # No pre-supplied embeddings: build a vocabulary so the model can
        # use a learned embedding table. Encode each node as a one-hot in
        # the vocab; the model concatenates this one-hot and lets the
        # input projection learn the embedding rows implicitly.
        # (Cheaper than threading a separate nn.Embedding through the model.)
        for name in nodes:
            loc = G.nodes[name].get("cellularLocation")
            if isinstance(loc, dict):
                key = (
                    loc.get("xref")
                    or loc.get("common_name")
                    or "_unknown_"
                )
            else:
                key = "_unknown_"
            if key not in compartment_vocab:
                compartment_vocab.append(key)
        n_compartments = len(compartment_vocab)
        comp_to_idx = {c: i for i, c in enumerate(compartment_vocab)}
        per_node = np.zeros((len(nodes), n_compartments), dtype=np.float32)
        for nid, name in enumerate(nodes):
            loc = G.nodes[name].get("cellularLocation")
            if isinstance(loc, dict):
                key = loc.get("xref") or loc.get("common_name") or "_unknown_"
            else:
                key = "_unknown_"
            per_node[nid, comp_to_idx[key]] = 1.0
        compartment_feat_tensor = torch.from_numpy(per_node)
        compartment_feat_dim = n_compartments

    return FISHData(
        feats_by_type=feats_tensors,
        node_type=torch.from_numpy(node_type_arr),
        node_intratype_idx=torch.from_numpy(intratype_idx),
        edge_index=torch.from_numpy(np.stack([src[train_mask_np], dst[train_mask_np]])),
        edge_type=torch.from_numpy(etypes[train_mask_np]),
        edge_time=torch.from_numpy(times[train_mask_np].astype(np.float32)),
        all_src=torch.from_numpy(src),
        all_dst=torch.from_numpy(dst),
        all_type=torch.from_numpy(etypes),
        all_time=torch.from_numpy(times.astype(np.float32)),
        train_mask=torch.from_numpy(train_mask_np),
        test_mask=torch.from_numpy(test_mask_np),
        inductive_mask=torch.from_numpy(inductive_mask_np),
        order_bin=torch.from_numpy(bins.astype(np.int64)) if bins is not None else None,
        order_target=torch.from_numpy(order_target_np),
        compartment_feat=compartment_feat_tensor,
        compartment_dim=compartment_feat_dim,
        compartment_vocab=compartment_vocab,
        node_type_names=node_type_names,
        edge_type_names=edge_type_names,
    )


# ── model ───────────────────────────────────────────────────────────────────


class FISHRGCN(nn.Module):
    """
    Shared RGCN encoder, three task heads.

    Order head is one of:
      * 'regression' — 1-d linear, MSE on (time - min) / (max - min).
      * 'corn'       — (n_bins - 1) binary outputs over quantile bins,
                       trained with the CORN consistent loss.
    """

    def __init__(
        self,
        data: FISHData,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        order_mode: str = "regression",
        n_order_bins: int = 20,
        n_negatives: int = 5,
        time_encoding: bool = False,
        time_dim: int = 64,
        architecture: str = "rgcn",
        rgat_heads: int = 1,
        use_compartment: bool = False,
        smart_negatives: bool = False,
    ):
        super().__init__()
        if architecture not in ("rgcn", "rgat"):
            raise ValueError(
                f"architecture must be 'rgcn' or 'rgat', not {architecture!r}"
            )
        self.order_mode = order_mode
        self.n_order_bins = n_order_bins
        self.n_negatives = n_negatives
        self.time_encoding = time_encoding
        self.time_dim = time_dim
        self.architecture = architecture
        self.use_compartment = use_compartment
        self.smart_negatives = smart_negatives

        # Per-type input projections. When use_compartment=True, each per-type
        # native feature vector is concatenated with the per-node compartment
        # embedding (which has the same dim for every node) before projection,
        # so the projection's in_dim grows by data.compartment_dim.
        compartment_extra = data.compartment_dim if use_compartment else 0
        if use_compartment and data.compartment_dim == 0:
            raise ValueError(
                "use_compartment=True but the dataset has no compartment "
                "features (data.compartment_dim == 0). Pass compartment "
                "embeddings to build_dataset."
            )

        self.input_projs = nn.ModuleDict()
        for ti, tensor in data.feats_by_type.items():
            in_dim = tensor.shape[1] + compartment_extra
            self.input_projs[str(ti)] = nn.Linear(in_dim, hidden)

        # Time encoder is shared across layers if time_encoding=True.
        self.time_encoder = TimeEncode(time_dim) if time_encoding else None

        # Encoder convs. Four cases via (architecture, time_encoding):
        #   rgcn / no time : RGCNConv
        #   rgcn / time    : RGCNTimeConv (custom; time folded into messages)
        #   rgat / no time : RGATConv
        #   rgat / time    : RGATConv with edge_dim=time_dim (uses edge_attr)
        if architecture == "rgcn":
            if time_encoding:
                self.convs = nn.ModuleList([
                    RGCNTimeConv(hidden, hidden, num_relations=data.n_edge_types,
                                 time_dim=time_dim)
                    for _ in range(n_layers)
                ])
            else:
                self.convs = nn.ModuleList([
                    RGCNConv(hidden, hidden, num_relations=data.n_edge_types)
                    for _ in range(n_layers)
                ])
        else:  # 'rgat'
            # Notes on the settings below:
            #   * dropout=0.0 inside RGATConv — that arg is *attention* dropout
            #     which compounds with the F.dropout we apply on the encoder
            #     output. Stacking both was producing a degenerate existence
            #     head (AUC ~ 0.52 with std 0.004 across 60 runs).
            #   * heads=4, concat=False — single-head attention collapses on
            #     heavy-tailed hubs (ATP has degree 753 in Immune); averaging
            #     across heads gives the encoder enough capacity without
            #     inflating the output dim.
            edge_dim_arg = time_dim if time_encoding else None
            effective_heads = max(rgat_heads, 4)
            self.convs = nn.ModuleList([
                RGATConv(
                    hidden, hidden,
                    num_relations=data.n_edge_types,
                    heads=effective_heads, concat=False,
                    edge_dim=edge_dim_arg,
                    dropout=0.0,
                )
                for _ in range(n_layers)
            ])
        self.dropout = dropout

        # Three task heads, all operating on concat(h_u, h_v).
        head_in = 2 * hidden
        self.head_exist = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.head_type = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, data.n_edge_types),
        )
        if order_mode == "regression":
            self.head_order = nn.Sequential(
                nn.Linear(head_in, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )
        elif order_mode in ("corn", "coral"):
            # Both CORN and CORAL produce (K-1) ordinal binary outputs.
            self.head_order = nn.Sequential(
                nn.Linear(head_in, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, n_order_bins - 1),
            )
        else:
            raise ValueError(
                f"order_mode must be 'regression', 'corn', or 'coral', "
                f"not {order_mode!r}"
            )

    # ── encoder ──────────────────────────────────────────────────────────────

    def encode(self, data: FISHData) -> torch.Tensor:
        """Returns node embeddings of shape (N, hidden)."""
        device = next(self.parameters()).device

        # Per-type input: optionally concatenate the per-node compartment
        # embedding with the type's native features before projection. This is
        # the GNN analogue of the REM '+ compartment' ablation in the paper.
        if self.use_compartment:
            compartment = data.compartment_feat.to(device)  # (N, compartment_dim)
        projected_per_type = []
        for ti in range(data.n_node_types):
            tensor = data.feats_by_type[ti].to(device)
            if self.use_compartment:
                # Gather the compartment rows for this type's nodes, in the
                # same intra-type ordering as the feature matrix.
                type_node_ids = (data.node_type == ti).nonzero(as_tuple=False).squeeze(1)
                # Sort by intra-type index so rows align with tensor.
                order = data.node_intratype_idx[type_node_ids].argsort()
                type_node_ids = type_node_ids[order]
                comp_block = compartment[type_node_ids]  # (n_ti, compartment_dim)
                tensor = torch.cat([tensor, comp_block], dim=1)
            projected_per_type.append(self.input_projs[str(ti)](tensor))

        h = torch.zeros(
            data.n_nodes,
            projected_per_type[0].shape[1] if projected_per_type else 0,
            device=device,
        )
        for ti in range(data.n_node_types):
            mask = (data.node_type == ti)
            if mask.any():
                idx = data.node_intratype_idx[mask]
                h[mask] = projected_per_type[ti][idx]

        edge_index = data.edge_index.to(device)
        edge_type = data.edge_type.to(device)

        # Compute time features once if needed; the four (arch, time) cases
        # below dispatch to the right conv signature.
        edge_time_feat = None
        if self.time_encoding:
            # Normalise time to [0, 1] before encoding so the GraphMixer
            # log-spaced frequencies span a meaningful range.
            t = data.edge_time.to(device)
            t_max = float(data.all_time.max())
            t_norm = t / max(t_max, 1.0)
            edge_time_feat = self.time_encoder(t_norm)  # (E_tr, time_dim)

        for conv in self.convs:
            if self.architecture == "rgcn":
                if self.time_encoding:
                    h = conv(h, edge_index, edge_type, edge_time_feat)
                else:
                    h = conv(h, edge_index, edge_type)
            else:  # 'rgat'
                if self.time_encoding:
                    # RGATConv consumes edge features via the edge_attr arg
                    # when constructed with edge_dim=time_dim.
                    h = conv(h, edge_index, edge_type=edge_type,
                             edge_attr=edge_time_feat)
                else:
                    h = conv(h, edge_index, edge_type=edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    # ── heads ────────────────────────────────────────────────────────────────

    def _pair_repr(self, h, src, dst):
        return torch.cat([h[src], h[dst]], dim=-1)

    def existence_logit(self, h, src, dst):
        return self.head_exist(self._pair_repr(h, src, dst)).squeeze(-1)

    def type_logits(self, h, src, dst):
        return self.head_type(self._pair_repr(h, src, dst))

    def order_output(self, h, src, dst):
        return self.head_order(self._pair_repr(h, src, dst))


# ── training ────────────────────────────────────────────────────────────────


def _sample_negatives(data: FISHData, src: torch.Tensor, n_negatives: int,
                      n_nodes: int, device) -> torch.Tensor:
    """Uniformly sample negative destination nodes for each src."""
    return torch.randint(0, n_nodes, (src.numel(), n_negatives), device=device)


def _build_train_negative_state(data: FISHData, device) -> dict:
    """
    Precompute structures for training-time smart negatives.

    Stores per-node-type id lists as a single padded tensor with offset
    pointers, so we can sample from a node-type's id pool in pure
    tensor ops (no Python loop, GPU-friendly).
    """
    nt = data.node_type.cpu().numpy()
    # Per-type sorted lists of node ids.
    type_to_ids: list[np.ndarray] = []
    max_len = 0
    for t in range(data.n_node_types):
        ids = np.where(nt == t)[0].astype(np.int64)
        type_to_ids.append(ids)
        max_len = max(max_len, len(ids))

    # Pad each type's id-list to max_len so we have a (n_types, max_len)
    # tensor we can gather from. Padding value is 0 (a valid node id);
    # we use a separate (n_types,) lengths tensor to mask out the padded
    # slots when sampling.
    n_types = data.n_node_types
    pool_tensor = torch.zeros((n_types, max_len), dtype=torch.long, device=device)
    pool_lens = torch.zeros(n_types, dtype=torch.long, device=device)
    for t, ids in enumerate(type_to_ids):
        pool_tensor[t, :len(ids)] = torch.from_numpy(ids).to(device)
        pool_lens[t] = len(ids)

    return {
        "pool_tensor": pool_tensor,   # (n_node_types, max_pool_size)
        "pool_lens": pool_lens,        # (n_node_types,)
        "node_type": data.node_type.to(device),  # (N,)
    }


def _sample_train_smart_negatives(
    src: torch.Tensor, dst: torch.Tensor, train_neg_state: dict,
    n_negatives: int,
) -> torch.Tensor:
    """
    Type-matched negative sampling for training. Fully vectorised:
    for each positive (u, v), sample n_negatives random nodes from the
    pool of nodes whose node-type matches v.

    Does NOT exclude u's known training partners (would require a per-row
    rejection loop that kills training throughput). The fraction of
    accidentally-sampled true partners is bounded by mean_partners/|type|,
    typically <1% on biological graphs; we accept this as label noise
    in exchange for ~100x speedup over filtered sampling.
    """
    pool_tensor = train_neg_state["pool_tensor"]  # (n_types, max_len)
    pool_lens = train_neg_state["pool_lens"]      # (n_types,)
    node_type = train_neg_state["node_type"]       # (N,)

    n_pos = dst.numel()
    # For each positive, find v's node type and its pool length.
    dst_types = node_type[dst]                     # (n_pos,)
    dst_pool_lens = pool_lens[dst_types]            # (n_pos,)

    # Sample random indices in [0, pool_lens[type]) for each positive,
    # for each of n_negatives draws. Shape: (n_pos, n_negatives).
    rand01 = torch.rand(n_pos, n_negatives, device=src.device)
    sampled_idx = (rand01 * dst_pool_lens.unsqueeze(1).float()).long()
    sampled_idx = sampled_idx.clamp(max=pool_tensor.shape[1] - 1)

    # Gather from the per-type pool. Need to expand dst_types for gathering.
    # pool_tensor[dst_types[i], sampled_idx[i, j]] -> negative id
    type_rep = dst_types.unsqueeze(1).expand(n_pos, n_negatives)  # (n_pos, n_neg)
    flat_pool = pool_tensor.flatten()                              # (n_types*max_len,)
    flat_idx = type_rep * pool_tensor.shape[1] + sampled_idx       # global pool idx
    neg_ids = flat_pool[flat_idx]                                  # (n_pos, n_neg)
    return neg_ids


def train(
    model: FISHRGCN,
    data: FISHData,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: Optional[str] = None,
    log_every: int = 20,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Regression target for the order head is precomputed in build_dataset
    # under the chosen time_target transform (min_max | log_min_max | rank).
    order_target_tr = data.order_target[data.train_mask].to(device)

    src_tr = data.all_src[data.train_mask].to(device)
    dst_tr = data.all_dst[data.train_mask].to(device)
    typ_tr = data.all_type[data.train_mask].to(device)
    bin_tr = (data.order_bin[data.train_mask].to(device)
              if data.order_bin is not None else None)

    # Precompute CORAL extended-binary levels once outside the training loop.
    # coral-pytorch's `levels_from_labelbatch` does a Python for-loop over
    # every label, which costs ~13 min on Immune at 200 epochs; the actual
    # operation is a one-line broadcast. Since bin labels are constant
    # across epochs we only build this matrix once.
    coral_levels = None
    if model.order_mode == "coral" and bin_tr is not None:
        thresholds = torch.arange(model.n_order_bins - 1, device=device)
        coral_levels = (bin_tr.unsqueeze(1) > thresholds.unsqueeze(0)).float()

    # Training-time negative sampler. Precomputed once. When
    # model.smart_negatives is True, samples come from v's node-type pool
    # (without filtering for known partners, for throughput). Otherwise
    # uniformly random over all nodes.
    train_neg_state = None
    if model.smart_negatives:
        train_neg_state = _build_train_negative_state(data, device)

    history = {"exist": [], "type": [], "order": [], "total": []}

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        h = model.encode(data)

        # 1) Existence: positives + K negatives per positive.
        pos_logit = model.existence_logit(h, src_tr, dst_tr)
        if model.smart_negatives and train_neg_state is not None:
            negs = _sample_train_smart_negatives(
                src_tr, dst_tr, train_neg_state, model.n_negatives,
            )
        else:
            negs = _sample_negatives(
                data, src_tr, model.n_negatives, data.n_nodes, device,
            )
        src_rep = src_tr.repeat_interleave(model.n_negatives)
        neg_logit = model.existence_logit(h, src_rep, negs.reshape(-1))
        loss_exist = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logit, neg_logit]),
            torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)]),
        )

        # 2) Type: cross-entropy on positives only.
        type_lgt = model.type_logits(h, src_tr, dst_tr)
        loss_type = F.cross_entropy(type_lgt, typ_tr)

        # 3) Order: regression, CORN, or CORAL.
        out_order = model.order_output(h, src_tr, dst_tr)
        if model.order_mode == "regression":
            loss_order = F.mse_loss(out_order.squeeze(-1), order_target_tr)
        elif model.order_mode == "corn":
            loss_order = _corn_loss(out_order, bin_tr, model.n_order_bins)
        else:  # 'coral'
            loss_order = _coral_loss(out_order, coral_levels)

        a, b, c = loss_weights
        total = a * loss_exist + b * loss_type + c * loss_order
        total.backward()
        opt.step()

        history["exist"].append(loss_exist.item())
        history["type"].append(loss_type.item())
        history["order"].append(loss_order.item())
        history["total"].append(total.item())

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"epoch {epoch:4d} | total {total.item():.4f} "
                f"| exist {loss_exist.item():.4f} "
                f"| type {loss_type.item():.4f} "
                f"| order {loss_order.item():.4f}"
            )

    return history


# ── evaluation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def _build_negative_sampler_state(data: FISHData) -> dict:
    """
    Precompute structures needed for smart negative sampling.

    The smart sampler tests "could this *specific* node have been the partner
    instead of v?" — i.e. discrimination between v and other plausible
    candidates. For that to be a fair negative-sampling protocol, the sampled
    candidates must NOT themselves be known training partners of u (otherwise
    we're sampling true edges as negatives, which depresses AUC arbitrarily).

    State produced:
        nodes_by_type        — per node_type id, an array of node ids of
                                that type. Used so we can sample candidates
                                matching v's node type ("could it have been
                                a protein instead?").
        train_partners_of_u  — per source node id, the set of destination
                                ids it has any training edge to. Used to
                                EXCLUDE these from the negative sample.
    """
    tr_src = data.all_src[data.train_mask].cpu().numpy()
    tr_dst = data.all_dst[data.train_mask].cpu().numpy()

    nodes_by_type: dict[int, np.ndarray] = {}
    nt = data.node_type.cpu().numpy()
    for t in range(data.n_node_types):
        nodes_by_type[t] = np.where(nt == t)[0].astype(np.int64)

    train_partners_of_u: dict[int, set] = {}
    for u, v in zip(tr_src, tr_dst):
        train_partners_of_u.setdefault(int(u), set()).add(int(v))

    return {
        "nodes_by_type": nodes_by_type,
        "train_partners_of_u": train_partners_of_u,
    }


def _sample_smart_negatives(
    src_np: np.ndarray, dst_np: np.ndarray, typ_np: np.ndarray,
    state: dict, n_neg_per_pos: int, n_nodes: int, rng: np.random.Generator,
    node_type_of_dst: np.ndarray,
) -> np.ndarray:
    """
    Sample type-matched, partner-excluded negatives.

    For positive (u, v) where v has node-type T:
      negatives are drawn from {n : type(n) == T} minus {u's training partners}.

    This tests "could a different node of the same type have been the partner
    instead?" — discrimination between plausible candidates, without falsely
    sampling u's other true partners as negatives.
    """
    n_pos = len(src_np)
    nodes_by_type = state["nodes_by_type"]
    partners = state["train_partners_of_u"]

    neg = np.empty((n_pos, n_neg_per_pos), dtype=np.int64)
    for i in range(n_pos):
        u = int(src_np[i])
        v = int(dst_np[i])
        dst_type = int(node_type_of_dst[v])

        type_pool = nodes_by_type[dst_type]
        u_partners = partners.get(u, set())
        # Exclude v itself and any other known partner of u (for any edge type).
        forbidden = u_partners | {v}
        candidate_mask = ~np.isin(type_pool, list(forbidden), assume_unique=True)
        pool = type_pool[candidate_mask]

        if len(pool) >= n_neg_per_pos:
            neg[i] = rng.choice(pool, size=n_neg_per_pos, replace=False)
        elif len(pool) > 0:
            neg[i] = rng.choice(pool, size=n_neg_per_pos, replace=True)
        else:
            # Pathological: no type-matched non-partners exist. Fall back to
            # uniform random, just so we return something.
            neg[i] = rng.integers(0, n_nodes, size=n_neg_per_pos)

    return neg


@torch.no_grad()
def _hits_at_k(
    model: FISHRGCN,
    h: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    candidate_pool: torch.Tensor,
    n_nodes_total: int,
    ks: tuple[int, ...] = (1, 10, 100),
    dyads_per_chunk: int = 1_000_000,
) -> dict[int, float]:
    """
    Hits@K over a candidate pool. For each positive (src[i], dst[i]),
    score against every candidate in ``candidate_pool`` and check whether
    the true dst is ranked in the top K by existence-head score.

    Performance notes:
      * The candidate embeddings ``h[candidate_pool]`` are computed once
        and reused across positive chunks (avoids redundant gathers).
      * The src embeddings within a chunk are broadcast against the cached
        candidate embeddings, then run through the MLP heads. This keeps
        the gather pattern out of the hot loop.
      * dyads_per_chunk=1M is the CUDA-friendly default; reduce to ~250k
        if running on MPS or 8GB GPUs where the chunk's ~1GB activations
        would force swapping.

    Tie-breaking: pessimistic. Rank = #candidates with strictly higher score
    plus 1; ties count against the positive.
    """
    n_pos = src.numel()
    n_cand = candidate_pool.numel()
    if n_pos == 0 or n_cand == 0:
        return {k: float("nan") for k in ks}

    # Precompute candidate embeddings once. Shape (n_cand, hidden).
    cand_h = h[candidate_pool]

    # Global node id -> column-in-this-pool lookup, built once.
    cand_to_col = -torch.ones(n_nodes_total, dtype=torch.long, device=h.device)
    cand_to_col[candidate_pool] = torch.arange(n_cand, device=h.device)

    batch_size = max(1, dyads_per_chunk // n_cand)
    hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    scored = 0

    for start in range(0, n_pos, batch_size):
        end = min(start + batch_size, n_pos)
        sub_src = src[start:end]
        sub_dst = dst[start:end]
        B = sub_src.numel()

        # Embeddings for this chunk's sources. (B, hidden).
        src_h = h[sub_src]
        # Expand src_h to (B, n_cand, hidden) and broadcast cand_h.
        src_exp = src_h.unsqueeze(1).expand(B, n_cand, src_h.shape[-1])
        cand_exp = cand_h.unsqueeze(0).expand(B, n_cand, cand_h.shape[-1])
        pair = torch.cat([src_exp, cand_exp], dim=-1)        # (B, n_cand, 2H)
        logits = model.head_exist(pair).squeeze(-1)          # (B, n_cand)

        dst_cols = cand_to_col[sub_dst]      # (B,), -1 where dst isn't in pool
        valid = dst_cols >= 0
        if not valid.any():
            continue

        v_rows = torch.arange(B, device=logits.device)[valid]
        true_scores = logits[v_rows, dst_cols[valid]]
        ranks = (logits[v_rows] > true_scores.unsqueeze(1)).sum(dim=1) + 1
        for k in ks:
            hits[k] += int((ranks <= k).sum().item())
        # MRR contribution from this chunk: sum of 1/rank for valid rows.
        reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
        scored += int(valid.sum().item())

    if scored == 0:
        out: dict = {k: float("nan") for k in ks}
        out["mrr"] = float("nan")
        return out
    out = {k: hits[k] / scored for k in ks}
    out["mrr"] = reciprocal_rank_sum / scored
    return out


def _score_edge_set(
    model: FISHRGCN, data: FISHData, h: torch.Tensor, mask: torch.Tensor,
    n_neg_per_pos: int, device, seed: int,
    neg_state: Optional[dict] = None,
    compute_hits: bool = False,
    hits_ks: tuple[int, ...] = (1, 10, 100),
    hits_batch: int = 2048,
    hits_fn: Optional[Callable] = None,
) -> dict:
    """
    Compute the three task metrics on the edges selected by ``mask``.

    Existence is scored two ways when ``neg_state`` is provided:
      * exist_auc            — uniform random destinations (the easy default)
      * exist_auc_smart      — type-matched, partner-excluded negatives

    When ``compute_hits=True``, additionally computes Hits@K against both
    candidate pools (all nodes / type-matched), reported as ``hits_at_{k}``
    and ``hits_at_{k}_type`` for each k in ``hits_ks``. This is the
    KG-completion-style retrieval metric: for each positive (u, v), rank
    the true v against the entire candidate pool and check whether v lands
    in the top K. Expensive (~5000 positives x 7000 candidates) so it's
    off by default.

    ``hits_fn`` overrides the Hits@K implementation (default ``_hits_at_k``,
    which assumes ``h`` is a (N, hidden) per-node embedding table). Models
    whose ``existence_logit`` doesn't follow that contract — e.g. FISHSTHN,
    which scores each (src, dst) query via its own subgraph — pass a
    compatible replacement with the same signature as ``_hits_at_k``.

    Type and order use positive edges only and are unaffected by the
    negative-sampling or hits choices.
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    hits_fn = hits_fn or _hits_at_k

    if int(mask.sum()) == 0:
        empty = {
            "exist_auc": float("nan"),
            "exist_auc_smart": float("nan"),
            "type_macro_auc": float("nan"),
            "type_top1": float("nan"),
            "type_majority_baseline": float("nan"),
            "type_classes_in_test": [],
            "type_classes_missing_from_test": list(range(data.n_edge_types)),
            "order_spearman": float("nan"),
            "order_pairwise_acc": float("nan"),
            "n_edges": 0,
        }
        for k in hits_ks:
            empty[f"hits_at_{k}"] = float("nan")
            empty[f"hits_at_{k}_type"] = float("nan")
        empty["mrr"] = float("nan")
        empty["mrr_type"] = float("nan")
        return empty

    src = data.all_src[mask].to(device)
    dst = data.all_dst[mask].to(device)
    typ = data.all_type[mask].to(device)
    times = data.all_time[mask].to(device)

    # Existence (random): pos vs n_neg_per_pos uniform-random negatives.
    g = torch.Generator(device=device).manual_seed(seed)
    pos_logit = model.existence_logit(h, src, dst)
    src_rep = src.repeat_interleave(n_neg_per_pos)
    neg_dst_random = torch.randint(0, data.n_nodes, (src.numel() * n_neg_per_pos,),
                                   generator=g, device=device)
    neg_logit_random = model.existence_logit(h, src_rep, neg_dst_random)
    y_true_r = torch.cat(
        [torch.ones_like(pos_logit), torch.zeros_like(neg_logit_random)]
    ).cpu().numpy()
    y_score_r = torch.cat([pos_logit, neg_logit_random]).cpu().numpy()
    exist_auc = float(roc_auc_score(y_true_r, y_score_r))

    # Existence (smart): type-matched, partner-excluded negatives.
    exist_auc_smart = float("nan")
    if neg_state is not None:
        rng = np.random.default_rng(seed + 12345)
        node_type_np = data.node_type.cpu().numpy()
        smart_neg_np = _sample_smart_negatives(
            src.cpu().numpy(), dst.cpu().numpy(), typ.cpu().numpy(),
            neg_state, n_neg_per_pos, data.n_nodes, rng,
            node_type_of_dst=node_type_np,
        )
        smart_neg = torch.from_numpy(smart_neg_np).to(device).reshape(-1)
        neg_logit_smart = model.existence_logit(h, src_rep, smart_neg)
        y_true_s = torch.cat(
            [torch.ones_like(pos_logit), torch.zeros_like(neg_logit_smart)]
        ).cpu().numpy()
        y_score_s = torch.cat([pos_logit, neg_logit_smart]).cpu().numpy()
        exist_auc_smart = float(roc_auc_score(y_true_s, y_score_s))

    # Type: macro-AUC + top-1.
    type_logits = model.type_logits(h, src, dst).cpu().numpy()
    type_true = typ.cpu().numpy()
    present_classes = np.unique(type_true)
    type_auc = float("nan")
    type_auc_classes_used: list[int] = []
    if len(present_classes) >= 2:
        try:
            sub_logits = type_logits[:, present_classes]
            oh = (type_true[:, None] == present_classes[None, :]).astype(int)
            type_auc = float(roc_auc_score(
                oh, sub_logits, average="macro", multi_class="ovr"
            ))
            type_auc_classes_used = [int(c) for c in present_classes]
        except ValueError:
            type_auc = float("nan")
    type_top1 = float((type_logits.argmax(axis=1) == type_true).mean())
    if len(type_true) > 0:
        counts = np.bincount(type_true, minlength=data.n_edge_types)
        type_majority = float(counts.max() / counts.sum())
    else:
        type_majority = float("nan")

    # Order.
    out_order = model.order_output(h, src, dst)
    if model.order_mode == "regression":
        score = out_order.squeeze(-1).cpu().numpy()
    elif model.order_mode == "corn":
        score = _corn_label_from_logits(out_order).cpu().numpy().astype(float)
    else:  # 'coral'
        probs = torch.sigmoid(out_order)
        score = (probs > 0.5).sum(dim=-1).cpu().numpy().astype(float)
    target = times.cpu().numpy()
    rho = float(spearmanr(score, target).statistic) if len(score) > 1 else float("nan")

    rng = np.random.default_rng(seed)
    n_pairs = min(50_000, len(score) * (len(score) - 1) // 2)
    if n_pairs >= 1:
        i = rng.integers(0, len(score), n_pairs)
        j = rng.integers(0, len(score), n_pairs)
        keep = (i != j) & (target[i] != target[j])
        i, j = i[keep], j[keep]
        true_order = target[i] < target[j]
        pred_order = score[i] < score[j]
        pairwise = float((true_order == pred_order).mean())
    else:
        pairwise = float("nan")

    # Hits@K and MRR: rank the true partner against the full candidate pool.
    # Two pools: all nodes, and type-matched (only nodes of v's type compete).
    # Both _hits_at_k calls return dicts keyed by K plus an 'mrr' key.
    hits_all: dict = {k: float("nan") for k in hits_ks}
    hits_all["mrr"] = float("nan")
    hits_type: dict = {k: float("nan") for k in hits_ks}
    hits_type["mrr"] = float("nan")
    if compute_hits and src.numel() > 0:
        all_candidates = torch.arange(data.n_nodes, device=device)
        hits_all = hits_fn(
            model, h, src, dst, all_candidates,
            n_nodes_total=data.n_nodes,
            ks=hits_ks,
        )
        # Type-matched: for each positive, the candidate pool is the set of
        # nodes whose node-type matches the true dst. We compute one batch
        # per dst-type, since within a type the pool is shared.
        node_type_np = data.node_type.cpu().numpy()
        dst_types_np = node_type_np[dst.cpu().numpy()]
        per_type_accum: dict = {k: 0.0 for k in hits_ks}
        per_type_accum["mrr"] = 0.0
        per_type_total = 0
        for t in np.unique(dst_types_np):
            ids_of_type = np.where(node_type_np == int(t))[0]
            if len(ids_of_type) == 0:
                continue
            mask_t = dst_types_np == t
            sub_src = src[torch.from_numpy(mask_t)].to(device)
            sub_dst = dst[torch.from_numpy(mask_t)].to(device)
            pool = torch.from_numpy(ids_of_type.astype(np.int64)).to(device)
            sub_hits = hits_fn(
                model, h, sub_src, sub_dst, pool,
                n_nodes_total=data.n_nodes,
                ks=hits_ks,
            )
            n_sub = int(sub_src.numel())
            for key, frac in sub_hits.items():
                per_type_accum[key] += frac * n_sub
            per_type_total += n_sub
        if per_type_total:
            hits_type = {key: per_type_accum[key] / per_type_total
                         for key in per_type_accum}

    out = {
        "exist_auc": exist_auc,
        "exist_auc_smart": exist_auc_smart,
        "type_macro_auc": type_auc,
        "type_top1": type_top1,
        "type_majority_baseline": type_majority,
        "type_classes_in_test": type_auc_classes_used,
        "type_classes_missing_from_test": [
            int(c) for c in range(data.n_edge_types) if c not in present_classes
        ],
        "order_spearman": rho,
        "order_pairwise_acc": pairwise,
        "n_edges": int(mask.sum()),
    }
    for k in hits_ks:
        out[f"hits_at_{k}"] = hits_all[k]
        out[f"hits_at_{k}_type"] = hits_type[k]
    out["mrr"] = hits_all.get("mrr", float("nan"))
    out["mrr_type"] = hits_type.get("mrr", float("nan"))
    return out


def evaluate(
    model: FISHRGCN, data: FISHData, n_neg_per_pos: int = 20,
    device: Optional[str] = None, seed: int = 0,
    compute_hits: bool = False,
    hits_ks: tuple[int, ...] = (1, 10, 100),
    hits_batch: int = 2048,
) -> dict:
    """
    Compute the three task metrics on the held-out test set(s).

    Returns metrics keyed by ``{metric}`` for the primary test mask
    (semi-inductive bridge edges, the original protocol). When an
    inductive set exists (semi_inductive split with both-endpoints-unseen
    edges), additionally reports ``ind_{metric}`` from those edges.

    When ``compute_hits=True``, Hits@K is also computed against both an
    all-nodes pool and a type-matched pool. Costly (~minutes per call on
    Immune-sized graphs) so off by default.
    """
    with torch.no_grad():
        return _evaluate_impl(
            model, data, n_neg_per_pos, device, seed,
            compute_hits=compute_hits,
            hits_ks=hits_ks,
            hits_batch=hits_batch,
        )


def _evaluate_impl(
    model: FISHRGCN, data: FISHData, n_neg_per_pos: int,
    device: Optional[str], seed: int,
    compute_hits: bool,
    hits_ks: tuple[int, ...],
    hits_batch: int,
    hits_fn: Optional[Callable] = None,
) -> dict:
    device = device or next(model.parameters()).device
    model.eval()
    h = model.encode(data)

    # Build smart-negative state once from the training adjacency.
    neg_state = _build_negative_sampler_state(data)

    # Primary: semi-inductive (bridge) test set.
    primary = _score_edge_set(
        model, data, h, data.test_mask, n_neg_per_pos, device, seed,
        neg_state=neg_state,
        compute_hits=compute_hits, hits_ks=hits_ks, hits_batch=hits_batch,
        hits_fn=hits_fn,
    )
    out: dict = dict(primary)
    out["n_test_edges"] = primary["n_edges"]

    # Secondary: inductive (both-unseen) test set, when present.
    if data.inductive_mask is not None and int(data.inductive_mask.sum()) > 0:
        inductive = _score_edge_set(
            model, data, h, data.inductive_mask, n_neg_per_pos, device, seed + 1,
            neg_state=neg_state,
            compute_hits=compute_hits, hits_ks=hits_ks, hits_batch=hits_batch,
            hits_fn=hits_fn,
        )
        for k, v in inductive.items():
            out[f"ind_{k}"] = v
    else:
        out["ind_exist_auc"] = float("nan")
        out["ind_exist_auc_smart"] = float("nan")
        out["ind_type_top1"] = float("nan")
        out["ind_order_pairwise_acc"] = float("nan")
        out["ind_n_edges"] = 0

    # Cold-start diagnostic on the primary set.
    seen_nodes = set(data.all_src[data.train_mask].cpu().numpy().tolist())
    seen_nodes |= set(data.all_dst[data.train_mask].cpu().numpy().tolist())
    src_np = data.all_src[data.test_mask].cpu().numpy()
    dst_np = data.all_dst[data.test_mask].cpu().numpy()
    both_seen = sum(1 for s, d in zip(src_np, dst_np)
                    if s in seen_nodes and d in seen_nodes)
    one_seen = sum(1 for s, d in zip(src_np, dst_np)
                   if (s in seen_nodes) != (d in seen_nodes))
    neither_seen = len(src_np) - both_seen - one_seen
    out["test_both_endpoints_seen"] = both_seen
    out["test_one_endpoint_seen"] = one_seen
    out["test_neither_endpoint_seen"] = neither_seen

    return out

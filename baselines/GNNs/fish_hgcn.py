"""
fish_hgcn.py — HypergraphConv multi-task baseline for the FISH benchmark.

Same three tasks as fish_rgcn.py / fish_tgat.py:
  * existence — binary BCE with K negatives per positive edge
  * type      — k-way cross-entropy on positive edges
  * order     — regression / CORN / CORAL on normalised step times

Encoder is PyG's HypergraphConv (Bai et al. 2021). Because HypergraphConv is
not heterogeneous, node-type information is injected in two ways:

  1. Per-node-type input projections (same as RGCN) collapse the heterogeneous
     feature spaces into a shared hidden dim.
  2. A learnable node-type embedding is *added* to the projected features so
     the uniform HypergraphConv layers can still distinguish node types.

Hyperedge construction (from FISHData, no HyperNetX required)
--------------------------------------------------------------
  Base hyperedges  — one per unique time step in the training edges.
                     All src and dst nodes at that step are members.
                     Captures "all participants in this pathway event".

  Type hyperedges  — optionally added (use_type_hyperedges=True).
                     One per unique edge type; all nodes connected by edges
                     of that type become members. Captures "all nodes playing
                     the same biochemical role". Off by default; useful ablation.

Splits, evaluation, and all helper functions are shared with fish_rgcn.py so
metrics are directly comparable across all three baselines.

Usage
-----
    from fish_hgcn import build_dataset, FISHGCN, train, evaluate

    data = build_dataset(G, order_mode="regression")
    model = FISHGCN(data, hidden=128, n_layers=2)
    history = train(model, data, epochs=200, lr=1e-3)
    metrics = evaluate(model, data)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

from coral_pytorch.losses import corn_loss as _corn_loss
from coral_pytorch.losses import coral_loss as _coral_loss
from coral_pytorch.dataset import corn_label_from_logits as _corn_label_from_logits

from fish_rgcn import (
    FISHData,
    build_dataset,
    _sample_negatives,
    _build_train_negative_state,
    _sample_train_smart_negatives,
    _build_negative_sampler_state,
    _sample_smart_negatives,
    _hits_at_k,
    _score_edge_set,
    _evaluate_impl,
)


# ── hyperedge construction ────────────────────────────────────────────────────


def _build_hyperedge_index(
    data: FISHData,
    use_type_hyperedges: bool,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    Build the hyperedge_index tensor for PyG's HypergraphConv.

    Returns
    -------
    hyperedge_index : (2, E_inc) long
        Row 0 = node indices, row 1 = hyperedge indices. Duplicate (node,
        hyperedge) pairs are removed so each node appears at most once per
        hyperedge (HypergraphConv is invariant to multiplicities).
    n_hyperedges : int
        Total number of hyperedges (needed for HypergraphConv's num_edges).

    Hyperedges
    ----------
    Time-step hyperedges (always):
      One per unique time value in the training edges. All src and dst nodes
      of training edges at step t are members of hyperedge t.

    Edge-type hyperedges (when use_type_hyperedges=True):
      One per unique edge type in training. All src and dst nodes of training
      edges of type r are members of hyperedge r. These are appended after
      the time-step hyperedges.
    """
    src_np = data.all_src[data.train_mask].cpu().numpy().astype(np.int64)
    dst_np = data.all_dst[data.train_mask].cpu().numpy().astype(np.int64)
    time_np = data.all_time[data.train_mask].cpu().numpy()
    etype_np = data.all_type[data.train_mask].cpu().numpy().astype(np.int64)

    # ── Time-step hyperedges ──────────────────────────────────────────────
    unique_steps = np.unique(time_np)
    step_to_he = {float(t): i for i, t in enumerate(unique_steps)}
    n_time_he = len(unique_steps)

    node_ids: list[int] = []
    he_ids: list[int] = []

    for s, d, t in zip(src_np, dst_np, time_np):
        he = step_to_he[float(t)]
        node_ids.append(int(s)); he_ids.append(he)
        node_ids.append(int(d)); he_ids.append(he)

    n_type_he = 0
    if use_type_hyperedges:
        unique_etypes = np.unique(etype_np)
        type_to_he = {int(et): n_time_he + i for i, et in enumerate(unique_etypes)}
        n_type_he = len(unique_etypes)
        for s, d, et in zip(src_np, dst_np, etype_np):
            he = type_to_he[int(et)]
            node_ids.append(int(s)); he_ids.append(he)
            node_ids.append(int(d)); he_ids.append(he)

    # Deduplicate (node, hyperedge) pairs.
    pairs = np.unique(
        np.stack([np.array(node_ids, dtype=np.int64),
                  np.array(he_ids,   dtype=np.int64)], axis=0),
        axis=1,
    )

    hyperedge_index = torch.from_numpy(pairs).to(device)
    n_hyperedges = n_time_he + n_type_he
    return hyperedge_index, n_hyperedges


# ── encoder ───────────────────────────────────────────────────────────────────


class FISHHypergraphEncoder(nn.Module):
    """
    Multi-layer HypergraphConv encoder.

    Applies n_layers of PyG's HypergraphConv with ReLU + dropout between each.
    Optionally uses the attention variant (use_attention=True, n_heads > 1).
    """

    def __init__(
        self,
        hidden: int,
        n_layers: int,
        dropout: float,
        use_attention: bool = False,
        n_heads: int = 1,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_ch = hidden
            out_ch = hidden
            if use_attention and n_heads > 1:
                self.convs.append(
                    HypergraphConv(
                        in_ch, out_ch,
                        use_attention=True,
                        heads=n_heads,
                        concat=False,  # average heads → keep dim = hidden
                        dropout=dropout,
                    )
                )
            else:
                self.convs.append(HypergraphConv(in_ch, out_ch))

    def forward(
        self,
        x: torch.Tensor,            # (N, hidden)
        hyperedge_index: torch.Tensor,  # (2, E_inc)
        n_hyperedges: int,
    ) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, hyperedge_index, num_edges=n_hyperedges)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ── model ─────────────────────────────────────────────────────────────────────


class FISHGCN(nn.Module):
    """
    Hypergraph Convolutional Network multi-task model for FISH.

    Drop-in complement to FISHRGCN and FISHTGAT: identical heads, identical
    FISHData input, identical evaluation protocol.

    Node-type heterogeneity handled by:
      - Per-type input projections (raw feature dims → hidden)
      - Learnable node-type embedding added to projected features so
        the homogeneous HypergraphConv layers still see type signals
    """

    def __init__(
        self,
        data: FISHData,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = False,
        n_heads: int = 1,
        use_type_hyperedges: bool = False,
        order_mode: str = "regression",
        n_order_bins: int = 20,
        n_negatives: int = 5,
        use_compartment: bool = False,
        smart_negatives: bool = False,
    ):
        super().__init__()
        self.order_mode = order_mode
        self.n_order_bins = n_order_bins
        self.n_negatives = n_negatives
        self.use_compartment = use_compartment
        self.smart_negatives = smart_negatives
        self.use_type_hyperedges = use_type_hyperedges

        if use_compartment and data.compartment_dim == 0:
            raise ValueError(
                "use_compartment=True but dataset has no compartment features. "
                "Pass compartment_embeddings to build_dataset."
            )

        # Per-type input projections — same as RGCN / TGAT.
        compartment_extra = data.compartment_dim if use_compartment else 0
        self.input_projs = nn.ModuleDict()
        for ti, tensor in data.feats_by_type.items():
            in_dim = tensor.shape[1] + compartment_extra
            self.input_projs[str(ti)] = nn.Linear(in_dim, hidden)

        # Learnable node-type embedding added to the initial projection.
        # This injects heterogeneity into the uniform HypergraphConv layers
        # without requiring per-type message passing.
        self.node_type_emb = nn.Embedding(data.n_node_types, hidden)

        # HypergraphConv encoder.
        self.hg_encoder = FISHHypergraphEncoder(
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
            use_attention=use_attention,
            n_heads=n_heads,
        )

        # Task heads — identical architecture to FISHRGCN / FISHTGAT.
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

    def _project_inputs(self, data: FISHData, device) -> torch.Tensor:
        """Project heterogeneous features + node-type embedding → (N, hidden)."""
        if self.use_compartment:
            compartment = data.compartment_feat.to(device)

        projected_per_type = []
        for ti in range(data.n_node_types):
            tensor = data.feats_by_type[ti].to(device)
            if self.use_compartment:
                type_node_ids = (data.node_type == ti).nonzero(as_tuple=False).squeeze(1)
                order = data.node_intratype_idx[type_node_ids].argsort()
                type_node_ids = type_node_ids[order]
                comp_block = compartment[type_node_ids]
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

        # Add node-type embedding so HypergraphConv layers see type signals.
        node_type = data.node_type.to(device)
        h = h + self.node_type_emb(node_type)
        return h

    def encode(self, data: FISHData) -> torch.Tensor:
        """Return node embeddings of shape (N, hidden)."""
        device = next(self.parameters()).device
        h = self._project_inputs(data, device)
        hyperedge_index, n_he = _build_hyperedge_index(
            data, self.use_type_hyperedges, device
        )
        return self.hg_encoder(h, hyperedge_index, n_he)

    # ── heads ────────────────────────────────────────────────────────────────

    def _pair_repr(self, h, src, dst):
        return torch.cat([h[src], h[dst]], dim=-1)

    def existence_logit(self, h, src, dst):
        return self.head_exist(self._pair_repr(h, src, dst)).squeeze(-1)

    def type_logits(self, h, src, dst):
        return self.head_type(self._pair_repr(h, src, dst))

    def order_output(self, h, src, dst):
        return self.head_order(self._pair_repr(h, src, dst))


# ── training ──────────────────────────────────────────────────────────────────


def train(
    model: FISHGCN,
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

    order_target_tr = data.order_target[data.train_mask].to(device)
    src_tr = data.all_src[data.train_mask].to(device)
    dst_tr = data.all_dst[data.train_mask].to(device)
    typ_tr = data.all_type[data.train_mask].to(device)
    bin_tr = (data.order_bin[data.train_mask].to(device)
              if data.order_bin is not None else None)

    coral_levels = None
    if model.order_mode == "coral" and bin_tr is not None:
        thresholds = torch.arange(model.n_order_bins - 1, device=device)
        coral_levels = (bin_tr.unsqueeze(1) > thresholds.unsqueeze(0)).float()

    train_neg_state = None
    if model.smart_negatives:
        train_neg_state = _build_train_negative_state(data, device)

    history = {"exist": [], "type": [], "order": [], "total": []}

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        h = model.encode(data)

        # 1) Existence.
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

        # 2) Type.
        type_lgt = model.type_logits(h, src_tr, dst_tr)
        loss_type = F.cross_entropy(type_lgt, typ_tr)

        # 3) Order.
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


# ── evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    model: FISHGCN,
    data: FISHData,
    n_neg_per_pos: int = 20,
    device: Optional[str] = None,
    seed: int = 0,
    compute_hits: bool = False,
    hits_ks: tuple[int, ...] = (1, 10, 100),
    hits_batch: int = 2048,
) -> dict:
    """Delegate to the shared _evaluate_impl from fish_rgcn."""
    with torch.no_grad():
        return _evaluate_impl(
            model, data, n_neg_per_pos, device, seed,
            compute_hits=compute_hits,
            hits_ks=hits_ks,
            hits_batch=hits_batch,
        )

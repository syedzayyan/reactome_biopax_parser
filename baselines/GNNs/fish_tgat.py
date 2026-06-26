"""
fish_tgat.py — TGAT multi-task baseline for the FISH benchmark.

Same three tasks as fish_rgcn.py:
  * existence — binary BCE with K negatives per positive edge
  * type      — k-way classification over edge types, cross-entropy on positives
  * order     — regression / CORN / CORAL on normalised step times

Encoder is TGAT (Xu et al. 2020), using TGM library's TemporalAttention with
the shared fixed-frequency TimeEncode (time_encoding.py, also used by
fish_rgcn.py and fish_sthn.py) for time encoding, adapted to the
heterogeneous FISH graph:
  - Per-node-type input projections to a shared hidden dim (same as FISHRGCN)
  - Edge type embedded to edge_feat_dim before temporal attention
  - Reference time per node = normalised max incident training-edge time
  - Most-recent max_neighbors neighbours per node, padded to fixed width
    and passed together as one batch (vectorised, no per-node Python loop)

Data loading, splits, and evaluation are shared with fish_rgcn.py — FISHData,
build_dataset and all _score_edge_set helpers are imported directly from there,
so both baselines produce exactly comparable metrics.

Usage
-----
    from fish_tgat import build_dataset, FISHTGAT, train, evaluate

    data = build_dataset(G, order_mode="regression")
    model = FISHTGAT(data, hidden=128, n_layers=2, n_heads=2)
    history = train(model, data, epochs=200, lr=1e-3)
    metrics = evaluate(model, data)
    print(metrics)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgm.nn import TemporalAttention

# Shared fixed-frequency cosine time encoder (GraphMixer-style), also used by
# fish_rgcn.py and fish_sthn.py. Note this replaces the learnable Time2Vec
# (Kazemi et al. 2019 / TGAT) this module used previously -- "time encoding"
# now means the same frozen encoder across all three baselines.
from time_encoding import TimeEncode
from coral_pytorch.losses import corn_loss as _corn_loss
from coral_pytorch.losses import coral_loss as _coral_loss
from coral_pytorch.dataset import corn_label_from_logits as _corn_label_from_logits

# Shared data structures and evaluation helpers from the RGCN baseline.
# FISHData and build_dataset are re-exported so callers only need fish_tgat.
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


# ── neighbour table ──────────────────────────────────────────────────────────


def _build_neighbor_table(
    data: FISHData,
    max_neighbors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a padded incoming-neighbour table for all nodes.

    For each node v, collects training edges that point INTO v (v is the
    destination), sorted by time descending (most recent first), keeping at
    most max_neighbors. Shorter lists are right-padded.

    Time values are normalised to [0, 1] by the global training-edge maximum.

    Returns
    -------
    nbr_ids    (N, K) long  — source-node ids of neighbours (0-padded)
    nbr_types  (N, K) long  — edge-type ids; n_edge_types = padding sentinel
    nbr_times  (N, K) float — normalised edge timestamps [0, 1]
    valid_mask (N, K) bool  — True where the slot is a real neighbour
    ref_times  (N,)   float — normalised reference time per node (max incident)
    """
    N = data.n_nodes
    K = max_neighbors
    pad_etype = data.n_edge_types  # sentinel index for the padding embedding

    src_tr = data.all_src[data.train_mask].cpu().numpy()
    dst_tr = data.all_dst[data.train_mask].cpu().numpy()
    etype_tr = data.all_type[data.train_mask].cpu().numpy()
    time_tr = data.all_time[data.train_mask].cpu().numpy().astype(np.float32)

    t_max = float(data.all_time.max())
    if t_max <= 0.0:
        t_max = 1.0
    time_norm = (time_tr / t_max).astype(np.float32)

    # Per-node incoming-edge lists: [(t_norm, src, etype), ...]
    per_node: list[list] = [[] for _ in range(N)]
    for s, d, et, t in zip(src_tr, dst_tr, etype_tr, time_norm):
        per_node[d].append((float(t), int(s), int(et)))

    # Reference time per node = max of all incident training-edge times
    # (either src or dst side). Isolated nodes get ref_time = 0.
    ref_np = np.zeros(N, dtype=np.float32)
    for v, nbrs in enumerate(per_node):
        if nbrs:
            ref_np[v] = max(t for t, _, _ in nbrs)
    for s, t in zip(src_tr, time_norm):
        if float(t) > ref_np[s]:
            ref_np[s] = float(t)

    # Padded arrays.
    nbr_ids_np = np.zeros((N, K), dtype=np.int64)
    nbr_types_np = np.full((N, K), pad_etype, dtype=np.int64)
    nbr_times_np = np.zeros((N, K), dtype=np.float32)
    valid_np = np.zeros((N, K), dtype=bool)

    for v, nbrs in enumerate(per_node):
        # Most-recent K neighbours first.
        for j, (t, s, et) in enumerate(sorted(nbrs, reverse=True)[:K]):
            nbr_ids_np[v, j] = s
            nbr_types_np[v, j] = et
            nbr_times_np[v, j] = t
            valid_np[v, j] = True

    return (
        torch.from_numpy(nbr_ids_np).to(device),
        torch.from_numpy(nbr_types_np).to(device),
        torch.from_numpy(nbr_times_np).to(device),
        torch.from_numpy(valid_np).to(device),
        torch.from_numpy(ref_np).to(device),
    )


# ── TGAT encoder ────────────────────────────────────────────────────────────


class FISHTGATEncoder(nn.Module):
    """
    Multi-layer TGAT encoder for FISH heterogeneous graphs.

    Each layer runs TGM's TemporalAttention over incoming training neighbours,
    then merges attention output with the initial projected features via a
    two-layer MLP (residual skip to input, following the original TGAT paper).

    Edge type is embedded to edge_feat_dim; time is encoded by TimeEncode
    (shared fixed-frequency cosine encoder, see time_encoding.py) on the
    per-edge (ref_time - edge_time) delta, so each layer sees genuine causal
    temporal context.

    use_time_encoding=False zeroes both the reference-time and time-delta
    inputs to TimeEncode before every call, so time_feat/nbr_time_feat
    collapse to the constant vector cos(b) for every node and neighbour
    regardless of real timestamps. TemporalAttention's signature requires
    time tensors of fixed shape, so (unlike fish_sthn.py, which drops its
    time submodule entirely) this is a constant-time ablation rather than an
    architectural one. TimeEncode's weights are frozen either way (same
    encoder fish_rgcn.py uses), so this ablation purely tests whether the
    model sees a real timestamp at all, not whether the encoder can learn.
    """

    def __init__(
        self,
        hidden: int,
        n_layers: int,
        n_heads: int,
        time_dim: int,
        edge_feat_dim: int,
        n_edge_types: int,
        dropout: float,
        use_time_encoding: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_time_encoding = use_time_encoding

        self.time_encoder = TimeEncode(time_dim)

        # n_edge_types + 1 embeddings; index n_edge_types is the padding sentinel.
        self.edge_type_emb = nn.Embedding(
            n_edge_types + 1, edge_feat_dim, padding_idx=n_edge_types
        )

        self.attn_layers = nn.ModuleList()
        self.merge_layers = nn.ModuleList()

        for _ in range(n_layers):
            attn = TemporalAttention(
                n_heads=n_heads,
                node_dim=hidden,
                edge_dim=edge_feat_dim,
                time_dim=time_dim,
                dropout=dropout,
            )
            self.attn_layers.append(attn)
            # Merge: concat(attn_out, h_init) → hidden.
            # Residual skip uses the initial projected features (h_init),
            # same convention as the local modules/TGAT.py reference.
            self.merge_layers.append(nn.Sequential(
                nn.Linear(attn.out_dim + hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            ))

    def forward(
        self,
        h_init: torch.Tensor,         # (N, hidden)
        nbr_ids: torch.Tensor,        # (N, K) long
        nbr_edge_types: torch.Tensor, # (N, K) long
        nbr_times: torch.Tensor,      # (N, K) float, normalised [0, 1]
        valid_mask: torch.Tensor,     # (N, K) bool
        ref_times: torch.Tensor,      # (N,) float, normalised
    ) -> torch.Tensor:
        """Return updated node embeddings of shape (N, hidden)."""
        h = h_init

        if not self.use_time_encoding:
            ref_times = torch.zeros_like(ref_times)

        # Reference-time encoding, computed once for all layers.
        # TimeEncode(ref_times) with ref_times.shape=(N,):
        #   reshape(-1,1) → (N,1) → Linear → cos → (N, time_dim).
        time_feat = self.time_encoder(ref_times)  # (N, time_dim)

        for attn, merge in zip(self.attn_layers, self.merge_layers):
            # Gather neighbour hidden states from the previous layer.
            nbr_node_feat = h[nbr_ids]                        # (N, K, hidden)
            nbr_node_feat = nbr_node_feat * valid_mask.unsqueeze(-1).float()

            # Edge-type embeddings for the valid neighbours.
            edge_feat = self.edge_type_emb(nbr_edge_types)   # (N, K, edge_feat_dim)
            edge_feat = edge_feat * valid_mask.unsqueeze(-1).float()

            # Temporal delta: ref_time - edge_time (causal, so >= 0 by construction).
            # TimeEncode flattens its input to (-1, 1) internally regardless of
            # the original shape, so time_diffs.shape=(N,K) comes back as
            # (N*K, time_dim) -- reshape back before using it as (N,K,time_dim).
            if self.use_time_encoding:
                time_diffs = (ref_times.unsqueeze(1) - nbr_times).clamp(min=0.0)
                time_diffs = time_diffs * valid_mask.float()
            else:
                time_diffs = torch.zeros_like(nbr_times)
            nbr_time_feat = self.time_encoder(time_diffs).reshape(*time_diffs.shape, -1)
            nbr_time_feat = nbr_time_feat * valid_mask.unsqueeze(-1).float()

            # Temporal attention (all N nodes in one batch).
            h_attn = attn(
                node_feat=h,
                time_feat=time_feat,
                edge_feat=edge_feat,
                nbr_node_feat=nbr_node_feat,
                nbr_time_feat=nbr_time_feat,
                valid_nbr_mask=valid_mask,
            )  # (N, attn.out_dim)

            # Merge with initial projected features (residual skip to input).
            h = merge(torch.cat([h_attn, h_init], dim=-1))   # (N, hidden)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h


# ── model ────────────────────────────────────────────────────────────────────


class FISHTGAT(nn.Module):
    """
    TGAT multi-task model for FISH: shared TGAT encoder + three task heads.

    Drop-in complement to FISHRGCN: identical heads, identical FISHData
    input, identical evaluation protocol. Swap RGCN message passing for
    TGAT temporal attention and compare.

    Order head modes
    ----------------
    'regression' — MSE on normalised step time.
    'corn'       — CORN consistent ordinal loss over quantile bins.
    'coral'      — CORAL ordinal loss over quantile bins.
    """

    def __init__(
        self,
        data: FISHData,
        hidden: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
        time_dim: int = 64,
        edge_feat_dim: int = 32,
        max_neighbors: int = 20,
        order_mode: str = "regression",
        n_order_bins: int = 20,
        n_negatives: int = 5,
        use_compartment: bool = False,
        smart_negatives: bool = False,
        use_time_encoding: bool = True,
    ):
        super().__init__()
        self.order_mode = order_mode
        self.n_order_bins = n_order_bins
        self.n_negatives = n_negatives
        self.use_compartment = use_compartment
        self.smart_negatives = smart_negatives
        self.use_time_encoding = use_time_encoding
        self.max_neighbors = max_neighbors

        if use_compartment and data.compartment_dim == 0:
            raise ValueError(
                "use_compartment=True but dataset has no compartment features "
                "(data.compartment_dim == 0). Pass compartment_embeddings to build_dataset."
            )

        # Per-type input projections — identical to FISHRGCN.
        compartment_extra = data.compartment_dim if use_compartment else 0
        self.input_projs = nn.ModuleDict()
        for ti, tensor in data.feats_by_type.items():
            in_dim = tensor.shape[1] + compartment_extra
            self.input_projs[str(ti)] = nn.Linear(in_dim, hidden)

        # TGAT encoder.
        self.tgat_encoder = FISHTGATEncoder(
            hidden=hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            time_dim=time_dim,
            edge_feat_dim=edge_feat_dim,
            n_edge_types=data.n_edge_types,
            dropout=dropout,
            use_time_encoding=use_time_encoding,
        )

        # Cached neighbor table — built once from training edges (leakage-proof).
        # Analogous to FISHSTHN's _adj_etype/_adj_dt/_adj_valid cache.
        self._nbr_ids:   Optional[torch.Tensor] = None
        self._nbr_types: Optional[torch.Tensor] = None
        self._nbr_times: Optional[torch.Tensor] = None
        self._nbr_valid: Optional[torch.Tensor] = None
        self._nbr_ref:   Optional[torch.Tensor] = None

        # Task heads — identical architecture to FISHRGCN for comparability.
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
        """Project heterogeneous per-type features to the shared hidden dim."""
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
        return h

    def encode(self, data: FISHData) -> torch.Tensor:
        """Return node embeddings of shape (N, hidden)."""
        device = next(self.parameters()).device
        if self._nbr_ids is None or self._nbr_ids.device != device:
            (self._nbr_ids, self._nbr_types, self._nbr_times,
             self._nbr_valid, self._nbr_ref) = _build_neighbor_table(
                data, self.max_neighbors, device
            )
        h_init = self._project_inputs(data, device)
        return self.tgat_encoder(
            h_init,
            self._nbr_ids, self._nbr_types, self._nbr_times,
            self._nbr_valid, self._nbr_ref,
        )

    # ── heads ────────────────────────────────────────────────────────────────

    def _pair_repr(self, h, src, dst):
        return torch.cat([h[src], h[dst]], dim=-1)

    def existence_logit(self, h, src, dst):
        return self.head_exist(self._pair_repr(h, src, dst)).squeeze(-1)

    def type_logits(self, h, src, dst):
        return self.head_type(self._pair_repr(h, src, dst))

    def order_output(self, h, src, dst):
        return self.head_order(self._pair_repr(h, src, dst))


# ── training ─────────────────────────────────────────────────────────────────


def train(
    model: FISHTGAT,
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

        # 2) Type: cross-entropy on positives.
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


# ── evaluation ───────────────────────────────────────────────────────────────


def evaluate(
    model: FISHTGAT,
    data: FISHData,
    n_neg_per_pos: int = 20,
    device: Optional[str] = None,
    seed: int = 0,
    compute_hits: bool = False,
    hits_ks: tuple[int, ...] = (1, 10, 100),
    hits_batch: int = 2048,
) -> dict:
    """
    Compute the three task metrics on the held-out test set(s).

    Delegates to the shared _evaluate_impl from fish_rgcn so metrics are
    directly comparable across RGCN and TGAT baselines.
    """
    with torch.no_grad():
        return _evaluate_impl(
            model, data, n_neg_per_pos, device, seed,
            compute_hits=compute_hits,
            hits_ks=hits_ks,
            hits_batch=hits_batch,
        )

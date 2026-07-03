"""
fish_sthn.py — STHN multi-task baseline for the FISH benchmark.

Same three tasks as fish_rgcn.py / fish_tgat.py:
  * existence — binary BCE with K negatives per positive edge
  * type      — k-way classification over edge types, cross-entropy on
                positives
  * order     — regression / CORN / CORAL on normalised step times

STHN (Subgraph/Simplifying Temporal Heterogeneous Network) scores a query
(src, dst) by encoding the *local temporal neighbourhood* of both endpoints
rather than relying on a single static per-node embedding:

  - A train-set-only, time-sorted adjacency table is built once per node
    (most-recent `max_edges` incident training edges, padded), analogous to
    `_build_neighbor_table` in fish_tgat.py but undirected (both endpoints
    of every training edge see it in their context).
  - For a query (src, dst), the "subgraph" is the concatenation of src's and
    dst's adjacency rows: 2*max_edges edges, each described by its edge-type
    embedding and a TimeEncode encoding of (ref_time[node] - edge_time).
  - A windowed Transformer mixer (`_PatchMixer`, in the spirit of the
    Patch_Encoding / channel-mixer used by STHN) compresses this sequence
    into a single link-context vector per query.
  - The final pair representation concatenates this link-context vector
    with the projected node features of src and dst (the same per-type
    input projections used by FISHRGCN/FISHTGAT), so STHN sees both
    "what's been happening around here recently" and "what kind of nodes
    are these".

Data loading, splits, and most evaluation machinery are shared with
fish_rgcn.py — FISHData, build_dataset, negative samplers, _score_edge_set
and _evaluate_impl are imported directly from there. The one exception is
Hits@K: `_hits_at_k` assumes a cached (N, hidden) embedding table and calls
`model.head_exist` on raw pairs, which doesn't fit STHN's per-query subgraph
encoder. `_hits_at_k_sthn` provides an approximate drop-in replacement that
samples a bounded candidate set per query (always including the true
destination) instead of scoring the full pool.

Usage
-----
    from fish_sthn import build_dataset, FISHSTHN, train, evaluate

    data = build_dataset(G, order_mode="regression")
    model = FISHSTHN(data, hidden=128, max_edges=20, window_size=5)
    history = train(model, data, epochs=200, lr=1e-3)
    metrics = evaluate(model, data)
    print(metrics)
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared fixed-frequency cosine time encoder (GraphMixer-style), also used by
# fish_rgcn.py and fish_tgat.py. Replaces the learnable Time2Vec (Kazemi et
# al. 2019 / TGAT) this module used previously.
from time_encoding import TimeEncode
from coral_pytorch.losses import corn_loss as _corn_loss
from coral_pytorch.losses import coral_loss as _coral_loss
from coral_pytorch.dataset import corn_label_from_logits as _corn_label_from_logits

try:
    # The memory-efficient SDPA backend caps batch*heads at 65535 and can't
    # save dropout RNG state above that, which the mixer's batched (all
    # pairs + negatives in one forward pass) attention call easily exceeds
    # on larger pathways. Restrict to flash/math, which have no such cap.
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend as _SDPBackend
    def _mixer_attention_backend():
        return _sdpa_kernel([_SDPBackend.FLASH_ATTENTION, _SDPBackend.MATH])
except ImportError:
    from contextlib import nullcontext as _mixer_attention_backend

# Shared data structures and evaluation helpers from the RGCN baseline.
# FISHData and build_dataset are re-exported so callers only need fish_sthn.
from fish_rgcn import (
    FISHData,
    build_dataset,
    _sample_negatives,
    _build_train_negative_state,
    _sample_train_smart_negatives,
    _build_negative_sampler_state,
    _sample_smart_negatives,
    _score_edge_set,
    _evaluate_impl,
)


# ── train-edge adjacency table ───────────────────────────────────────────────


def _build_train_adjacency(
    data: FISHData,
    max_edges: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a padded per-node adjacency table from training edges only.

    For each node v, collects every training edge touching v (as either
    endpoint), sorted by time descending (most recent first), keeping at
    most `max_edges`. Shorter lists are right-padded.

    Time deltas are relative to a per-node reference time (the most recent
    training edge touching that node), normalised to [0, 1] by the global
    training-edge maximum — the same convention as
    `_build_neighbor_table` in fish_tgat.py.

    Returns
    -------
    nbr_etype  (N, max_edges) long  — edge-type ids; n_edge_types = padding
    nbr_dt     (N, max_edges) float — ref_time[v] - edge_time, clamped >= 0
    valid_mask (N, max_edges) bool  — True where the slot is a real edge
    """
    N = data.n_nodes
    K = max_edges
    pad_etype = data.n_edge_types  # sentinel index for the padding embedding

    src_tr = data.all_src[data.train_mask].cpu().numpy()
    dst_tr = data.all_dst[data.train_mask].cpu().numpy()
    etype_tr = data.all_type[data.train_mask].cpu().numpy()
    time_tr = data.all_time[data.train_mask].cpu().numpy().astype(np.float32)

    # Use training-only max for normalisation so test-edge times are never
    # needed to define the scale (avoids even the minor normalisation leak).
    t_max = float(time_tr.max())
    if t_max <= 0.0:
        t_max = 1.0
    time_norm = (time_tr / t_max).astype(np.float32)

    # Per-node incident-edge lists: [(t_norm, etype), ...]. Both endpoints of
    # every training edge see it as part of their recent local context.
    per_node: list[list] = [[] for _ in range(N)]
    for s, d, et, t in zip(src_tr, dst_tr, etype_tr, time_norm):
        per_node[s].append((float(t), int(et)))
        per_node[d].append((float(t), int(et)))

    nbr_etype_np = np.full((N, K), pad_etype, dtype=np.int64)
    nbr_time_np = np.zeros((N, K), dtype=np.float32)   # actual normalised times
    valid_np = np.zeros((N, K), dtype=bool)

    for v, nbrs in enumerate(per_node):
        for j, (t, et) in enumerate(sorted(nbrs, reverse=True)[:K]):
            nbr_etype_np[v, j] = et
            nbr_time_np[v, j] = t          # store time, not delta — delta is per-query
            valid_np[v, j] = True

    return (
        torch.from_numpy(nbr_etype_np).to(device),
        torch.from_numpy(nbr_time_np).to(device),
        torch.from_numpy(valid_np).to(device),
    )


# ── encoder pieces ────────────────────────────────────────────────────────────


class _FeatEncode(nn.Module):
    """Edge-type embedding + TimeEncode(time-delta) -> hidden.

    use_time_encoding=False drops the TimeEncode submodule entirely (rather
    than zeroing its output) so the no-time condition has no path for
    temporal information to reach the mixer, mirroring fish_rgcn.py's and
    fish_tgat.py's time-encoding ablations. TimeEncode (time_encoding.py) is
    the same frozen GraphMixer-style encoder used by all three baselines.
    """

    def __init__(
        self, time_dim: int, edge_feat_dim: int, hidden: int,
        use_time_encoding: bool = True,
    ):
        super().__init__()
        self.use_time_encoding = use_time_encoding
        self.time_encoder = TimeEncode(time_dim) if use_time_encoding else None
        proj_in = edge_feat_dim + (time_dim if use_time_encoding else 0)
        self.proj = nn.Linear(proj_in, hidden)

    def forward(self, edge_feat: torch.Tensor, time_delta: torch.Tensor) -> torch.Tensor:
        # edge_feat: (B, L, edge_feat_dim); time_delta: (B, L)
        if self.use_time_encoding:
            # TimeEncode flattens internally to (-1, 1) regardless of input
            # shape, so reshape its output back to (B, L, time_dim).
            t_feat = self.time_encoder(time_delta).reshape(*time_delta.shape, -1)
            x = torch.cat([edge_feat, t_feat], dim=-1)
        else:
            x = edge_feat
        return self.proj(x)


class _PatchMixer(nn.Module):
    """
    Windowed Transformer mixer over a (B, per_graph_size, hidden) subgraph.

    Splits the subgraph sequence into `per_graph_size // window_size`
    windows, projects each window to `hidden`, adds a learned positional
    embedding, runs a small TransformerEncoder stack, and mean-pools over
    windows to produce one link-context vector per query.
    """

    def __init__(
        self,
        per_graph_size: int,
        hidden: int,
        window_size: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        channel_expansion_factor: int,
    ):
        super().__init__()
        if per_graph_size % window_size != 0:
            raise ValueError(
                f"per_graph_size ({per_graph_size}) must be divisible by "
                f"window_size ({window_size})"
            )
        self.window_size = window_size
        self.n_windows = per_graph_size // window_size

        self.pad_projector = nn.Linear(window_size * hidden, hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_windows, hidden) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * channel_expansion_factor,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.layernorm = nn.LayerNorm(hidden)

    def forward(self, edge_repr: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # edge_repr: (B, per_graph_size, hidden); valid_mask: (B, per_graph_size)
        edge_repr = edge_repr * valid_mask.unsqueeze(-1).float()
        B, L, H = edge_repr.shape
        x = edge_repr.reshape(B, self.n_windows, self.window_size * H)
        x = self.pad_projector(x)
        x = x + self.pos_emb
        with _mixer_attention_backend():
            for layer in self.layers:
                x = layer(x)
        x = self.layernorm(x)
        return x.mean(dim=1)  # (B, hidden)


# ── model ─────────────────────────────────────────────────────────────────────


class FISHSTHN(nn.Module):
    """
    STHN multi-task model for FISH: subgraph-mixer encoder + three task heads.

    Drop-in complement to FISHRGCN/FISHTGAT: identical FISHData input and
    evaluation protocol (modulo Hits@K, see _hits_at_k_sthn). Each (src, dst)
    pair is represented as
        cat([mixer(subgraph(src, dst)), h_init[src], h_init[dst]])
    where h_init is the per-type-projected node features (same as
    FISHRGCN/FISHTGAT's encoder input) and `subgraph(src, dst)` concatenates
    the train-only, time-sorted adjacency rows of src and dst.

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
        n_layers: int = 1,
        n_heads: int = 2,
        dropout: float = 0.2,
        time_dim: int = 64,
        edge_feat_dim: int = 32,
        max_edges: int = 20,
        window_size: int = 5,
        channel_expansion_factor: int = 2,
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
        self.max_edges = max_edges

        if use_compartment and data.compartment_dim == 0:
            raise ValueError(
                "use_compartment=True but dataset has no compartment features "
                "(data.compartment_dim == 0). Pass compartment_embeddings to build_dataset."
            )

        # Per-type input projections — identical to FISHRGCN/FISHTGAT.
        compartment_extra = data.compartment_dim if use_compartment else 0
        self.input_projs = nn.ModuleDict()
        for ti, tensor in data.feats_by_type.items():
            in_dim = tensor.shape[1] + compartment_extra
            self.input_projs[str(ti)] = nn.Linear(in_dim, hidden)

        # Subgraph encoder: edge-type + time-delta -> hidden, then mixer.
        self.edge_type_emb = nn.Embedding(
            data.n_edge_types + 1, edge_feat_dim, padding_idx=data.n_edge_types
        )
        self.feat_encode = _FeatEncode(
            time_dim, edge_feat_dim, hidden, use_time_encoding=use_time_encoding,
        )
        per_graph_size = 2 * max_edges
        self.mixer = _PatchMixer(
            per_graph_size=per_graph_size,
            hidden=hidden,
            window_size=window_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            channel_expansion_factor=channel_expansion_factor,
        )

        # Task heads — link-context (hidden) + src/dst node features (2*hidden).
        head_in = 3 * hidden
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

        # Train-only adjacency table, built lazily on first encode() call.
        # _adj_time stores actual normalised edge timestamps; time deltas are
        # computed per-query as (t_query - _adj_time) to enforce temporal causality.
        self._adj_etype: Optional[torch.Tensor] = None
        self._adj_time: Optional[torch.Tensor] = None
        self._adj_valid: Optional[torch.Tensor] = None

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
        """Return per-node feature projections of shape (N, hidden).

        The adjacency table (training edges only, actual normalised timestamps)
        is cached here. Time deltas and causal masks are applied per-query in
        _subgraph_repr to enforce t_j < t_query temporal causality.
        """
        device = next(self.parameters()).device
        if self._adj_etype is None or self._adj_etype.device != device:
            self._adj_etype, self._adj_time, self._adj_valid = _build_train_adjacency(
                data, self.max_edges, device
            )
        return self._project_inputs(data, device)

    # ── subgraph + heads ─────────────────────────────────────────────────────

    def _subgraph_repr(
        self, src: torch.Tensor, dst: torch.Tensor, t_query: torch.Tensor
    ) -> torch.Tensor:
        """Build the 2K-edge subgraph representation with temporal causal masking.

        Only edges with timestamp < t_query are included (tj < t_query per
        TGAT/STHN design). Time deltas are computed as t_query - edge_time so
        that ref_time is the actual query time rather than a per-node maximum.
        """
        t_q = t_query.unsqueeze(1)                      # (B, 1)
        # Causal filter: only edges preceding t_query.
        valid_s = self._adj_valid[src] & (self._adj_time[src] < t_q)
        valid_d = self._adj_valid[dst] & (self._adj_time[dst] < t_q)
        valid = torch.cat([valid_s, valid_d], dim=1)    # (B, 2K)
        # Time delta: t_query - edge_time (>= 0 by causal constraint).
        dt_s = (t_q - self._adj_time[src]).clamp(min=0.0) * valid_s.float()
        dt_d = (t_q - self._adj_time[dst]).clamp(min=0.0) * valid_d.float()
        dt = torch.cat([dt_s, dt_d], dim=1)             # (B, 2K)
        etype = torch.cat([self._adj_etype[src], self._adj_etype[dst]], dim=1)
        edge_feat = self.edge_type_emb(etype)            # (B, 2K, edge_feat_dim)
        enc = self.feat_encode(edge_feat, dt)             # (B, 2K, hidden)
        return self.mixer(enc, valid)                     # (B, hidden)

    def _pair_repr(
        self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, t_query: torch.Tensor
    ) -> torch.Tensor:
        sub = self._subgraph_repr(src, dst, t_query)
        return torch.cat([sub, h[src], h[dst]], dim=-1)

    def existence_logit(self, h, src, dst, t_query):
        return self.head_exist(self._pair_repr(h, src, dst, t_query)).squeeze(-1)

    def type_logits(self, h, src, dst, t_query):
        return self.head_type(self._pair_repr(h, src, dst, t_query))

    def order_output(self, h, src, dst, t_query):
        return self.head_order(self._pair_repr(h, src, dst, t_query))


# ── training ──────────────────────────────────────────────────────────────────


def train(
    model: FISHSTHN,
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
    # Normalised query times for training edges — same scale as _adj_time.
    _t_max_tr = float(data.all_time[data.train_mask].max())
    if _t_max_tr <= 0.0:
        _t_max_tr = 1.0
    time_tr = (data.all_time[data.train_mask].float().to(device) / _t_max_tr).clamp(min=0.0)

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
        # t_query for negatives = same event time as the positive (the query
        # happens at t_positive; we ask whether the negative edge exists then).
        pos_logit = model.existence_logit(h, src_tr, dst_tr, time_tr)
        if model.smart_negatives and train_neg_state is not None:
            negs = _sample_train_smart_negatives(
                src_tr, dst_tr, train_neg_state, model.n_negatives,
            )
        else:
            negs = _sample_negatives(
                data, src_tr, model.n_negatives, data.n_nodes, device,
            )
        src_rep = src_tr.repeat_interleave(model.n_negatives)
        time_tr_rep = time_tr.repeat_interleave(model.n_negatives)
        neg_logit = model.existence_logit(h, src_rep, negs.reshape(-1), time_tr_rep)
        loss_exist = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logit, neg_logit]),
            torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)]),
        )

        # 2) Type: cross-entropy on positives.
        type_lgt = model.type_logits(h, src_tr, dst_tr, time_tr)
        loss_type = F.cross_entropy(type_lgt, typ_tr)

        # 3) Order: regression, CORN, or CORAL.
        out_order = model.order_output(h, src_tr, dst_tr, time_tr)
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


@torch.no_grad()
def _hits_at_k_sthn(
    model: FISHSTHN,
    h: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    candidate_pool: torch.Tensor,
    n_nodes_total: int,
    ks: tuple[int, ...] = (1, 10, 100),
    dyads_per_chunk: int = 9_000,
    max_candidates: int = 300,
) -> dict:
    """
    Approximate Hits@K / MRR for STHN.

    _hits_at_k (fish_rgcn) assumes a cached (N, hidden) embedding table and
    scores every candidate via a single MLP on cat([h[src], h[cand]]).
    STHN has no such table — each (src, cand) pair needs its own subgraph
    extraction + mixer forward, so scoring the full candidate pool (often
    thousands of nodes) per positive is infeasible.

    Instead, for each positive (src, dst) we build a per-row candidate
    block of size min(max_candidates, |candidate_pool|), with the true dst
    fixed at position 0 (so it is always scored) and the remaining slots
    sampled uniformly at random (with replacement) from candidate_pool.
    Rank = #candidates with a strictly higher existence score than dst, + 1.
    Hits@K / MRR are accumulated the same way as _hits_at_k.

    dyads_per_chunk controls the CUDA batch size: batch_size = dyads_per_chunk
    // max_candidates source nodes are scored per call, so the peak tensor in
    the mixer is (batch_size * max_candidates, 2*max_edges, hidden). The
    default 9_000 caps this at ~30 * 300 = 9K pairs → ~184 MB at hidden=128,
    max_edges=20. Unlike _hits_at_k's 1M default (cheap MLP on fixed
    embeddings), the mixer's memory cost scales with pairs, not just src nodes.
    Bind a larger value via functools.partial if GPU memory allows it.

    This is a sampled-ranking approximation, not the exact full-pool metric
    that RGCN/TGAT report — treat STHN's Hits@K/MRR as directionally
    comparable but not numerically identical to the other baselines.
    """
    n_pos = src.numel()
    n_cand = candidate_pool.numel()
    if n_pos == 0 or n_cand == 0:
        out: dict = {k: float("nan") for k in ks}
        out["mrr"] = float("nan")
        return out

    n_block = min(max_candidates, n_cand)
    n_extra = max(n_block - 1, 0)
    batch_size = max(1, dyads_per_chunk // max(n_block, 1))

    hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    scored = 0

    for start in range(0, n_pos, batch_size):
        end = min(start + batch_size, n_pos)
        sub_src = src[start:end]
        sub_dst = dst[start:end]
        B = sub_src.numel()

        if n_extra > 0:
            rand_idx = torch.randint(0, n_cand, (B, n_extra), device=h.device)
            rand_cands = candidate_pool[rand_idx]                  # (B, n_extra)
            cand_block = torch.cat([sub_dst.unsqueeze(1), rand_cands], dim=1)
        else:
            cand_block = sub_dst.unsqueeze(1)

        src_rep = sub_src.unsqueeze(1).expand(B, n_block).reshape(-1)
        cand_rep = cand_block.reshape(-1)
        logits = model.existence_logit(h, src_rep, cand_rep).view(B, n_block)

        true_scores = logits[:, 0]
        ranks = (logits > true_scores.unsqueeze(1)).sum(dim=1) + 1
        for k in ks:
            hits[k] += int((ranks <= k).sum().item())
        reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
        scored += B

    if scored == 0:
        out = {k: float("nan") for k in ks}
        out["mrr"] = float("nan")
        return out
    out = {k: hits[k] / scored for k in ks}
    out["mrr"] = reciprocal_rank_sum / scored
    return out


@torch.no_grad()
def _score_edge_set_temporal(
    model: FISHSTHN,
    data: FISHData,
    h: torch.Tensor,
    mask: torch.Tensor,
    n_neg_per_pos: int,
    device,
    seed: int,
    neg_state=None,
) -> dict:
    """Score a set of edges selected by ``mask`` using per-query t_query.

    Mirrors _score_edge_set from fish_rgcn but passes the edge's own
    normalised timestamp as t_query to every model call so that only training
    neighbours with t_j < t_query contribute to each pair representation.
    Hits@K is not computed here (always nan) because _hits_at_k_sthn would
    need an analogous per-query t_query that varies across the candidate block,
    which isn't supported; the tune scripts use compute_hits=False.
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    nan = float("nan")
    empty: dict = {
        "exist_auc": nan, "exist_auc_smart": nan,
        "type_macro_auc": nan, "type_top1": nan,
        "type_majority_baseline": nan, "type_classes_in_test": [],
        "type_classes_missing_from_test": list(range(data.n_edge_types)),
        "order_spearman": nan, "order_pairwise_acc": nan,
        "n_edges": 0, "mrr": nan, "mrr_type": nan,
    }
    if int(mask.sum()) == 0:
        return empty

    src = data.all_src[mask].to(device)
    dst = data.all_dst[mask].to(device)
    typ = data.all_type[mask].to(device)
    times = data.all_time[mask].to(device)

    # Normalise timestamps on training-only scale (same as _build_train_adjacency).
    t_max_tr = float(data.all_time[data.train_mask].max())
    if t_max_tr <= 0.0:
        t_max_tr = 1.0
    t_query = (times.float() / t_max_tr).clamp(min=0.0)

    # ── Existence (random negatives) ─────────────────────────────────────────
    g = torch.Generator(device=device).manual_seed(seed)
    pos_logit = model.existence_logit(h, src, dst, t_query)
    src_rep = src.repeat_interleave(n_neg_per_pos)
    t_query_rep = t_query.repeat_interleave(n_neg_per_pos)
    neg_dst_rnd = torch.randint(
        0, data.n_nodes, (src.numel() * n_neg_per_pos,), generator=g, device=device,
    )
    neg_logit_rnd = model.existence_logit(h, src_rep, neg_dst_rnd, t_query_rep)
    y_r = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit_rnd)]).cpu().numpy()
    s_r = torch.cat([pos_logit, neg_logit_rnd]).cpu().numpy()
    exist_auc = float(roc_auc_score(y_r, s_r))

    # ── Existence (smart negatives) ───────────────────────────────────────────
    exist_auc_smart = nan
    if neg_state is not None:
        rng = np.random.default_rng(seed + 12345)
        node_type_np = data.node_type.cpu().numpy()
        smart_neg_np = _sample_smart_negatives(
            src.cpu().numpy(), dst.cpu().numpy(), typ.cpu().numpy(),
            neg_state, n_neg_per_pos, data.n_nodes, rng,
            node_type_of_dst=node_type_np,
        )
        smart_neg = torch.from_numpy(smart_neg_np).to(device).reshape(-1)
        neg_logit_smt = model.existence_logit(h, src_rep, smart_neg, t_query_rep)
        y_s = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit_smt)]).cpu().numpy()
        s_s = torch.cat([pos_logit, neg_logit_smt]).cpu().numpy()
        exist_auc_smart = float(roc_auc_score(y_s, s_s))

    # ── Type ─────────────────────────────────────────────────────────────────
    type_lgt_np = model.type_logits(h, src, dst, t_query).cpu().numpy()
    type_true = typ.cpu().numpy()
    present_cls = np.unique(type_true)
    type_auc = nan
    type_auc_classes: list[int] = []
    if len(present_cls) >= 2:
        try:
            sub_lgt = type_lgt_np[:, present_cls]
            oh = (type_true[:, None] == present_cls[None, :]).astype(int)
            type_auc = float(roc_auc_score(oh, sub_lgt, average="macro", multi_class="ovr"))
            type_auc_classes = [int(c) for c in present_cls]
        except ValueError:
            pass
    type_top1 = float((type_lgt_np.argmax(axis=1) == type_true).mean())
    if len(type_true) > 0:
        counts = np.bincount(type_true, minlength=data.n_edge_types)
        type_majority = float(counts.max() / counts.sum())
    else:
        type_majority = nan

    # ── Order ─────────────────────────────────────────────────────────────────
    out_ord = model.order_output(h, src, dst, t_query)
    if model.order_mode == "regression":
        score = out_ord.squeeze(-1).cpu().numpy()
    elif model.order_mode == "corn":
        score = _corn_label_from_logits(out_ord).cpu().numpy().astype(float)
    else:  # coral
        score = (torch.sigmoid(out_ord) > 0.5).sum(dim=-1).cpu().numpy().astype(float)
    target = times.cpu().numpy()
    rho = float(spearmanr(score, target).statistic) if len(score) > 1 else nan
    rng2 = np.random.default_rng(seed)
    n_pairs = min(50_000, len(score) * (len(score) - 1) // 2)
    if n_pairs >= 1:
        ii = rng2.integers(0, len(score), n_pairs)
        jj = rng2.integers(0, len(score), n_pairs)
        keep = (ii != jj) & (target[ii] != target[jj])
        ii, jj = ii[keep], jj[keep]
        pairwise = float(((target[ii] < target[jj]) == (score[ii] < score[jj])).mean())
    else:
        pairwise = nan

    return {
        "exist_auc": exist_auc,
        "exist_auc_smart": exist_auc_smart,
        "type_macro_auc": type_auc,
        "type_top1": type_top1,
        "type_majority_baseline": type_majority,
        "type_classes_in_test": type_auc_classes,
        "type_classes_missing_from_test": [
            int(c) for c in range(data.n_edge_types) if c not in present_cls
        ],
        "order_spearman": rho,
        "order_pairwise_acc": pairwise,
        "n_edges": int(mask.sum()),
        "mrr": nan, "mrr_type": nan,
    }


def evaluate(
    model: FISHSTHN,
    data: FISHData,
    n_neg_per_pos: int = 20,
    device: Optional[str] = None,
    seed: int = 0,
    compute_hits: bool = False,
    hits_ks: tuple[int, ...] = (1, 10, 100),
    hits_batch: int = 9_000,
    max_hits_candidates: int = 300,
) -> dict:
    """Compute the three task metrics on the held-out test set(s).

    Uses per-query temporal causality: each pair (src, dst, t) sees only
    training neighbours with edge_time < t, with t as the reference for time
    deltas. This makes STHN fully faithful to its design.
    """
    with torch.no_grad():
        device = device or next(model.parameters()).device
        model.eval()
        h = model.encode(data)
        neg_state = _build_negative_sampler_state(data)

        primary = _score_edge_set_temporal(
            model, data, h, data.test_mask, n_neg_per_pos, device, seed,
            neg_state=neg_state,
        )
        out: dict = dict(primary)
        out["n_test_edges"] = primary["n_edges"]

        if data.inductive_mask is not None and int(data.inductive_mask.sum()) > 0:
            inductive = _score_edge_set_temporal(
                model, data, h, data.inductive_mask, n_neg_per_pos, device, seed + 1,
                neg_state=neg_state,
            )
            for k, v in inductive.items():
                out[f"ind_{k}"] = v
        else:
            out["ind_exist_auc"] = float("nan")
            out["ind_exist_auc_smart"] = float("nan")
            out["ind_type_top1"] = float("nan")
            out["ind_order_pairwise_acc"] = float("nan")
            out["ind_n_edges"] = 0

        seen_nodes = set(data.all_src[data.train_mask].cpu().numpy().tolist())
        seen_nodes |= set(data.all_dst[data.train_mask].cpu().numpy().tolist())
        src_np = data.all_src[data.test_mask].cpu().numpy()
        dst_np = data.all_dst[data.test_mask].cpu().numpy()
        out["test_both_endpoints_seen"] = sum(
            1 for s, d in zip(src_np, dst_np) if s in seen_nodes and d in seen_nodes
        )
        out["test_one_endpoint_seen"] = sum(
            1 for s, d in zip(src_np, dst_np)
            if (s in seen_nodes) != (d in seen_nodes)
        )
        out["test_neither_endpoint_seen"] = (
            len(src_np) - out["test_both_endpoints_seen"] - out["test_one_endpoint_seen"]
        )
        return out

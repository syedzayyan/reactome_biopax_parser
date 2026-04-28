"""
ReactomeViz — visualisation and statistics for ReactomeBioPAX NetworkX graphs.

Time model
----------
Edges carry three ordering attributes when they belong to a pathway step:

    pathway      : str   — the leaf pathway an edge was attributed to
    local_order  : int   — 0-based position within that pathway's step chain
    time         : int   — monotonic global rank across all pathways
                           (pathway sort, then local_order within pathway)

Edges that don't belong to any step (notably complex_component edges) carry
``time`` only, not ``pathway`` / ``local_order``.

Two ways to look at temporal structure:

* **Global** plots use ``time`` on the x-axis. They show the graph being
  assembled in rank order — useful for overall edge/node accumulation and
  for watching fragmentation resolve. The x-axis is a *serialised* ordering
  across pathways, not a biological clock; sibling pathways have no real
  sequence between them, so don't read biological meaning into the crossing
  of pathway boundaries.
* **Per-pathway** plots use ``local_order`` scoped by ``pathway``. They show
  trajectories within individual pathways, overlaid with a median. This is
  the biologically honest version — every pathway starts at step 0, so
  cross-pathway comparison is meaningful.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
_PALETTE = {
    "bg": "#ffffff",
    "fg": "#2e3440",
    "grid": "#e6e9ef",
    "protein": "#4c78a8",
    "molecule": "#59a14f",
    "complex": "#b279a2",
    "physical": "#f28e2b",
    "other": "#9aa1a6",
    "reaction": "#4c78a8",
    "catalysis": "#e15759",
    "reaction_partner": "#8cd17d",
    "complex_component": "#b279a2",
}

_TYPE_COLOR = {
    "protein": _PALETTE["protein"],
    "small_molecule": _PALETTE["molecule"],
    "complex": _PALETTE["complex"],
    "physical_entity": _PALETTE["physical"],
}

_EDGE_COLOR = {
    "reaction": _PALETTE["reaction"],
    "catalysis": _PALETTE["catalysis"],
    "reaction_partner": _PALETTE["reaction_partner"],
    "complex_component": _PALETTE["complex_component"],
    "complex": _PALETTE["complex_component"],
    "translocation": "#76b7b2",
    "expression": "#af7aa1",
    "left_reactant": "#bab0ab",
    "right_product": "#d4a6a6",
}

_PAPER_RC = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


# ── helpers ───────────────────────────────────────────────────────────────────


def _style(fig: plt.Figure, axes, fontsize: int = 8):
    fig.patch.set_facecolor(_PALETTE["bg"])
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(_PALETTE["bg"])
        ax.tick_params(colors=_PALETTE["fg"], labelsize=fontsize - 1)
        ax.xaxis.label.set_color(_PALETTE["fg"])
        ax.yaxis.label.set_color(_PALETTE["fg"])
        ax.title.set_color(_PALETTE["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(_PALETTE["grid"])
        ax.grid(color=_PALETTE["grid"], linewidth=0.5, linestyle="--", axis="x")


def _panel_label(ax: plt.Axes, label: str, fontsize: int = 11):
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
        color=_PALETTE["fg"],
    )


def _percentile_limits(
    values: np.ndarray,
    lower_pct: float = 0.0,
    upper_pct: float = 99.0,
) -> tuple[float, float]:
    lo = float(np.percentile(values, lower_pct))
    hi = float(np.percentile(values, upper_pct))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    return max(0.0, lo - pad), hi + pad


def _empty_panel(ax, fig, msg: str, fontsize: int, show: bool):
    ax.text(
        0.5,
        0.5,
        msg,
        ha="center",
        va="center",
        color=_PALETTE["fg"],
        fontsize=fontsize,
        transform=ax.transAxes,
    )
    _style(fig, ax, fontsize=fontsize)
    if show:
        plt.tight_layout()
        plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════


class ReactomeViz:
    """
    Visualisation and statistics for a ReactomeBioPAX NetworkX DiGraph.
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self._precompute()

    # ── pre-computation ───────────────────────────────────────────────────────

    def _precompute(self):
        G = self.G

        self._edges_by_type: dict[str, list] = defaultdict(list)
        # Only accept int-typed time values — guards against stray tuples from
        # legacy runs or complex-edge builders that haven't been updated.
        self._edges_by_time: dict[int, list] = defaultdict(list)
        self._edges_by_pathway: dict[str, dict[int, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._reaction_nodes: dict[tuple, set] = defaultdict(set)

        for u, v, d in G.edges(data=True):
            etype = d.get("type", "other")
            self._edges_by_type[etype].append((u, v, d))

            t = d.get("time")
            if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
                self._edges_by_time[int(t)].append((u, v, d))

            pid = d.get("pathway")
            lo = d.get("local_order")
            if pid is not None and isinstance(lo, (int, np.integer)):
                self._edges_by_pathway[pid][int(lo)].append((u, v, d))
                if etype == "reaction":
                    self._reaction_nodes[(pid, int(lo))].update([u, v])

        self._nodes_by_type: dict[str, list] = defaultdict(list)
        for n, d in G.nodes(data=True):
            self._nodes_by_type[d.get("type", "other")].append(n)

        self._node_first_seen: dict[str, int] = {}
        for t in sorted(self._edges_by_time):
            for u, v, _ in self._edges_by_time[t]:
                self._node_first_seen.setdefault(u, t)
                self._node_first_seen.setdefault(v, t)

        self._location_counts: Counter = Counter()
        for _, d in G.nodes(data=True):
            loc = d.get("cellularLocation")
            if isinstance(loc, dict):
                self._location_counts[loc.get("common_name", "unknown")] += 1
            elif loc:
                self._location_counts[str(loc)] += 1
            else:
                self._location_counts["unknown"] += 1

    # ── overall stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        G = self.G
        degrees = dict(G.degree())
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        chain_lens = [
            max(locals_map) + 1
            for locals_map in self._edges_by_pathway.values()
            if locals_map
        ]
        orphan_edges = sum(
            1
            for _, _, d in G.edges(data=True)
            if d.get("pathway") is None or d.get("local_order") is None
        )

        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "weakly_connected_components": nx.number_weakly_connected_components(G),
            "strongly_connected_components": nx.number_strongly_connected_components(G),
            "avg_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
            "max_degree": max(degrees.values()) if degrees else 0,
            "max_degree_node": max(degrees, key=degrees.get) if degrees else "n/a",
            "avg_in_degree": float(np.mean(list(in_deg.values()))) if in_deg else 0.0,
            "avg_out_degree": float(np.mean(list(out_deg.values())))
            if out_deg
            else 0.0,
            "node_types": {k: len(v) for k, v in self._nodes_by_type.items()},
            "edge_types": {k: len(v) for k, v in self._edges_by_type.items()},
            "pathways_with_steps": len(chain_lens),
            "steps_per_pathway_min": min(chain_lens) if chain_lens else 0,
            "steps_per_pathway_median": float(np.median(chain_lens))
            if chain_lens
            else 0.0,
            "steps_per_pathway_max": max(chain_lens) if chain_lens else 0,
            "global_ranks_used": len(self._edges_by_time),
            "orphan_edges": orphan_edges,
            "top_hubs": sorted(degrees.items(), key=lambda x: -x[1])[:10],
        }

    def print_stats(self):
        s = self.stats()
        sep = "─" * 62
        print(f"\n{'REACTOME GRAPH STATISTICS':^62}")
        print(sep)
        print(f"  Nodes                  {s['nodes']:>10,}")
        print(f"  Edges                  {s['edges']:>10,}")
        print(f"  Density                {s['density']:>10.6f}")
        print(f"  Is DAG                 {str(s['is_dag']):>10}")
        print(f"  Weakly conn. comps     {s['weakly_connected_components']:>10,}")
        print(f"  Strongly conn. comps   {s['strongly_connected_components']:>10,}")
        print(f"  Avg degree             {s['avg_degree']:>10.2f}")
        print(
            f"  Max degree             {s['max_degree']:>10,}  ({s['max_degree_node']})"
        )
        print(sep)
        print(f"  Pathways with steps    {s['pathways_with_steps']:>10,}")
        print(
            f"  Steps per pathway      "
            f"min {s['steps_per_pathway_min']:>4,}   "
            f"median {s['steps_per_pathway_median']:>6.1f}   "
            f"max {s['steps_per_pathway_max']:>5,}"
        )
        print(f"  Global ranks used      {s['global_ranks_used']:>10,}")
        print(f"  Edges with no pathway  {s['orphan_edges']:>10,}")
        print(sep)
        print("  Node types:")
        for k, v in sorted(s["node_types"].items(), key=lambda x: -x[1]):
            print(f"    {k:<22} {v:>6,}")
        print("  Edge types:")
        for k, v in sorted(s["edge_types"].items(), key=lambda x: -x[1]):
            print(f"    {k:<22} {v:>6,}")
        print(sep)
        print("  Top 10 hubs:")
        for name, deg in s["top_hubs"]:
            label = name if len(name) <= 40 else name[:37] + "..."
            print(f"    {label:<40} {deg:>5,}")
        print(sep + "\n")

    # ── global temporal plots ─────────────────────────────────────────────────

    def plot_edge_growth(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Cumulative edge count over global rank, stacked by edge type."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        if not self._edges_by_time:
            return _empty_panel(ax, fig, "No ranked edges found", fontsize, show)

        ranks = sorted(self._edges_by_time)
        edge_types = sorted(
            self._edges_by_type, key=lambda t: -len(self._edges_by_type[t])
        )

        running = {et: 0 for et in edge_types}
        cumulative = {et: [] for et in edge_types}

        for r in ranks:
            counts = Counter(
                d.get("type", "other") for _, _, d in self._edges_by_time[r]
            )
            for et in edge_types:
                running[et] += counts.get(et, 0)
                cumulative[et].append(running[et])

        bottom = np.zeros(len(ranks))
        for et in edge_types:
            vals = np.array(cumulative[et])
            if vals[-1] == 0:
                continue
            color = _EDGE_COLOR.get(et, _PALETTE["other"])
            ax.fill_between(
                ranks, bottom, bottom + vals, alpha=0.75, color=color, label=et
            )
            bottom = bottom + vals

        ax.set_xlabel("Global rank (serialised across pathways)", fontsize=fontsize)
        ax.set_ylabel("Cumulative edges", fontsize=fontsize)
        ax.set_title("Edge Growth by Type", fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 1, framealpha=0.2, labelcolor=_PALETTE["fg"])
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_node_recruitment(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Per-rank new node count (bars) + cumulative node count (line)."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        if not self._node_first_seen:
            return _empty_panel(ax, fig, "No ranked nodes found", fontsize, show)

        rank_counts: Counter = Counter(self._node_first_seen.values())
        ranks = sorted(rank_counts)
        counts = [rank_counts[r] for r in ranks]
        cumulative = np.cumsum(counts)

        ax2 = ax.twinx()
        ax.bar(
            ranks,
            counts,
            color=_PALETTE["protein"],
            alpha=0.5,
            width=1.0,
            label="New nodes per rank",
        )
        ax2.plot(
            ranks,
            cumulative,
            color=_PALETTE["catalysis"],
            linewidth=1.5,
            label="Cumulative nodes",
        )

        ax.set_xlabel("Global rank", fontsize=fontsize)
        ax.set_ylabel("New nodes", fontsize=fontsize, color=_PALETTE["protein"])
        ax2.set_ylabel(
            "Cumulative nodes", fontsize=fontsize, color=_PALETTE["catalysis"]
        )
        ax.set_title("Node Recruitment Over Global Rank", fontsize=fontsize + 1)
        ax2.tick_params(colors=_PALETTE["fg"], labelsize=fontsize - 1)
        ax2.yaxis.label.set_color(_PALETTE["fg"])
        _style(fig, ax, fontsize=fontsize)
        ax2.set_facecolor(_PALETTE["bg"])
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_connected_components_over_rank(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Weakly-connected component count as edges are inserted in rank order."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        if not self._edges_by_time:
            return _empty_panel(ax, fig, "No ranked edges found", fontsize, show)

        ranks = sorted(self._edges_by_time)
        G_running = nx.DiGraph()
        component_counts = []

        for r in ranks:
            for u, v, d in self._edges_by_time[r]:
                G_running.add_edge(u, v, **d)
            component_counts.append(nx.number_weakly_connected_components(G_running))

        ax.plot(
            ranks, component_counts, color=_PALETTE["reaction_partner"], linewidth=1.5
        )
        ax.fill_between(
            ranks, component_counts, alpha=0.2, color=_PALETTE["reaction_partner"]
        )
        ax.set_xlabel("Global rank (serialised across pathways)", fontsize=fontsize)
        ax.set_ylabel("Weakly connected components", fontsize=fontsize)
        ax.set_title("Network Fragmentation During Assembly", fontsize=fontsize + 1)
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    # ── per-pathway plots ─────────────────────────────────────────────────────

    def plot_edge_growth_per_pathway(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
        max_step: Optional[int] = None,
    ) -> plt.Figure:
        """Cumulative edges vs local step, overlaid for every pathway."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        if not self._edges_by_pathway:
            return _empty_panel(
                ax, fig, "No pathway-scoped edges found", fontsize, show
            )

        observed_max = max(
            max(locals_map)
            for locals_map in self._edges_by_pathway.values()
            if locals_map
        )
        if max_step is not None:
            observed_max = min(observed_max, max_step)
        xs = list(range(observed_max + 1))

        curves = []
        for pid, locals_map in self._edges_by_pathway.items():
            running = 0
            cum = []
            for i in xs:
                running += len(locals_map.get(i, []))
                cum.append(running)
            curves.append(cum)
            ax.plot(xs, cum, color=_PALETTE["protein"], alpha=0.08, linewidth=0.7)

        arr = np.array(curves)
        median = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        ax.fill_between(
            xs, p25, p75, color=_PALETTE["catalysis"], alpha=0.18, label="25–75% band"
        )
        ax.plot(
            xs,
            median,
            color=_PALETTE["catalysis"],
            linewidth=1.8,
            label="Median pathway",
        )

        ax.set_xlabel("Local step within pathway", fontsize=fontsize)
        ax.set_ylabel("Cumulative edges", fontsize=fontsize)
        ax.set_title("Edge Growth per Pathway (overlay)", fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 1, framealpha=0.2, labelcolor=_PALETTE["fg"])
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_pathway_size_distribution(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Histogram of pathway chain lengths."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(7, 4)) if show else (ax.get_figure(), ax)

        chain_lens = [
            max(locals_map) + 1
            for locals_map in self._edges_by_pathway.values()
            if locals_map
        ]
        if not chain_lens:
            return _empty_panel(ax, fig, "No pathway step chains found", fontsize, show)

        top = max(chain_lens)
        if top <= 1:
            bin_edges = np.array([0.5, 1.5, 2.5])
        else:
            bin_edges = np.logspace(0, np.log10(top + 1), 25)
        ax.hist(
            chain_lens,
            bins=bin_edges,
            color=_PALETTE["molecule"],
            alpha=0.85,
            edgecolor=_PALETTE["bg"],
            linewidth=0.5,
        )
        if top > 1:
            ax.set_xscale("log")

        med = float(np.median(chain_lens))
        ax.axvline(
            med,
            color=_PALETTE["catalysis"],
            linewidth=1.2,
            linestyle="--",
            label=f"median = {med:.1f}",
        )

        ax.set_xlabel(
            "Steps per pathway" + (" (log scale)" if top > 1 else ""), fontsize=fontsize
        )
        ax.set_ylabel("Pathway count", fontsize=fontsize)
        ax.set_title("Pathway Size Distribution", fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 1, framealpha=0.2, labelcolor=_PALETTE["fg"])
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_edges_per_pathway(
        self,
        top_n: int = 20,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Top-N pathways by edge count."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 5)) if show else (ax.get_figure(), ax)

        pathway_edge_counts = Counter()
        for pid, locals_map in self._edges_by_pathway.items():
            pathway_edge_counts[pid] = sum(len(v) for v in locals_map.values())

        if not pathway_edge_counts:
            return _empty_panel(
                ax, fig, "No pathway-scoped edges found", fontsize, show
            )

        most_common = pathway_edge_counts.most_common(top_n)
        labels = [p for p, _ in most_common]
        counts = [c for _, c in most_common]
        short = [lbl[:34] + "…" if len(lbl) > 36 else lbl for lbl in labels]

        ax.barh(range(len(labels)), counts, color=_PALETTE["protein"], height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(short, fontsize=fontsize - 1)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
        ax.invert_yaxis()
        ax.set_xlabel("Edges contributed", fontsize=fontsize)
        ax.set_title(f"Top {top_n} Pathways by Edge Count", fontsize=fontsize + 1)
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    # ── topology / structural plots ───────────────────────────────────────────

    def plot_degree_distribution(
        self,
        log_scale: bool = True,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        show = ax is None
        fig, ax = plt.subplots(figsize=(7, 4)) if show else (ax.get_figure(), ax)

        degrees = [d for _, d in self.G.degree()]
        counts = Counter(degrees)
        xs = sorted(counts)
        ys = [counts[x] for x in xs]

        ax.scatter(xs, ys, s=14, color=_PALETTE["protein"], alpha=0.8, zorder=3)
        ax.plot(xs, ys, color=_PALETTE["protein"], alpha=0.3, linewidth=0.8)

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel("Degree", fontsize=fontsize)
        ax.set_ylabel("Count", fontsize=fontsize)
        ax.set_title(
            "Degree Distribution" + (" (log–log)" if log_scale else ""),
            fontsize=fontsize + 1,
        )
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_in_vs_out_degree(
        self,
        remove_outliers: bool = True,
        upper_percentile: float = 99.0,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        show = ax is None
        fig, ax = plt.subplots(figsize=(7, 6)) if show else (ax.get_figure(), ax)

        type_data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        all_in: list[int] = []
        all_out: list[int] = []

        for n, d in self.G.nodes(data=True):
            ntype = d.get("type", "other")
            in_d = self.G.in_degree(n)
            out_d = self.G.out_degree(n)
            type_data[ntype][0].append(in_d)
            type_data[ntype][1].append(out_d)
            all_in.append(in_d)
            all_out.append(out_d)

        for ntype, (xs, ys) in type_data.items():
            color = _TYPE_COLOR.get(ntype, _PALETTE["other"])
            ax.scatter(xs, ys, s=10, alpha=0.55, color=color, label=ntype)

        if remove_outliers and all_in and all_out:
            x_lo, x_hi = _percentile_limits(np.array(all_in), 0.0, upper_percentile)
            y_lo, y_hi = _percentile_limits(np.array(all_out), 0.0, upper_percentile)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(
            [0, lim],
            [0, lim],
            color=_PALETTE["grid"],
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

        ax.set_xlabel("In-degree", fontsize=fontsize)
        ax.set_ylabel("Out-degree", fontsize=fontsize)
        ax.set_title("In-degree vs Out-degree by Entity Type", fontsize=fontsize + 1)
        ax.tick_params(labelsize=fontsize - 1)
        ax.legend(fontsize=fontsize - 1, framealpha=0.2, labelcolor=_PALETTE["fg"])

        if remove_outliers:
            x_hi = ax.get_xlim()[1]
            y_hi = ax.get_ylim()[1]
            n_clipped = sum(
                1 for xi, yi in zip(all_in, all_out) if xi > x_hi or yi > y_hi
            )
            if n_clipped:
                ax.text(
                    0.98,
                    0.02,
                    f"{n_clipped} outlier{'s' if n_clipped != 1 else ''} not shown",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=max(fontsize - 2, 6),
                    color=_PALETTE["other"],
                    style="italic",
                )

        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_node_type_breakdown(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        show = ax is None
        fig, ax = plt.subplots(figsize=(7, 3.5)) if show else (ax.get_figure(), ax)

        types = sorted(self._nodes_by_type, key=lambda t: len(self._nodes_by_type[t]))
        counts = [len(self._nodes_by_type[t]) for t in types]
        colors = [_TYPE_COLOR.get(t, _PALETTE["other"]) for t in types]

        bars = ax.barh(types, counts, color=colors, height=0.55)
        for bar, val in zip(bars, counts):
            ax.text(
                val + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}",
                va="center",
                color=_PALETTE["fg"],
                fontsize=fontsize - 1,
            )

        ax.set_xlabel("Node count", fontsize=fontsize)
        ax.set_title("Node Types", fontsize=fontsize + 1)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize - 1, rotation=0, ha="right")
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_edge_type_breakdown(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        show = ax is None
        fig, ax = plt.subplots(figsize=(7, 3.5)) if show else (ax.get_figure(), ax)

        types = sorted(self._edges_by_type, key=lambda t: len(self._edges_by_type[t]))
        counts = [len(self._edges_by_type[t]) for t in types]
        colors = [_EDGE_COLOR.get(t, _PALETTE["other"]) for t in types]

        bars = ax.barh(types, counts, color=colors, height=0.55)
        for bar, val in zip(bars, counts):
            ax.text(
                val + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}",
                va="center",
                color=_PALETTE["fg"],
                fontsize=fontsize - 1,
            )

        ax.set_xlabel("Edge count", fontsize=fontsize)
        ax.set_title("Edge Types", fontsize=fontsize + 1)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize - 1, rotation=0, ha="right")
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_cellular_location(
        self,
        top_n: int = 12,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        most_common = self._location_counts.most_common(top_n)
        labels = [lbl for lbl, _ in most_common]
        counts = [c for _, c in most_common]

        cmap = plt.get_cmap("cool", len(labels) or 1)
        colors = [cmap(i) for i in range(len(labels))]

        ax.bar(range(len(labels)), counts, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=fontsize - 1)
        ax.set_ylabel("Node count", fontsize=fontsize)
        ax.set_title(f"Top {top_n} Cellular Locations", fontsize=fontsize + 1)
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_catalyst_reuse(
        self,
        top_n: int = 20,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 5)) if show else (ax.get_figure(), ax)

        catalyst_counts: Counter = Counter()
        for u, _, d in self.G.edges(data=True):
            if d.get("type") == "catalysis":
                catalyst_counts[u] += 1

        most_common = catalyst_counts.most_common(top_n)
        if not most_common:
            return _empty_panel(ax, fig, "No catalysis edges found", fontsize, show)

        labels = [n for n, _ in most_common]
        counts = [c for _, c in most_common]
        short_labels = [lbl[:28] + "…" if len(lbl) > 30 else lbl for lbl in labels]

        ax.barh(range(len(labels)), counts, color=_PALETTE["catalysis"], height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(short_labels, fontsize=fontsize - 1)
        plt.setp(ax.get_yticklabels(), rotation=15, ha="right", fontsize=fontsize - 1)
        ax.invert_yaxis()
        ax.set_xlabel("Reactions catalysed", fontsize=fontsize)
        ax.set_title(f"Top {top_n} Most Reused Catalysts", fontsize=fontsize + 1)
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_reaction_size_distribution(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Histogram of reaction sizes (unique nodes per reaction step).

        A reaction step here is a (pathway, local_order) pair — the actual
        biological reaction event.
        """
        show = ax is None
        fig, ax = plt.subplots(figsize=(7, 4)) if show else (ax.get_figure(), ax)

        sizes = [len(nodes) for nodes in self._reaction_nodes.values()]
        if not sizes:
            return _empty_panel(ax, fig, "No reaction edges found", fontsize, show)

        bins = range(min(sizes), max(sizes) + 2)
        ax.hist(
            sizes,
            bins=bins,
            color=_PALETTE["molecule"],
            alpha=0.85,
            edgecolor=_PALETTE["bg"],
            linewidth=0.5,
        )

        ax.set_xlabel("Unique nodes per reaction", fontsize=fontsize)
        ax.set_ylabel("Count", fontsize=fontsize)
        ax.set_title("Reaction Size Distribution", fontsize=fontsize + 1)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=fontsize - 1)
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    # ── paper dashboard ───────────────────────────────────────────────────────

    def dashboard(self, figsize=(18, 28), fontsize: int = 11) -> plt.Figure:
        """
        Render all plots in a paper-friendly dashboard.

        Layout (4 columns × 6 rows)
        ------
        row 0:  a  edge growth — global (cols 0-2)        b  pathway size dist  (col 3)
        row 1:  c  edge growth — per pathway (cols 0-2)   d  node recruitment   (col 3)
        row 2:  e  node types (col 0)   f  edge types (col 1)   g  degree dist  (cols 2-3)
        row 3:  h  in vs out (cols 0-1)                   i  cellular location  (cols 2-3)
        row 4:  j  catalyst reuse (cols 0-2)              k  reaction size      (col 3)
        row 5:  l  edges per pathway (cols 0-2)           m  components vs rank (col 3)
        """
        with plt.rc_context(_PAPER_RC):
            fig = plt.figure(figsize=figsize)
            fig.patch.set_facecolor(_PALETTE["bg"])

            gs = fig.add_gridspec(
                6,
                4,
                hspace=0.78,
                wspace=0.48,
                left=0.09,
                right=0.97,
                top=0.97,
                bottom=0.05,
            )

            ax_eg_global = fig.add_subplot(gs[0, :3])
            ax_pw_size = fig.add_subplot(gs[0, 3])
            ax_eg_pw = fig.add_subplot(gs[1, :3])
            ax_nrecruit = fig.add_subplot(gs[1, 3])
            ax_ntypes = fig.add_subplot(gs[2, 0])
            ax_etypes = fig.add_subplot(gs[2, 1])
            ax_degree = fig.add_subplot(gs[2, 2:])
            ax_inout = fig.add_subplot(gs[3, :2])
            ax_loc = fig.add_subplot(gs[3, 2:])
            ax_cat = fig.add_subplot(gs[4, :3])
            ax_rxn = fig.add_subplot(gs[4, 3])
            ax_edges_pw = fig.add_subplot(gs[5, :3])
            ax_comp = fig.add_subplot(gs[5, 3])

            self.plot_edge_growth(ax=ax_eg_global, fontsize=fontsize)
            self.plot_pathway_size_distribution(ax=ax_pw_size, fontsize=fontsize)
            self.plot_edge_growth_per_pathway(ax=ax_eg_pw, fontsize=fontsize)
            self.plot_node_recruitment(ax=ax_nrecruit, fontsize=fontsize)
            self.plot_node_type_breakdown(ax=ax_ntypes, fontsize=fontsize)
            self.plot_edge_type_breakdown(ax=ax_etypes, fontsize=fontsize)
            self.plot_degree_distribution(ax=ax_degree, fontsize=fontsize)
            self.plot_in_vs_out_degree(ax=ax_inout, fontsize=fontsize)
            self.plot_cellular_location(top_n=10, ax=ax_loc, fontsize=fontsize)
            self.plot_catalyst_reuse(ax=ax_cat, fontsize=fontsize)
            self.plot_reaction_size_distribution(ax=ax_rxn, fontsize=fontsize)
            self.plot_edges_per_pathway(ax=ax_edges_pw, fontsize=fontsize)
            self.plot_connected_components_over_rank(ax=ax_comp, fontsize=fontsize)

            for ax, label in zip(
                [
                    ax_eg_global,
                    ax_pw_size,
                    ax_eg_pw,
                    ax_nrecruit,
                    ax_ntypes,
                    ax_etypes,
                    ax_degree,
                    ax_inout,
                    ax_loc,
                    ax_cat,
                    ax_rxn,
                    ax_edges_pw,
                    ax_comp,
                ],
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
            ):
                _panel_label(ax, label, fontsize=fontsize)

        return fig

"""
ReactomeViz — visualisation and statistics for ReactomeBioPAX NetworkX graphs.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def _style(fig: plt.Figure, axes, fontsize: int = 8):
    """Apply clean white theme to figure and one or more axes."""
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
    """Add a bold lowercase panel label to the top-left of an axis."""
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


# ══════════════════════════════════════════════════════════════════════════════


class ReactomeViz:
    """
    Visualisation and statistics for a ReactomeBioPAX NetworkX DiGraph.

    Parameters
    ----------
    G : nx.DiGraph
        Output of ``ReactomeBioPAX.parse_biopax_into_networkx``.
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self._precompute()

    # ── pre-computation ───────────────────────────────────────────────────────

    def _precompute(self):
        G = self.G

        self._edges_by_type: dict[str, list] = defaultdict(list)
        self._edges_by_time: dict[int, list] = defaultdict(list)
        for u, v, d in G.edges(data=True):
            t = d.get("type", "other")
            step = d.get("time")
            self._edges_by_type[t].append((u, v, d))
            if step is not None:
                self._edges_by_time[step].append((u, v, d))

        self._nodes_by_type: dict[str, list] = defaultdict(list)
        for n, d in G.nodes(data=True):
            self._nodes_by_type[d.get("type", "other")].append(n)

        self._node_first_seen: dict[str, int] = {}
        for step in sorted(self._edges_by_time):
            for u, v, _ in self._edges_by_time[step]:
                self._node_first_seen.setdefault(u, step)
                self._node_first_seen.setdefault(v, step)

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
        """Return a dictionary of summary statistics."""
        G = self.G
        degrees = dict(G.degree())
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "weakly_connected_components": nx.number_weakly_connected_components(G),
            "strongly_connected_components": nx.number_strongly_connected_components(G),
            "avg_degree": np.mean(list(degrees.values())),
            "max_degree": max(degrees.values()),
            "max_degree_node": max(degrees, key=degrees.get),
            "avg_in_degree": np.mean(list(in_deg.values())),
            "avg_out_degree": np.mean(list(out_deg.values())),
            "node_types": {k: len(v) for k, v in self._nodes_by_type.items()},
            "edge_types": {k: len(v) for k, v in self._edges_by_type.items()},
            "reaction_steps": len(self._edges_by_time),
            "top_hubs": sorted(degrees.items(), key=lambda x: -x[1])[:10],
        }

    def print_stats(self):
        """Pretty-print summary statistics to stdout."""
        s = self.stats()
        sep = "─" * 52
        print(f"\n{'REACTOME GRAPH STATISTICS':^52}")
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
        print(f"  Reaction steps         {s['reaction_steps']:>10,}")
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
            label = name if len(name) <= 30 else name[:27] + "..."
            print(f"    {label:<30} {deg:>5,}")
        print(sep + "\n")

    # ── individual plots ──────────────────────────────────────────────────────

    def plot_edge_growth(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Cumulative edge count over reaction steps, broken down by edge type."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        steps = sorted(self._edges_by_time)
        edge_types = list(self._edges_by_type)

        cumulative: dict[str, list] = {et: [] for et in edge_types}
        running: dict[str, int] = {et: 0 for et in edge_types}

        for step in steps:
            step_type_counts: Counter = Counter(
                d.get("type", "other") for _, _, d in self._edges_by_time[step]
            )
            for et in edge_types:
                running[et] += step_type_counts.get(et, 0)
                cumulative[et].append(running[et])

        bottom = np.zeros(len(steps))
        for et in edge_types:
            vals = np.array(cumulative[et])
            color = _EDGE_COLOR.get(et, _PALETTE["other"])
            ax.fill_between(
                steps, bottom, bottom + vals, alpha=0.7, label=et, color=color
            )
            bottom = bottom + vals

        ax.set_xlabel("Reaction step", fontsize=fontsize)
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
        """How many new nodes appear at each reaction step."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        step_counts: Counter = Counter(self._node_first_seen.values())
        steps = sorted(step_counts)
        counts = [step_counts[s] for s in steps]
        cumulative = np.cumsum(counts)

        ax2 = ax.twinx()
        ax.bar(
            steps,
            counts,
            color=_PALETTE["protein"],
            alpha=0.5,
            width=0.8,
            label="New nodes per step",
        )
        ax2.plot(
            steps,
            cumulative,
            color=_PALETTE["catalysis"],
            linewidth=1.5,
            label="Cumulative nodes",
        )

        ax.set_xlabel("Reaction step", fontsize=fontsize)
        ax.set_ylabel("New nodes", fontsize=fontsize, color=_PALETTE["protein"])
        ax2.set_ylabel("Cumulative nodes", fontsize=fontsize, color=_PALETTE["catalysis"])
        ax.set_title("Node Recruitment Over Time", fontsize=fontsize + 1)
        ax2.tick_params(colors=_PALETTE["fg"], labelsize=fontsize - 1)
        ax2.yaxis.label.set_color(_PALETTE["fg"])
        _style(fig, ax, fontsize=fontsize)
        ax2.set_facecolor(_PALETTE["bg"])
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_degree_distribution(
        self,
        log_scale: bool = True,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Degree distribution, optionally on log-log axes."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(7, 4)) if show else (ax.get_figure(), ax)

        degrees = sorted([d for _, d in self.G.degree()], reverse=True)
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
        """
        Scatter of in-degree vs out-degree, coloured by node type.

        Parameters
        ----------
        remove_outliers : bool
            Clip axis limits at ``upper_percentile`` to avoid hub compression.
        upper_percentile : float
            Percentile for the upper axis limit (default 99).
        fontsize : int
            Font size for labels, ticks, title, and legend (default 9).
        ax : plt.Axes, optional
            Existing axes to draw into. If None a new figure is created.
        """
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
            [0, lim], [0, lim],
            color=_PALETTE["grid"], linewidth=0.8, linestyle="--", zorder=0,
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
                    0.98, 0.02,
                    f"{n_clipped} outlier{'s' if n_clipped != 1 else ''} not shown",
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=max(fontsize - 2, 6),
                    color=_PALETTE["other"], style="italic",
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
        """Horizontal bar chart of node counts per entity type."""
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
        # y-tick labels are the type names — angle them so they never clip
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
        """Horizontal bar chart of edge counts per edge type."""
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
        """Bar chart of node counts per cellular location."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        most_common = self._location_counts.most_common(top_n)
        labels = [lbl for lbl, _ in most_common]
        counts = [c for _, c in most_common]

        cmap = plt.cm.get_cmap("cool", len(labels))
        colors = [cmap(i) for i in range(len(labels))]

        ax.bar(range(len(labels)), counts, color=colors)
        ax.set_xticks(range(len(labels)))
        # Angle x-tick labels so long location names never collide
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=fontsize - 1)
        ax.set_ylabel("Node count", fontsize=fontsize)
        ax.set_title(f"Top {top_n} Cellular Locations", fontsize=fontsize + 1)
        _style(fig, ax, fontsize=fontsize)
        # Extra bottom padding to accommodate rotated labels
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
        """Bar chart of catalysts ranked by number of reactions they control."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 5)) if show else (ax.get_figure(), ax)

        catalyst_counts: Counter = Counter()
        for u, v, d in self.G.edges(data=True):
            if d.get("type") == "catalysis":
                catalyst_counts[u] += 1

        most_common = catalyst_counts.most_common(top_n)
        if not most_common:
            ax.text(
                0.5, 0.5, "No catalysis edges found",
                ha="center", va="center",
                color=_PALETTE["fg"], fontsize=fontsize,
                transform=ax.transAxes,
            )
            _style(fig, ax, fontsize=fontsize)
            if show:
                plt.tight_layout()
                plt.show()
            return fig

        labels = [n for n, _ in most_common]
        counts = [c for _, c in most_common]
        # Shorten long catalyst names and angle them slightly on the y-axis
        short_labels = [lbl[:28] + "…" if len(lbl) > 30 else lbl for lbl in labels]

        ax.barh(range(len(labels)), counts, color=_PALETTE["catalysis"], height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(short_labels, fontsize=fontsize - 1)
        # Rotate y-tick labels at a gentle angle so overlapping names separate
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
        """Histogram of reaction sizes (number of unique nodes per reaction step)."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(7, 4)) if show else (ax.get_figure(), ax)

        step_nodes: dict[int, set] = defaultdict(set)
        for u, v, d in self.G.edges(data=True):
            if d.get("type") == "reaction":
                step = d.get("time", -1)
                step_nodes[step].update([u, v])

        sizes = [len(nodes) for nodes in step_nodes.values()]
        if not sizes:
            ax.text(
                0.5, 0.5, "No reaction edges found",
                ha="center", va="center",
                color=_PALETTE["fg"], fontsize=fontsize,
                transform=ax.transAxes,
            )
        else:
            bins = range(min(sizes), max(sizes) + 2)
            ax.hist(
                sizes, bins=bins,
                color=_PALETTE["molecule"], alpha=0.8,
                edgecolor=_PALETTE["bg"], linewidth=0.5,
            )

        ax.set_xlabel("Unique nodes per reaction step", fontsize=fontsize)
        ax.set_ylabel("Count", fontsize=fontsize)
        ax.set_title("Reaction Size Distribution", fontsize=fontsize + 1)
        # Angle x-ticks so numeric labels don't crowd at larger fontsizes
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=fontsize - 1)
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    def plot_connected_components_over_time(
        self,
        fontsize: int = 9,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """How the number of weakly connected components evolves over time."""
        show = ax is None
        fig, ax = plt.subplots(figsize=(9, 4)) if show else (ax.get_figure(), ax)

        steps = sorted(self._edges_by_time)
        G_running = nx.DiGraph()
        component_counts = []

        for step in steps:
            for u, v, d in self._edges_by_time[step]:
                G_running.add_edge(u, v, **d)
            component_counts.append(nx.number_weakly_connected_components(G_running))

        ax.plot(steps, component_counts, color=_PALETTE["reaction_partner"], linewidth=1.5)
        ax.fill_between(steps, component_counts, alpha=0.2, color=_PALETTE["reaction_partner"])
        ax.set_xlabel("Reaction step", fontsize=fontsize)
        ax.set_ylabel("Weakly connected components", fontsize=fontsize)
        ax.set_title("Network Fragmentation Over Time", fontsize=fontsize + 1)
        _style(fig, ax, fontsize=fontsize)
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    # ── paper dashboard ───────────────────────────────────────────────────────

    def dashboard(self, figsize=(18, 24), fontsize: int = 11) -> plt.Figure:
        """
        Render all plots in a paper-friendly dashboard.

        Layout (4 columns × 5 rows)
        ------
        row 0:  a  edge growth (cols 0-2)       node recruitment (col 3)
        row 1:  b  node types (col 0)   edge types (col 1)   degree dist (cols 2-3)
        row 2:  c  in vs out (cols 0-1)          cellular location (cols 2-3)
        row 3:  d  catalyst reuse (cols 0-2)     reaction size (col 3)
        row 4:     connected components (all 4)

        Parameters
        ----------
        figsize : tuple
            Overall figure dimensions (width, height) in inches.
        fontsize : int
            Base font size propagated to all subplots, labels, ticks, legends,
            and panel letters (default 11).

        Returns
        -------
        matplotlib.figure.Figure
        """
        with plt.rc_context(_PAPER_RC):
            fig = plt.figure(figsize=figsize)
            fig.patch.set_facecolor(_PALETTE["bg"])

            # Extra bottom/left margin to absorb rotated tick labels at larger
            # fontsizes; hspace increased so row titles don't overlap x-ticks
            gs = fig.add_gridspec(
                5, 4,
                hspace=0.72,
                wspace=0.48,
                left=0.09,
                right=0.97,
                top=0.96,
                bottom=0.06,
            )

            # ── row 0 ──────────────────────────────────────────────────────
            ax_edge_growth  = fig.add_subplot(gs[0, :3])
            ax_node_recruit = fig.add_subplot(gs[0, 3])
            # ── row 1 ──────────────────────────────────────────────────────
            ax_node_types   = fig.add_subplot(gs[1, 0])
            ax_edge_types   = fig.add_subplot(gs[1, 1])
            ax_degree       = fig.add_subplot(gs[1, 2:])
            # ── row 2 ──────────────────────────────────────────────────────
            ax_in_out       = fig.add_subplot(gs[2, :2])
            ax_location     = fig.add_subplot(gs[2, 2:])
            # ── row 3 ──────────────────────────────────────────────────────
            ax_catalyst     = fig.add_subplot(gs[3, :3])
            ax_rxn_size     = fig.add_subplot(gs[3, 3])
            # ── row 4 ──────────────────────────────────────────────────────
            ax_components   = fig.add_subplot(gs[4, :])

            self.plot_edge_growth(ax=ax_edge_growth, fontsize=fontsize)
            self.plot_node_recruitment(ax=ax_node_recruit, fontsize=fontsize)
            self.plot_node_type_breakdown(ax=ax_node_types, fontsize=fontsize)
            self.plot_edge_type_breakdown(ax=ax_edge_types, fontsize=fontsize)
            self.plot_degree_distribution(ax=ax_degree, fontsize=fontsize)
            self.plot_in_vs_out_degree(ax=ax_in_out, fontsize=fontsize)
            self.plot_cellular_location(top_n=10, ax=ax_location, fontsize=fontsize)
            self.plot_catalyst_reuse(ax=ax_catalyst, fontsize=fontsize)
            self.plot_reaction_size_distribution(ax=ax_rxn_size, fontsize=fontsize)
            self.plot_connected_components_over_time(ax=ax_components, fontsize=fontsize)

            # Panel labels a–j
            for ax, label in zip(
                [
                    ax_edge_growth, ax_node_recruit,
                    ax_node_types, ax_edge_types, ax_degree,
                    ax_in_out, ax_location,
                    ax_catalyst, ax_rxn_size,
                    ax_components,
                ],
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ):
                _panel_label(ax, label, fontsize=fontsize)

        return fig
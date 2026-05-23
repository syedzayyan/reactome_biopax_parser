"""
pathway_comparison.py — cross-pathway summary for the FISH paper.

Produces the single "bang bang bang" comparison figure and the matching
LaTeX table for a set of parsed Reactome graphs. Designed for the NeurIPS
space budget: one figure, one table, no per-graph dashboards.

Workflow
--------
    from reactome_graphs import ReactomeBioPAX
    from pathway_comparison import PathwaySet

    parser = ReactomeBioPAX(uniprot_accession_num=True)

    PATHWAYS = {
        "Immune":      "R-HSA-168256",
        "Cell Cycle":  "R-HSA-1640170",
        "Hemostasis":  "R-HSA-109582",
        "Metabolism":  "R-HSA-1430728",
        "Gα(s) signalling": "R-HSA-418555",
    }

    graphs = {
        name: parser.parse_biopax_into_networkx(
            f"./data/biopax3/{rid}.xml",
            reaction_partners=False,
            include_complexes=True,
        )
        for name, rid in PATHWAYS.items()
    }

    ps = PathwaySet(graphs)
    ps.summary_table()                       # pandas DataFrame
    ps.to_latex("tables/pathway_stats.tex")  # LaTeX table for the paper
    fig = ps.summary_figure()                # the single comparison figure
    fig.savefig("figures/pathway_comparison.pdf", dpi=300, bbox_inches="tight")

Assumptions about the graph (verified against nx_graph.py)
----------------------------------------------------------
* Nodes carry a ``type`` attribute (one of: protein, complex, small_molecule,
  DNA, RNA, physical_entity, other) — set in the final annotation pass.
* Edges carry ``type`` (reaction, catalysis, translocation, expression,
  complex_component, left_reactant, right_product) and, when step-scoped,
  ``time`` (int global rank), ``local_order`` (int), ``pathway`` (str).
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── styling ────────────────────────────────────────────────────────────────
_FG = "#2e3440"
_GRID = "#e6e9ef"
_BG = "#ffffff"

# Consistent per-pathway colour — eye tracks one pathway across all panels.
_PATHWAY_PALETTE = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
]

_NODE_TYPE_COLOR = {
    "protein": "#4c78a8",
    "complex": "#b279a2",
    "small_molecule": "#59a14f",
    "DNA": "#edc948",
    "RNA": "#ff9da7",
    "physical_entity": "#f28e2b",
    "other": "#9aa1a6",
}

_EDGE_TYPE_COLOR = {
    "reaction": "#4c78a8",
    "complex_component": "#b279a2",
    "catalysis": "#e15759",
    "translocation": "#76b7b2",
    "expression": "#af7aa1",
    "left_reactant": "#8cd17d",
    "right_product": "#d4a6a6",
    "other": "#9aa1a6",
}

_RC = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
}


def _style_ax(ax, fontsize: int = 8):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=fontsize - 1)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(color=_GRID, linewidth=0.5, linestyle="--")


# ════════════════════════════════════════════════════════════════════════════


class PathwaySet:
    """
    A set of parsed Reactome graphs, with summary statistics and a single
    comparison figure.

    Parameters
    ----------
    graphs : dict[str, nx.DiGraph]
        Mapping of pathway display-name to its parsed FISH graph.
        Insertion order is preserved everywhere.
    """

    def __init__(self, graphs: dict[str, nx.DiGraph]):
        if not graphs:
            raise ValueError("PathwaySet requires at least one graph")
        self.graphs = dict(graphs)
        self.names = list(self.graphs.keys())
        self.colors = {
            n: _PATHWAY_PALETTE[i % len(_PATHWAY_PALETTE)]
            for i, n in enumerate(self.names)
        }
        self._wcc_cache: dict[str, list[set]] = {}

    # ── connectivity helpers ─────────────────────────────────────────────────

    def _wccs(self, name: str) -> list[set]:
        """Weakly-connected components, largest-first, cached."""
        if name not in self._wcc_cache:
            G = self.graphs[name]
            self._wcc_cache[name] = sorted(
                (set(c) for c in nx.weakly_connected_components(G)),
                key=len,
                reverse=True,
            )
        return self._wcc_cache[name]

    @staticmethod
    def _node_type_counts(G: nx.DiGraph) -> Counter:
        return Counter(d.get("type", "other") for _, d in G.nodes(data=True))

    @staticmethod
    def _edge_type_counts(G: nx.DiGraph) -> Counter:
        return Counter(d.get("type", "other") for _, _, d in G.edges(data=True))

    @staticmethod
    def _times(G: nx.DiGraph) -> list[int]:
        """All integer global-rank values present on edges."""
        out = []
        for _, _, d in G.edges(data=True):
            t = d.get("time")
            if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
                out.append(int(t))
        return out

    def _node_first_seen(self, name: str) -> dict:
        """node -> earliest global rank at which an incident edge appears."""
        G = self.graphs[name]
        first: dict = {}
        edges = [
            (u, v, int(d["time"]))
            for u, v, d in G.edges(data=True)
            if isinstance(d.get("time"), (int, np.integer))
            and not isinstance(d.get("time"), bool)
        ]
        for u, v, t in sorted(edges, key=lambda e: e[2]):
            first.setdefault(u, t)
            first.setdefault(v, t)
        return first

    # ── summary table ────────────────────────────────────────────────────────

    def summary_table(self) -> pd.DataFrame:
        """One row per pathway. This is the paper's pathway-statistics table."""
        rows = []
        for name, G in self.graphs.items():
            degrees = [d for _, d in G.degree()]
            wccs = self._wccs(name)
            largest = len(wccs[0]) if wccs else 0
            isolated = sum(1 for c in wccs if len(c) == 1)
            ntypes = self._node_type_counts(G)
            etypes = self._edge_type_counts(G)
            times = self._times(G)

            first_seen = self._node_first_seen(name)
            if first_seen:
                ranks = sorted(first_seen.values())
                max_r = ranks[-1] or 1
                tail_frac = sum(1 for r in ranks if r > 0.8 * max_r) / len(ranks)
            else:
                tail_frac = float("nan")

            n = G.number_of_nodes()
            rows.append(
                {
                    "Nodes": n,
                    "Edges": G.number_of_edges(),
                    "Density": nx.density(G),
                    "Avg degree": float(np.mean(degrees)) if degrees else 0.0,
                    "Max degree": max(degrees) if degrees else 0,
                    "Is DAG": nx.is_directed_acyclic_graph(G),
                    "WCCs": len(wccs),
                    "Largest WCC %": 100.0 * largest / n if n else 0.0,
                    "Isolated": isolated,
                    "Global ranks": len(set(times)),
                    "Recruit tail %": 100.0 * tail_frac,
                    "% Protein": 100.0 * ntypes.get("protein", 0) / n if n else 0.0,
                    "% Complex": 100.0 * ntypes.get("complex", 0) / n if n else 0.0,
                    "% SmallMol": 100.0 * ntypes.get("small_molecule", 0) / n
                    if n
                    else 0.0,
                    "% Reaction edges": (
                        100.0 * etypes.get("reaction", 0) / G.number_of_edges()
                        if G.number_of_edges()
                        else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows, index=self.names)

    def to_latex(self, path: Optional[str] = None, transpose: bool = True) -> str:
        """
        LaTeX table. ``transpose=True`` puts properties as rows and pathways
        as columns, which fits a NeurIPS column far better when you have 5
        pathways and ~15 properties.
        """
        df = self.summary_table()
        if transpose:
            df = df.T
        latex = df.to_latex(
            float_format="%.3g",
            bold_rows=False,
            caption="Graph statistics for the five FISH pathway graphs.",
            label="tab:pathway_stats",
        )
        if path:
            with open(path, "w") as fh:
                fh.write(latex)
        return latex

    # ── the single comparison figure ─────────────────────────────────────────

    def summary_figure(self, figsize=(11, 7), fontsize: int = 8) -> plt.Figure:
        """
        Six-panel cross-pathway comparison figure.

        Layout (3 columns x 2 rows)
        ------
        a  size scatter (nodes vs edges, log-log)
        b  node-type composition (normalised stacked bars)
        c  edge-type composition (normalised stacked bars)
        d  degree distribution overlay (log-log)
        e  node recruitment overlay (normalised rank)
        f  connectivity breakdown (giant vs fragment fractions)
        """
        with plt.rc_context(_RC):
            fig = plt.figure(figsize=figsize)
            fig.patch.set_facecolor(_BG)
            gs = fig.add_gridspec(
                2,
                3,
                hspace=0.62,
                wspace=0.42,
                left=0.07,
                right=0.98,
                top=0.92,
                bottom=0.20,
            )
            ax_a = fig.add_subplot(gs[0, 0])
            ax_b = fig.add_subplot(gs[0, 1])
            ax_c = fig.add_subplot(gs[0, 2])
            ax_d = fig.add_subplot(gs[1, 0])
            ax_e = fig.add_subplot(gs[1, 1])
            ax_f = fig.add_subplot(gs[1, 2])

            self._panel_size_scatter(ax_a, fontsize)
            node_h, node_l = self._panel_composition(ax_b, "node", fontsize)
            edge_h, edge_l = self._panel_composition(ax_c, "edge", fontsize)
            self._panel_degree_overlay(ax_d, fontsize)
            self._panel_recruitment(ax_e, fontsize)
            conn_h, conn_l = self._panel_connectivity(ax_f, fontsize)

            for ax, lbl in zip(
                [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
                ["a", "b", "c", "d", "e", "f"],
            ):
                ax.text(
                    -0.16,
                    1.12,
                    lbl,
                    transform=ax.transAxes,
                    fontsize=fontsize + 3,
                    fontweight="bold",
                    va="top",
                    color=_FG,
                )

            # Three shared figure-level legends along the bottom strip, each
            # built from explicit handles so nothing leaks between panels:
            #   left   — pathway colours (used by panels a, d, e)
            #   middle — node + edge composition types (panels b, c)
            #   right  — connectivity buckets (panel f)
            pathway_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=self.colors[n],
                    markeredgecolor=_BG,
                )
                for n in self.names
            ]
            fig.legend(
                pathway_handles,
                self.names,
                title="Pathway",
                loc="lower left",
                bbox_to_anchor=(0.07, 0.01),
                ncol=2,
                fontsize=fontsize - 2,
                title_fontsize=fontsize - 1,
                framealpha=0.0,
            )
            # De-duplicate type labels across node + edge composition.
            seen, comp_h, comp_l = set(), [], []
            for h, l in zip(node_h + edge_h, node_l + edge_l):
                if l not in seen:
                    seen.add(l)
                    comp_h.append(h)
                    comp_l.append(l)
            fig.legend(
                comp_h,
                comp_l,
                title="Node / edge type",
                loc="lower center",
                bbox_to_anchor=(0.52, 0.01),
                ncol=3,
                fontsize=fontsize - 2,
                title_fontsize=fontsize - 1,
                framealpha=0.0,
            )
            fig.legend(
                conn_h,
                conn_l,
                title="Component size",
                loc="lower right",
                bbox_to_anchor=(0.98, 0.01),
                ncol=2,
                fontsize=fontsize - 2,
                title_fontsize=fontsize - 1,
                framealpha=0.0,
            )
        return fig

    # ── individual panels ────────────────────────────────────────────────────

    def _panel_size_scatter(self, ax, fontsize):
        for name, G in self.graphs.items():
            ax.scatter(
                G.number_of_nodes(),
                G.number_of_edges(),
                s=70,
                color=self.colors[name],
                edgecolor=_BG,
                linewidth=0.8,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Nodes (log)", fontsize=fontsize)
        ax.set_ylabel("Edges (log)", fontsize=fontsize)
        ax.set_title("Graph Size", fontsize=fontsize + 1)
        _style_ax(ax, fontsize)

    def _panel_composition(self, ax, kind, fontsize):
        """
        Draws a normalised stacked bar per pathway. Returns (handles, labels)
        for the caller to render as a single figure-level legend — drawing
        per-axes legends here causes outside-anchored legends to leak into
        neighbouring panels.
        """
        if kind == "node":
            counter_fn = self._node_type_counts
            color_map = _NODE_TYPE_COLOR
            order = [
                "protein",
                "complex",
                "small_molecule",
                "DNA",
                "RNA",
                "physical_entity",
                "other",
            ]
            title = "Node Composition"
        else:
            counter_fn = self._edge_type_counts
            color_map = _EDGE_TYPE_COLOR
            order = [
                "reaction",
                "complex_component",
                "catalysis",
                "translocation",
                "expression",
                "left_reactant",
                "right_product",
                "other",
            ]
            title = "Edge Composition"

        per = {n: counter_fn(G) for n, G in self.graphs.items()}
        present = [t for t in order if any(per[n].get(t, 0) for n in self.names)]

        handles, labels = [], []
        bottoms = np.zeros(len(self.names))
        for t in present:
            vals = np.array([per[n].get(t, 0) for n in self.names], dtype=float)
            totals = np.array([sum(per[n].values()) for n in self.names], dtype=float)
            totals[totals == 0] = 1.0
            frac = vals / totals
            bars = ax.barh(
                self.names,
                frac,
                left=bottoms,
                color=color_map.get(t, "#9aa1a6"),
                edgecolor=_BG,
                linewidth=0.5,
                height=0.62,
            )
            handles.append(bars[0])
            labels.append(t)
            bottoms += frac
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Fraction", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize + 1)
        _style_ax(ax, fontsize)
        return handles, labels

    def _panel_degree_overlay(self, ax, fontsize):
        """
        Complementary CDF of degree, one curve per pathway. The CCDF
        P(degree >= k) is the standard way to show heavy-tailed degree
        distributions — five smooth monotone curves overlay cleanly,
        unlike raw degree-count scatter which is jagged in the tail.
        """
        for name, G in self.graphs.items():
            degrees = np.array([d for _, d in G.degree() if d > 0])
            if degrees.size == 0:
                continue
            xs = np.sort(degrees)
            # P(D >= x) for each observed degree.
            ccdf = 1.0 - np.arange(len(xs)) / len(xs)
            ax.plot(
                xs,
                ccdf,
                color=self.colors[name],
                alpha=0.9,
                linewidth=1.3,
                label=name,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (log)", fontsize=fontsize)
        ax.set_ylabel("P(degree \u2265 k)", fontsize=fontsize)
        ax.set_title("Degree Distribution (CCDF)", fontsize=fontsize + 1)
        _style_ax(ax, fontsize)

    def _panel_recruitment(self, ax, fontsize):
        """
        Cumulative node recruitment vs normalised global rank. No legend is
        drawn here — pathway colours are carried by the shared figure legend.
        The dashed diagonal marks perfectly uniform recruitment.
        """
        for name in self.names:
            first_seen = self._node_first_seen(name)
            if not first_seen:
                continue
            ranks = sorted(first_seen.values())
            frac = np.arange(1, len(ranks) + 1) / len(ranks)
            max_r = ranks[-1] or 1
            xs = np.array(ranks) / max_r
            ax.plot(xs, frac, color=self.colors[name], linewidth=1.4, alpha=0.9)
        # Diagonal = perfectly uniform recruitment, for reference.
        ax.plot(
            [0, 1], [0, 1], color="#9aa1a6", linewidth=0.9, linestyle="--", zorder=0
        )
        ax.text(
            0.62,
            0.50,
            "uniform",
            fontsize=fontsize - 2,
            color="#9aa1a6",
            rotation=38,
            va="center",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Normalised global rank", fontsize=fontsize)
        ax.set_ylabel("Fraction of nodes seen", fontsize=fontsize)
        ax.set_title("Node Recruitment", fontsize=fontsize + 1)
        _style_ax(ax, fontsize)

    def _panel_connectivity(self, ax, fontsize):
        buckets = ["Giant", ">20", "6-20", "2-5", "Isolated"]
        bucket_color = {
            "Giant": "#4c78a8",
            ">20": "#54a24b",
            "6-20": "#8cd17d",
            "2-5": "#f28e2b",
            "Isolated": "#9aa1a6",
        }
        data = {b: [] for b in buckets}
        for name in self.names:
            wccs = self._wccs(name)
            total = sum(len(c) for c in wccs) or 1
            giant = len(wccs[0]) if wccs else 0
            data["Giant"].append(giant / total)
            data[">20"].append(sum(len(c) for c in wccs[1:] if len(c) > 20) / total)
            data["6-20"].append(sum(len(c) for c in wccs if 6 <= len(c) <= 20) / total)
            data["2-5"].append(sum(len(c) for c in wccs if 2 <= len(c) <= 5) / total)
            data["Isolated"].append(sum(1 for c in wccs if len(c) == 1) / total)

        bottoms = np.zeros(len(self.names))
        handles, labels = [], []
        for b in buckets:
            vals = np.array(data[b])
            bars = ax.barh(
                self.names,
                vals,
                left=bottoms,
                color=bucket_color[b],
                edgecolor=_BG,
                linewidth=0.5,
                height=0.62,
            )
            handles.append(bars[0])
            labels.append(b)
            bottoms += vals
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Fraction of nodes", fontsize=fontsize)
        ax.set_title("Connectivity", fontsize=fontsize + 1)
        _style_ax(ax, fontsize)
        return handles, labels


# ── component inspection (table, not figure) ────────────────────────────────


def component_inventory(
    G: nx.DiGraph, max_components: int = 20, min_size: int = 2
) -> pd.DataFrame:
    """
    Inventory of non-giant weakly-connected components in one graph.

    For each small component reports size, dominant node type, the
    highest-degree representative node, and the earliest-ranked edge
    (rank, type, and a u -> v label). Answers "what are the disconnected
    pieces, and when does the first reaction in each fire?"
    """
    wccs = sorted(
        (set(c) for c in nx.weakly_connected_components(G)),
        key=len,
        reverse=True,
    )
    small = [c for c in wccs[1:] if len(c) >= min_size][:max_components]

    rows = []
    for idx, comp in enumerate(small, start=1):
        subg = G.subgraph(comp)
        type_counts = Counter(G.nodes[n].get("type", "other") for n in comp)
        rep = max(subg.degree, key=lambda x: x[1])[0]
        ranked = [
            (u, v, d)
            for u, v, d in subg.edges(data=True)
            if isinstance(d.get("time"), (int, np.integer))
            and not isinstance(d.get("time"), bool)
        ]
        if ranked:
            u, v, d = min(ranked, key=lambda e: e[2]["time"])
            first_rank = int(d["time"])
            first_type = d.get("type", "other")
            first_edge = f"{str(u)[:24]} -> {str(v)[:24]}"
        else:
            first_rank, first_type, first_edge = None, None, "(no ranked edges)"

        rows.append(
            {
                "component": idx,
                "size": len(comp),
                "dominant_type": type_counts.most_common(1)[0][0],
                "representative": str(rep)[:48],
                "first_rank": first_rank,
                "first_edge_type": first_type,
                "first_edge": first_edge,
            }
        )
    return pd.DataFrame(rows)

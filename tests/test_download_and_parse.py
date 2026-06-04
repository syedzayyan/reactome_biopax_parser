"""
tests/test_download_and_parse.py
---------------------------------
Integration tests for the Reactome download utilities and BioPAX parser.

Test tiers
----------
Fast (no network, run every time):
  TestParser         — parses a small BioPAX file already on disk.

Network / slow (skipped unless --run-network is passed):
  TestDownload       — downloads a tiny pathway file from Reactome.
  TestDownloadParse  — downloads the Immune pathway (R-HSA-168256, ~42 MB) and
                       parses it end-to-end; matches the benchmark described in
                       baselines/GNNs/README.md.

Run the fast suite:
    pytest tests/test_download_and_parse.py -v

Run everything including network tests:
    pytest tests/test_download_and_parse.py -v --run-network
"""

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from reactome_graphs import (
    ReactomeBioPAX,
    download_single_biopax_file_by_pathway_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fixture file committed to tests/fixtures/ so the fast tests run in CI
# without any network access. (54 KB, 10 nodes, 14 edges, 4 edge types.)
SMALL_PATHWAY_ID = "R-HSA-139910"
SMALL_PATHWAY_FILE = Path(__file__).parent / "fixtures" / f"{SMALL_PATHWAY_ID}.xml"

# The Immune System pathway used by the GNN benchmarks (~42 MB).
# Not committed — downloaded on demand by the network test.
IMMUNE_PATHWAY_ID = "R-HSA-168256"
IMMUNE_PATHWAY_FILE = Path(__file__).parent.parent / "data" / "biopax3" / f"{IMMUNE_PATHWAY_ID}.xml"

KNOWN_EDGE_TYPES = {
    "reaction",
    "catalysis",
    "complex_component",
    "translocation",
    "expression",
    "left_reactant",
    "right_product",
}


def _require_network(request):
    """Skip the test if --run-network was not passed on the command line."""
    if not request.config.getoption("--run-network", default=False):
        pytest.skip("Network tests disabled — pass --run-network to enable.")


# ---------------------------------------------------------------------------
# Fast parser tests (no network)
# ---------------------------------------------------------------------------


class TestParser:
    """Parse a small BioPAX file that is already on disk."""

    def test_file_exists(self):
        assert SMALL_PATHWAY_FILE.exists(), (
            f"Expected fixture file {SMALL_PATHWAY_FILE} to exist. "
            "Run the project data pipeline to populate data/biopax3/."
        )

    def test_parse_returns_digraph(self):
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        assert isinstance(G, nx.DiGraph)

    def test_graph_has_nodes_and_edges(self):
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        assert G.number_of_nodes() > 0, "Graph has no nodes"
        assert G.number_of_edges() > 0, "Graph has no edges"

    def test_node_type_attribute_present(self):
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        for n, data in G.nodes(data=True):
            assert "type" in data, f"Node {n!r} missing 'type' attribute"

    def test_edge_type_attribute_present(self):
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        for u, v, data in G.edges(data=True):
            assert "type" in data, f"Edge ({u!r}, {v!r}) missing 'type' attribute"

    def test_edge_time_attribute_present(self):
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        for u, v, data in G.edges(data=True):
            assert "time" in data, f"Edge ({u!r}, {v!r}) missing 'time' attribute"

    def test_edge_types_are_known(self):
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        observed = {data["type"] for _, _, data in G.edges(data=True)}
        unknown = observed - KNOWN_EDGE_TYPES
        assert not unknown, f"Unexpected edge types: {unknown}"

    def test_small_pathway_counts(self):
        """Regression: R-HSA-139910 should produce exactly 10 nodes, 14 edges."""
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        assert G.number_of_nodes() == 10, (
            f"Expected 10 nodes, got {G.number_of_nodes()}"
        )
        assert G.number_of_edges() == 14, (
            f"Expected 14 edges, got {G.number_of_edges()}"
        )

    def test_small_pathway_has_multiple_edge_types(self):
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        etypes = {data["type"] for _, _, data in G.edges(data=True)}
        assert len(etypes) >= 2, (
            f"Expected at least 2 edge types, got {sorted(etypes)}"
        )

    def test_include_complexes_false(self):
        """include_complexes=False should still return a valid graph."""
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(
            str(SMALL_PATHWAY_FILE), include_complexes=False
        )
        assert isinstance(G, nx.DiGraph)
        for _, _, data in G.edges(data=True):
            assert "type" in data

    def test_reaction_partners_flag(self):
        """reaction_partners=True should not raise and should produce a graph."""
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(
            str(SMALL_PATHWAY_FILE), reaction_partners=True
        )
        assert isinstance(G, nx.DiGraph)


# ---------------------------------------------------------------------------
# Download tests  (require --run-network)
# ---------------------------------------------------------------------------


class TestDownload:
    """Download a small pathway and verify the file was created."""

    def test_download_single_file(self, request, tmp_path):
        _require_network(request)
        # R-HSA-168888 is a tiny 3-reaction pathway — fast to download.
        tiny_id = "R-HSA-168888"
        download_single_biopax_file_by_pathway_id(tiny_id, save_dir=str(tmp_path) + "/")
        out = tmp_path / f"{tiny_id}.xml"
        assert out.exists(), f"Expected downloaded file at {out}"
        assert out.stat().st_size > 0, "Downloaded file is empty"

    def test_downloaded_file_is_valid_xml(self, request, tmp_path):
        _require_network(request)
        import xml.etree.ElementTree as ET
        tiny_id = "R-HSA-168888"
        download_single_biopax_file_by_pathway_id(tiny_id, save_dir=str(tmp_path) + "/")
        out = tmp_path / f"{tiny_id}.xml"
        # Should parse without raising.
        ET.parse(str(out))

    def test_download_and_parse_tiny_pathway(self, request, tmp_path):
        """Download R-HSA-168888, parse it, check the resulting graph."""
        _require_network(request)
        tiny_id = "R-HSA-168888"
        download_single_biopax_file_by_pathway_id(tiny_id, save_dir=str(tmp_path) + "/")
        out = tmp_path / f"{tiny_id}.xml"
        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(out))
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
        for _, _, data in G.edges(data=True):
            assert "type" in data
            assert "time" in data


# ---------------------------------------------------------------------------
# End-to-end: download + parse the Immune pathway  (network, slow)
# ---------------------------------------------------------------------------


class TestImmunePathway:
    """
    End-to-end test for the Immune System pathway (R-HSA-168256).

    This is the benchmark pathway used by baselines/GNNs.  The file is ~42 MB
    and takes ~5 minutes to parse, so this test is gated behind --run-network.

    If the file already exists in data/biopax3/ the download step is skipped.
    """

    def test_immune_download_and_parse(self, request):
        _require_network(request)

        # Download only if not already present (allows re-use of cached file).
        if not IMMUNE_PATHWAY_FILE.exists():
            IMMUNE_PATHWAY_FILE.parent.mkdir(parents=True, exist_ok=True)
            download_single_biopax_file_by_pathway_id(
                IMMUNE_PATHWAY_ID, save_dir=str(IMMUNE_PATHWAY_FILE.parent) + "/"
            )

        assert IMMUNE_PATHWAY_FILE.exists()
        assert IMMUNE_PATHWAY_FILE.stat().st_size > 1_000_000, (
            "Immune pathway file seems too small — may be corrupt or truncated."
        )

        p = ReactomeBioPAX(uniprot_accession_num=False)
        G = p.parse_biopax_into_networkx(str(IMMUNE_PATHWAY_FILE))

        assert isinstance(G, nx.DiGraph)
        # Sanity bounds: the Immune pathway should have thousands of nodes/edges.
        assert G.number_of_nodes() >= 1_000, (
            f"Expected >= 1000 nodes, got {G.number_of_nodes()}"
        )
        assert G.number_of_edges() >= 5_000, (
            f"Expected >= 5000 edges, got {G.number_of_edges()}"
        )

        # All edges must carry type and time.
        for u, v, data in G.edges(data=True):
            assert "type" in data, f"Edge ({u!r}, {v!r}) missing 'type'"
            assert "time" in data, f"Edge ({u!r}, {v!r}) missing 'time'"

        # Must have at least reaction and catalysis edge types.
        observed_types = {data["type"] for _, _, data in G.edges(data=True)}
        assert "reaction" in observed_types
        assert "catalysis" in observed_types

        # All nodes must have a type attribute.
        for n, data in G.nodes(data=True):
            assert "type" in data, f"Node {n!r} missing 'type'"

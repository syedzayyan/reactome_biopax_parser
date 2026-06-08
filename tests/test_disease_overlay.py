"""
tests/test_disease_overlay.py
-----------------------------
Tests for DiseaseOverlay.

Test tiers
----------
Fast (no network, no large data files — run every time):
  TestDataclasses        — MutantVariant / DiseaseReaction field behaviour.
  TestHelpers            — Static helpers: _make_disease_node_label, _max_time.
  TestHealthyIndex       — _build_healthy_index uses the committed 54 KB fixture.
  TestVariantRegex       — _VARIANT_NAME_RE matches / rejects expected strings.

Data-dependent (skipped unless both data files are present):
  TestDiseaseOverlayFull — Full overlay of R-HSA-5260271.xml (Diseases of
                           Immune System, 856 KB) on R-HSA-168898.xml
                           (Toll-like Receptor Cascades, ~900 KB).
                           Verifies variant extraction, branch-node anchoring,
                           edge topology, time ordering, and ClinGen IDs.
"""

from pathlib import Path

import networkx as nx
import pytest

from reactome_graphs import ReactomeBioPAX
from reactome_graphs.parser.disease_parsing import (
    DiseaseOverlay,
    DiseaseReaction,
    MutantVariant,
    OverlayResult,
    _VARIANT_NAME_RE,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DATA_DIR = Path(__file__).parent.parent / "data" / "biopax3"

SMALL_PATHWAY_FILE = FIXTURE_DIR / "R-HSA-139910.xml"   # committed, 54 KB
DISEASE_FILE = DATA_DIR / "R-HSA-5260271.xml"           # Diseases of Immune System
HEALTHY_TLR_FILE = DATA_DIR / "R-HSA-168898.xml"        # Toll-like Receptor Cascades

_DATA_AVAILABLE = DISEASE_FILE.exists() and HEALTHY_TLR_FILE.exists()

# ---------------------------------------------------------------------------
# Module-level fixture shared across all integration test methods
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tlr_overlay():
    """Run the overlay once per test session; reuse across all methods."""
    if not _DATA_AVAILABLE:
        pytest.skip("data files not present")
    hp = ReactomeBioPAX(uniprot_accession_num=False)
    G_healthy = hp.parse_biopax_into_networkx(str(HEALTHY_TLR_FILE))
    overlay = DiseaseOverlay(ReactomeBioPAX, hp)
    result = overlay.apply(G_healthy, str(DISEASE_FILE))
    return overlay, result


# ---------------------------------------------------------------------------
# Fast: dataclass behaviour
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_mutant_variant_defaults(self):
        mv = MutantVariant(
            entity_id="Protein2",
            display_name="IRAK4 E402*",
            reactome_db_id="5602517",
            uniprot_id="Q9NWZ3",
            omim_ids=["615129"],
            mutations=["E402TER"],
        )
        assert mv.entity_id == "Protein2"
        assert mv.clingen_ids == []

    def test_mutant_variant_clingen(self):
        mv = MutantVariant(
            entity_id="P1",
            display_name="NFKBIA Q9*",
            reactome_db_id=None,
            uniprot_id=None,
            omim_ids=[],
            mutations=[],
            clingen_ids=["CA123693"],
        )
        assert mv.clingen_ids == ["CA123693"]

    def test_disease_reaction_lof_property(self):
        dr = DiseaseReaction(
            reaction_id="R1",
            display_name="Test LoF",
            left_ids=["P1", "C1"],
            right_ids=[],
            comments=[],
            omim_ids=[],
            disease_label="Test disease",
        )
        assert dr.is_loss_of_function

    def test_disease_reaction_gof_property(self):
        dr = DiseaseReaction(
            reaction_id="R2",
            display_name="Test GoF",
            left_ids=["P1"],
            right_ids=["C2"],
            comments=[],
            omim_ids=["102700"],
            disease_label="EDA-ID",
        )
        assert not dr.is_loss_of_function

    def test_disease_reaction_variant_left_ids_default(self):
        dr = DiseaseReaction(
            reaction_id="R1",
            display_name="x",
            left_ids=[],
            right_ids=[],
            comments=[],
            omim_ids=[],
            disease_label="x",
        )
        assert dr.variant_left_ids == []
        assert dr.branch_nodes == []
        assert dr.disease_node == ""


# ---------------------------------------------------------------------------
# Fast: static helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_disease_label_with_omim(self):
        dr = DiseaseReaction(
            reaction_id="r",
            display_name="x",
            left_ids=[],
            right_ids=[],
            comments=[],
            omim_ids=["615129"],
            disease_label="IRAK4 deficiency",
        )
        lbl = DiseaseOverlay._make_disease_node_label(dr)
        assert lbl == "[DISEASE] IRAK4 deficiency (OMIM:615129)"

    def test_disease_label_no_omim(self):
        dr = DiseaseReaction(
            reaction_id="r",
            display_name="x",
            left_ids=[],
            right_ids=[],
            comments=[],
            omim_ids=[],
            disease_label="TRAF3 deficiency - HSE",
        )
        lbl = DiseaseOverlay._make_disease_node_label(dr)
        assert lbl == "[DISEASE] TRAF3 deficiency - HSE"
        assert "OMIM" not in lbl

    def test_max_time_empty_graph(self):
        G = nx.DiGraph()
        G.add_edge("A", "B")
        assert DiseaseOverlay._max_time(G) == 0

    def test_max_time_with_timed_edges(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", time=7)
        G.add_edge("B", "C", time=3)
        G.add_edge("C", "D", time=21)
        assert DiseaseOverlay._max_time(G) == 21

    def test_max_time_mixed(self):
        G = nx.DiGraph()
        G.add_edge("A", "B")           # no time
        G.add_edge("B", "C", time=5)
        assert DiseaseOverlay._max_time(G) == 5


# ---------------------------------------------------------------------------
# Fast: healthy-graph index
# ---------------------------------------------------------------------------


class TestHealthyIndex:
    """_build_healthy_index must index every node by name and reactome_db_id."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        hp = ReactomeBioPAX(uniprot_accession_num=False)
        self.G = hp.parse_biopax_into_networkx(str(SMALL_PATHWAY_FILE))
        self.overlay = DiseaseOverlay(ReactomeBioPAX, hp)
        self.overlay._build_healthy_index(self.G)

    def test_every_node_in_name_index(self):
        # The index keys by the entity's 'name' attribute (without location suffix),
        # OR falls back to the full node label when no name attribute is present.
        for node, data in self.G.nodes(data=True):
            name = data.get("name")
            if name:
                key = name.strip().lower()
            else:
                key = str(node).strip().lower()
            assert key in self.overlay._healthy_name_to_label, (
                f"Node {node!r} (name={name!r}) missing from healthy name index"
            )

    def test_reactome_db_id_indexed(self):
        for node, data in self.G.nodes(data=True):
            db_id = data.get("reactome_db_id")
            if db_id:
                assert str(db_id) in self.overlay._healthy_id_to_label

    def test_aliases_indexed(self):
        for node, data in self.G.nodes(data=True):
            for alias in data.get("aliases", []) or []:
                if alias:
                    key = alias.strip().lower()
                    assert key in self.overlay._healthy_name_to_label, (
                        f"Alias {alias!r} of node {node!r} missing from index"
                    )

    def test_name_index_values_are_graph_nodes(self):
        for labels in self.overlay._healthy_name_to_label.values():
            for lbl in labels:
                assert lbl in self.G.nodes, (
                    f"Indexed label {lbl!r} is not a node in the healthy graph"
                )


# ---------------------------------------------------------------------------
# Fast: variant-name regex
# ---------------------------------------------------------------------------


class TestVariantRegex:
    """_VARIANT_NAME_RE must match HGVS protein-change notation."""

    @pytest.mark.parametrize("name", [
        "IRAK4 E402*",
        "IRAK4 Q293*",
        "IRAK4 R12C",
        "MyD88 R196C",
        "MyD88 E52del",
        "MyD88 L93P",
        "MyD88 S34Y",
        "MyD88 E65del",
        "TLR3 E746*",
        "TLR3 P554S",
        "UNC93B1 L230Afs*188",
        "TICAM1 R141*",
        "TICAM1 S186L",
        "TRAF3 R118W",
        "NEMO L153R",
        "NEMO E391*",
        "IKBKB Q432Pfs*62",
        "NFKBIA Q9*",
        "IkBA S32I",
    ])
    def test_matches_variant(self, name):
        assert _VARIANT_NAME_RE.search(name), (
            f"Expected variant regex to match {name!r}"
        )

    @pytest.mark.parametrize("name", [
        "p-4Y-MAL",
        "ATP",
        "MyD88",
        "TLR4",
        "MYD88",
        "IRAK4 variants",     # OR-group label — matched by keyword, not regex
        "NFkB Complex",
        "viral double-stranded RNA",
        "Flagellin",
    ])
    def test_does_not_match_normal(self, name):
        assert not _VARIANT_NAME_RE.search(name), (
            f"Expected variant regex NOT to match {name!r}"
        )


# ---------------------------------------------------------------------------
# Integration: full overlay on R-HSA-5260271 / R-HSA-168898
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="data/biopax3 files not present")
class TestDiseaseOverlayFull:

    # ── Anchoring ──────────────────────────────────────────────────────────

    def test_all_reactions_anchored(self, tlr_overlay):
        _, result = tlr_overlay
        assert result.unanchored == [], (
            "Unexpected unanchored reactions: "
            + str([dr.display_name for dr in result.unanchored])
        )

    def test_anchored_count(self, tlr_overlay):
        _, result = tlr_overlay
        assert len(result.anchored) == 14

    # ── Disease nodes ──────────────────────────────────────────────────────

    def test_disease_nodes_added(self, tlr_overlay):
        _, result = tlr_overlay
        assert len(result.disease_nodes) >= 10

    def test_disease_node_type(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        for dn in result.disease_nodes:
            assert dn in G.nodes
            assert G.nodes[dn]["type"] == "disease_phenotype"

    def test_disease_node_labels_prefixed(self, tlr_overlay):
        _, result = tlr_overlay
        for dn in result.disease_nodes:
            assert dn.startswith("[DISEASE]"), f"Unexpected label: {dn!r}"

    # ── Mutant variant nodes ───────────────────────────────────────────────

    def test_mut_nodes_added(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        mut_nodes = [n for n in G.nodes if str(n).startswith("[MUT]")]
        assert len(mut_nodes) > 0, "No [MUT] nodes in graph"

    def test_mut_nodes_have_type(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        for n in G.nodes:
            if str(n).startswith("[MUT]"):
                assert G.nodes[n]["type"] == "mutant_variant"

    # ── Variant extraction ─────────────────────────────────────────────────

    def test_irak4_variants_count(self, tlr_overlay):
        """IRAK4 deficiency should yield exactly 3 variants (E402*, Q293*, R12C)."""
        _, result = tlr_overlay
        irak4_rxns = [dr for dr in result.anchored if "IRAK4" in dr.disease_label]
        assert irak4_rxns, "No IRAK4 deficiency reaction found"
        for dr in irak4_rxns:
            assert len(dr.mutant_variants) == 3, (
                f"Expected 3 IRAK4 variants, got {len(dr.mutant_variants)}: "
                f"{[mv.display_name for mv in dr.mutant_variants]}"
            )

    def test_irak4_variant_names(self, tlr_overlay):
        _, result = tlr_overlay
        irak4_rxn = next(
            dr for dr in result.anchored if "IRAK4 deficiency (TLR2/4)" in dr.disease_label
        )
        names = {mv.display_name for mv in irak4_rxn.mutant_variants}
        assert "IRAK4 E402*" in names
        assert "IRAK4 Q293*" in names
        assert "IRAK4 R12C" in names

    def test_nemo_variants_count(self, tlr_overlay):
        """IKBKG/NEMO deficiency should extract >= 5 variants."""
        _, result = tlr_overlay
        nemo_rxns = [
            dr for dr in result.anchored
            if "IKBKG" in dr.disease_label or "NEMO" in dr.disease_label
            or "EDA-ID" in dr.disease_label
        ]
        assert nemo_rxns
        most_variants = max(len(dr.mutant_variants) for dr in nemo_rxns)
        assert most_variants >= 5, (
            f"Expected >= 5 NEMO variants, got {most_variants}"
        )

    def test_myd88_variants_count(self, tlr_overlay):
        """MyD88 deficiency OR-group should expand to >= 4 variants."""
        _, result = tlr_overlay
        myd88_rxns = [dr for dr in result.anchored if "MyD88" in dr.disease_label]
        assert myd88_rxns
        # Some MyD88 reactions use direct variants (R196C), some the OR-group
        total_variants = sum(len(dr.mutant_variants) for dr in myd88_rxns)
        assert total_variants >= 4

    def test_variant_left_ids_set_for_lof(self, tlr_overlay):
        """LoF reactions with an OR-group on the left must have variant_left_ids."""
        _, result = tlr_overlay
        lof_with_left_variants = [
            dr for dr in result.anchored
            if dr.is_loss_of_function and dr.mutant_variants and dr.variant_left_ids
        ]
        assert len(lof_with_left_variants) >= 8, (
            f"Expected >= 8 LoF reactions with variant_left_ids, "
            f"got {len(lof_with_left_variants)}"
        )

    def test_clingen_ids_extracted(self, tlr_overlay):
        """NFKBIA variants carry ClinGen allele IDs in R-HSA-5260271."""
        _, result = tlr_overlay
        all_mvs = [mv for dr in result.anchored for mv in dr.mutant_variants]
        clingen_mvs = [mv for mv in all_mvs if mv.clingen_ids]
        assert len(clingen_mvs) > 0, (
            "No ClinGen IDs extracted — expected NFKBIA variants to have them"
        )
        # Spot-check a known ID
        all_cg = {cg for mv in clingen_mvs for cg in mv.clingen_ids}
        assert "CA123693" in all_cg, f"Expected CA123693, got: {all_cg}"

    # ── Edge topology ──────────────────────────────────────────────────────

    def test_lof_edges_present(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        lof = [(u, v) for u, v, d in G.edges(data=True)
               if d.get("type") == "loss_of_function"]
        assert len(lof) > 0

    def test_disease_branch_edges_present(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        branches = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("type") == "disease_branch"]
        assert len(branches) > 0

    def test_disease_branch_src_is_healthy_node(self, tlr_overlay):
        """disease_branch edges must originate from healthy graph nodes, not [MUT] nodes."""
        _, result = tlr_overlay
        G = result.graph
        for u, v, d in G.edges(data=True):
            if d.get("type") == "disease_branch":
                assert not str(u).startswith("[MUT]"), (
                    f"disease_branch edge src {u!r} is a [MUT] node — should be healthy"
                )
                assert str(v).startswith("[MUT]"), (
                    f"disease_branch edge dst {v!r} should be a [MUT] node"
                )

    def test_lof_dst_is_disease_node(self, tlr_overlay):
        """loss_of_function edges must end at a [DISEASE] node."""
        _, result = tlr_overlay
        G = result.graph
        for u, v, d in G.edges(data=True):
            if d.get("type") == "loss_of_function":
                assert str(v).startswith("[DISEASE]"), (
                    f"loss_of_function edge dst {v!r} is not a [DISEASE] node"
                )

    def test_gof_aberrant_edges_present(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        gof = [(u, v) for u, v, d in G.edges(data=True)
               if d.get("subtype") == "disease_aberrant"]
        assert len(gof) > 0

    def test_all_disease_edges_have_time(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        disease_edge_types = {
            "disease_branch", "loss_of_function",
            "disease_progression", "aberrant_product",
        }
        for u, v, d in G.edges(data=True):
            if d.get("type") in disease_edge_types:
                assert "time" in d, (
                    f"Disease edge ({u!r}→{v!r}, type={d['type']!r}) missing 'time'"
                )

    def test_disease_times_after_healthy_minimum(self, tlr_overlay):
        """All disease-branch times must exceed the smallest healthy-graph time."""
        _, result = tlr_overlay
        G = result.graph
        healthy_types = {"reaction", "catalysis", "translocation",
                         "expression", "complex_component"}
        healthy_times = [
            d["time"] for _, _, d in G.edges(data=True)
            if d.get("type") in healthy_types and "time" in d
        ]
        disease_times = [
            d["time"] for _, _, d in G.edges(data=True)
            if d.get("type") == "disease_branch" and "time" in d
        ]
        if healthy_times and disease_times:
            assert min(disease_times) > min(healthy_times), (
                "Disease branch times should be after the first healthy reaction"
            )

    # ── Report ─────────────────────────────────────────────────────────────

    def test_report_contains_anchored_count(self, tlr_overlay):
        overlay, result = tlr_overlay
        report = overlay.report(result)
        assert "Anchored   : 14" in report

    def test_report_no_unanchored_section(self, tlr_overlay):
        overlay, result = tlr_overlay
        report = overlay.report(result)
        assert "Unanchored:" not in report

    def test_report_lists_disease_labels(self, tlr_overlay):
        overlay, result = tlr_overlay
        report = overlay.report(result)
        assert "IRAK4 deficiency" in report
        assert "TRAF3 deficiency" in report

    # ── Graph integrity ────────────────────────────────────────────────────

    def test_no_self_loops_in_disease_edges(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        disease_types = {
            "disease_branch", "loss_of_function",
            "disease_progression", "aberrant_product",
        }
        for u, v, d in G.edges(data=True):
            if d.get("type") in disease_types:
                assert u != v, f"Self-loop on disease edge at node {u!r}"

    def test_disease_nodes_not_isolated(self, tlr_overlay):
        _, result = tlr_overlay
        G = result.graph
        for dn in result.disease_nodes:
            assert G.in_degree(dn) > 0, (
                f"Disease node {dn!r} has no incoming edges"
            )

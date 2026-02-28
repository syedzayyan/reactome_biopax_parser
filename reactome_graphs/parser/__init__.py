"""
reactome_biopax/
    __init__.py         — public API
    _parser_base.py     — BioPAX parsing + shared helpers
    _nx_mixin.py        — NetworkX graph builder
    _hypergraph_mixin.py — HyperNetX hypergraph builder

Usage
-----
    from reactome_biopax import ReactomeBioPAX

    r = ReactomeBioPAX(uniprot_accession_num=True)
    G = r.parse_biopax_into_networkx("pathway.owl")
    H = r.parse_biopax_into_hypergraph("pathway.owl")
"""

from .hyper_graph import _HypergraphMixin
from .nx_graph import _NxGraphMixin
from .xml_parse import _ParserBase


class ReactomeBioPAX(_NxGraphMixin, _HypergraphMixin, _ParserBase):
    """
    Parse Reactome BioPAX Level 3 files into NetworkX or HyperNetX graphs.

    Parameters
    ----------
    uniprot_accession_num : bool
        If True, resolve UniProt accession numbers for proteins.
    logger : logging.Logger, optional
        Logger instance for debug/info output.
    """

    # All methods inherited from mixins and base.
    # MRO: ReactomeBioPAX -> _NxGraphMixin -> _HypergraphMixin -> _ParserBase


__all__ = ["ReactomeBioPAX"]

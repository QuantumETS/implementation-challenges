"""Fixed graph instances for the toy MIS challenge."""

from __future__ import annotations

import networkx as nx


def make_graph_5() -> nx.Graph:
    """Return the deterministic 5-node MIS instance."""
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
        ]
    )
    return graph


def make_graph_8() -> nx.Graph:
    """Return the deterministic 8-node MIS instance."""
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 5),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]
    )
    return graph


def get_problem_graphs() -> dict[str, nx.Graph]:
    """Return all named graph instances used in the reference solution."""
    return {
        "graph_5": make_graph_5(),
        "graph_8": make_graph_8(),
    }

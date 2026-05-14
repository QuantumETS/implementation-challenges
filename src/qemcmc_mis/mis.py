"""Maximum Independent Set utilities and exact brute-force verification."""

from __future__ import annotations

from itertools import product
from typing import TypeAlias

import networkx as nx
import pandas as pd

Bitstring: TypeAlias = tuple[int, ...]


def all_bitstrings(n: int) -> list[Bitstring]:
    """Return all n-bit bitstrings as tuples of 0/1."""
    return [tuple(bits) for bits in product((0, 1), repeat=n)]


def bitstring_to_str(x: Bitstring) -> str:
    """Convert a tuple bitstring to a compact string."""
    return "".join(str(bit) for bit in x)


def str_to_bitstring(s: str) -> Bitstring:
    """Convert a compact string to a tuple bitstring."""
    return tuple(int(char) for char in s)


def selected_vertices(x: Bitstring) -> set[int]:
    """Return selected vertices where x_i = 1."""
    return {i for i, bit in enumerate(x) if bit == 1}


def independent_set_size(x: Bitstring) -> int:
    """Return sum_i x_i."""
    return sum(x)


def is_valid_independent_set(graph: nx.Graph, x: Bitstring) -> bool:
    """Return True if no edge has both endpoints selected."""
    return violation_count(graph, x) == 0


def violation_count(graph: nx.Graph, x: Bitstring) -> int:
    """Return the number of edges with both endpoints selected."""
    return sum(1 for i, j in graph.edges() if x[i] and x[j])


def cost(graph: nx.Graph, x: Bitstring, penalty: float = 2.0) -> float:
    """Compute C(x) = -sum_i x_i + penalty * sum_edges x_i x_j."""
    return -independent_set_size(x) + penalty * violation_count(graph, x)


def brute_force_mis(graph: nx.Graph, penalty: float = 2.0) -> dict:
    """Enumerate all bitstrings and return exact MIS and cost information."""
    rows = []
    for state in all_bitstrings(graph.number_of_nodes()):
        valid = is_valid_independent_set(graph, state)
        size = independent_set_size(state)
        rows.append(
            {
                "state": state,
                "state_str": bitstring_to_str(state),
                "cost": cost(graph, state, penalty),
                "size": size,
                "is_valid": valid,
                "violations": violation_count(graph, state),
                "selected_vertices": tuple(sorted(selected_vertices(state))),
            }
        )

    states_df = pd.DataFrame(rows).sort_values(
        ["cost", "is_valid", "state_str"], ascending=[True, False, True]
    )
    best_cost = float(states_df["cost"].min())
    best_bitstrings = list(states_df.loc[states_df["cost"] == best_cost, "state"])

    valid_df = states_df[states_df["is_valid"]]
    best_valid_size = int(valid_df["size"].max())
    optimal_valid_bitstrings = list(
        valid_df.loc[valid_df["size"] == best_valid_size, "state"]
    )

    return {
        "best_cost": best_cost,
        "best_bitstrings": best_bitstrings,
        "best_valid_size": best_valid_size,
        "optimal_valid_bitstrings": optimal_valid_bitstrings,
        "states_df": states_df.reset_index(drop=True),
    }

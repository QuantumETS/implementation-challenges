"""Tests for the toy QeMCMC MIS implementation."""

import numpy as np
import pytest

from qemcmc_mis.classical_mcmc import (
    greedy_independent_set_start,
    metropolis_accept,
)
from qemcmc_mis.graphs import make_graph_5, make_graph_8
from qemcmc_mis.mis import (
    all_bitstrings,
    brute_force_mis,
    cost,
    independent_set_size,
    is_valid_independent_set,
)


def test_all_bitstrings_count():
    assert len(all_bitstrings(5)) == 32
    assert len(all_bitstrings(8)) == 256


def test_cost_penalizes_invalid_edge():
    graph = make_graph_5()
    valid = (1, 0, 0, 1, 1)
    invalid = (1, 1, 0, 0, 0)

    assert is_valid_independent_set(graph, valid)
    assert not is_valid_independent_set(graph, invalid)
    assert cost(graph, invalid, penalty=graph.number_of_nodes()) > cost(
        graph, valid, penalty=graph.number_of_nodes()
    )


@pytest.mark.parametrize("graph", [make_graph_5(), make_graph_8()])
def test_greedy_start_is_valid(graph):
    state = greedy_independent_set_start(graph)
    assert is_valid_independent_set(graph, state)
    assert independent_set_size(state) >= 1


@pytest.mark.parametrize("graph", [make_graph_5(), make_graph_8()])
def test_brute_force_returns_valid_mis(graph):
    result = brute_force_mis(graph, penalty=graph.number_of_nodes())
    expected_size = max(
        independent_set_size(state)
        for state in all_bitstrings(graph.number_of_nodes())
        if is_valid_independent_set(graph, state)
    )

    assert result["best_valid_size"] == expected_size
    assert all(
        is_valid_independent_set(graph, state)
        for state in result["optimal_valid_bitstrings"]
    )


def test_metropolis_always_accepts_lower_cost():
    rng = np.random.default_rng(123)
    assert metropolis_accept(current_cost=3.0, proposed_cost=2.0, beta=1.5, rng=rng)

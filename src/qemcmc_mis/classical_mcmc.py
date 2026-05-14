"""Classical Metropolis MCMC samplers for MIS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from qemcmc_mis.mis import (
    Bitstring,
    bitstring_to_str,
    cost,
    independent_set_size,
    is_valid_independent_set,
)


def random_bitstring(n: int, rng: np.random.Generator) -> Bitstring:
    """Sample a random n-bit state."""
    return tuple(int(bit) for bit in rng.integers(0, 2, size=n))


def flip_one_bit(x: Bitstring, index: int) -> Bitstring:
    """Return x with one bit flipped."""
    values = list(x)
    values[index] = 1 - values[index]
    return tuple(values)


def propose_bit_flip(x: Bitstring, rng: np.random.Generator) -> Bitstring:
    """Propose a local move by flipping one uniformly random bit."""
    return flip_one_bit(x, int(rng.integers(0, len(x))))


def metropolis_accept(
    current_cost: float,
    proposed_cost: float,
    beta: float,
    rng: np.random.Generator,
) -> bool:
    """Return whether a proposed Metropolis move is accepted."""
    delta = proposed_cost - current_cost
    if delta <= 0:
        return True
    return bool(rng.random() < np.exp(-beta * delta))


def greedy_independent_set_start(graph: nx.Graph) -> Bitstring:
    """Build a deterministic greedy independent-set initial state."""
    selected: set[int] = set()
    for node in sorted(graph.nodes(), key=lambda v: (graph.degree[v], v)):
        if all(not graph.has_edge(node, chosen) for chosen in selected):
            selected.add(node)
    return tuple(
        1 if node in selected else 0 for node in range(graph.number_of_nodes())
    )


def _record_row(
    iteration: int,
    state: Bitstring,
    state_cost: float,
    accepted: bool,
    proposed_state: Bitstring,
    proposed_cost: float,
    best_state: Bitstring,
    best_cost: float,
    graph: nx.Graph,
) -> dict:
    return {
        "iteration": iteration,
        "state": state,
        "state_str": bitstring_to_str(state),
        "cost": state_cost,
        "is_valid": is_valid_independent_set(graph, state),
        "size": independent_set_size(state),
        "accepted": accepted,
        "proposed_state": proposed_state,
        "proposed_state_str": bitstring_to_str(proposed_state),
        "proposed_cost": proposed_cost,
        "best_state": best_state,
        "best_state_str": bitstring_to_str(best_state),
        "best_cost": best_cost,
        "best_size": independent_set_size(best_state),
        "best_is_valid": is_valid_independent_set(graph, best_state),
    }


def run_classical_mcmc(
    graph: nx.Graph,
    initial_state: Bitstring,
    iterations: int,
    beta: float,
    penalty: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run a classical Metropolis chain with local bit-flip proposals."""
    current = initial_state
    current_cost = cost(graph, current, penalty)
    best_state = current
    best_cost = current_cost
    rows = [
        _record_row(
            0,
            current,
            current_cost,
            True,
            current,
            current_cost,
            best_state,
            best_cost,
            graph,
        )
    ]

    for iteration in range(1, iterations + 1):
        proposed = propose_bit_flip(current, rng)
        proposed_cost = cost(graph, proposed, penalty)
        accepted = metropolis_accept(current_cost, proposed_cost, beta, rng)
        if accepted:
            current = proposed
            current_cost = proposed_cost
        if current_cost < best_cost:
            best_state = current
            best_cost = current_cost

        rows.append(
            _record_row(
                iteration,
                current,
                current_cost,
                accepted,
                proposed,
                proposed_cost,
                best_state,
                best_cost,
                graph,
            )
        )

    return pd.DataFrame(rows)

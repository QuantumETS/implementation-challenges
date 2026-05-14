"""Qiskit-based toy proposal kernel for QeMCMC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from qemcmc_mis.classical_mcmc import _record_row, metropolis_accept
from qemcmc_mis.mis import Bitstring, cost


def _qiskit_imports():
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        msg = "Qiskit and qiskit-aer are required for the quantum proposal kernel."
        raise ImportError(msg) from exc
    return QuantumCircuit, transpile, AerSimulator


def build_quantum_proposal_circuit(
    graph: nx.Graph,
    current_state: Bitstring,
    theta: float,
    gamma: float,
    use_edge_phases: bool = True,
):
    """Build a small graph-aware proposal circuit initialized at current_state."""
    QuantumCircuit, _, _ = _qiskit_imports()
    n = graph.number_of_nodes()
    qc = QuantumCircuit(n, n)

    for i, bit in enumerate(current_state):
        if bit:
            qc.x(i)
    for i in range(n):
        qc.rx(theta, i)
    if use_edge_phases:
        for i, j in graph.edges():
            qc.cx(i, j)
            qc.rz(gamma, j)
            qc.cx(i, j)
    for i in range(n):
        qc.rx(theta / 2, i)
    qc.measure(range(n), range(n))
    return qc


def make_simulator():
    """Create an Aer simulator, keeping Qiskit imports lazy for testability."""
    _, _, AerSimulator = _qiskit_imports()
    return AerSimulator()


def sample_quantum_proposal(
    graph: nx.Graph,
    current_state: Bitstring,
    simulator,
    theta: float,
    gamma: float,
    rng: np.random.Generator,
    shots: int = 1,
    use_edge_phases: bool = True,
) -> Bitstring:
    """Sample a proposed bitstring from the Qiskit circuit."""
    _, transpile, _ = _qiskit_imports()
    circuit = build_quantum_proposal_circuit(
        graph, current_state, theta, gamma, use_edge_phases
    )
    transpiled = transpile(circuit, simulator)
    seed = int(rng.integers(0, 2**32 - 1))
    job = simulator.run(transpiled, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    if shots == 1:
        measured = next(iter(counts))
    else:
        states = list(counts.keys())
        weights = np.array([counts[state] for state in states], dtype=float)
        measured = str(rng.choice(states, p=weights / weights.sum()))
    return tuple(int(bit) for bit in measured[::-1])


def run_qemcmc(
    graph: nx.Graph,
    initial_state: Bitstring,
    iterations: int,
    beta: float,
    penalty: float,
    theta: float,
    gamma: float,
    rng: np.random.Generator,
    shots_per_proposal: int = 1,
    use_edge_phases: bool = True,
) -> pd.DataFrame:
    """Run Metropolis MCMC with a Qiskit-generated proposal at each step."""
    simulator = make_simulator()
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
        proposed = sample_quantum_proposal(
            graph,
            current,
            simulator,
            theta,
            gamma,
            rng,
            shots=shots_per_proposal,
            use_edge_phases=use_edge_phases,
        )
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

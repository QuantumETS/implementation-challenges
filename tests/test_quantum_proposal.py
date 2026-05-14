"""Smoke tests for the optional Qiskit proposal kernel."""

import numpy as np
import pytest

from qemcmc_mis.graphs import make_graph_5
from qemcmc_mis.quantum_proposal import make_simulator, sample_quantum_proposal


def test_quantum_proposal_returns_valid_length():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    graph = make_graph_5()
    simulator = make_simulator()
    state = sample_quantum_proposal(
        graph,
        current_state=(0, 0, 0, 0, 0),
        simulator=simulator,
        theta=0.7,
        gamma=0.4,
        rng=np.random.default_rng(123),
    )

    assert len(state) == graph.number_of_nodes()
    assert all(bit in (0, 1) for bit in state)

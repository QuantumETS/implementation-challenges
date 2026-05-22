"""Knapsack utilities and QUBO mapping helpers.

This module provides helper functions to evaluate knapsack objective and
penalty terms and a mapper that converts a knapsack instance into an Ising
Hamiltonian (QUBO) using Qiskit Optimization. It also exposes a small
script-style entrypoint for local demonstrations.
"""

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.applications import Knapsack


@dataclass
class KnapsackInstance:
    """Representation of a knapsack problem instance.

    Attributes:
        values: List of integer values (profits) for each item.
        weights: List of integer weights for each item. Must be same length as
            ``values``.
        capacity: Maximum allowed total weight (knapsack capacity).
    """

    values: list[int]
    weights: list[int]
    capacity: int


VALUES = [8, 5, 6, 9, 4, 7, 3, 10]
WEIGHTS = [4, 3, 5, 6, 2, 4, 1, 7]
CAPACITY = 12


def cost_function(x: list[int]) -> float:
    """Compute the (negative) total value for a binary selection vector.

    The function returns the negative sum of selected item values so that a
    minimizer will prefer selections with larger total value.

    Args:
        x: Binary list where 1 indicates the item is selected and 0 otherwise.

    Returns:
        The cost as a float (negative total value).
    """
    cost = 0
    for i in range(len(x)):
        cost -= x[i] * VALUES[i]

    return cost


def penalty_function(x: list[int], penalty_value: float = sum(VALUES)) -> float:
    """Compute a quadratic penalty for capacity violation.

    The penalty is applied when the total weight of the selected items
    exceeds the knapsack capacity. It uses a quadratic term scaled by
    ``penalty_value`` to heavily penalize infeasible selections.

    Args:
        x: Binary list where 1 indicates the item is selected.
        penalty_value: Scaling factor for the penalty (defaults to sum of
            global `VALUES`).

    Returns:
        The penalty term as a float.
    """
    penalty = penalty_value * np.square(
        max(0, sum(x[i] * WEIGHTS[i] for i in range(len(x))) - CAPACITY)
    )

    return penalty


def objective_function(x: list[int]) -> float:
    """Compute the combined objective (cost + penalty) for a selection.

    Args:
        x: Binary selection list.

    Returns:
        The scalar objective value (cost + penalty).
    """
    return cost_function(x) + penalty_function(x)


def map_hamiltonian(knapsack_instance: KnapsackInstance) -> tuple[SparsePauliOp, float]:
    """Convert a `KnapsackInstance` into an Ising Hamiltonian.

    This helper constructs a Qiskit Optimization `Knapsack` application
    model, converts it to a `QuadraticProgram`, maps it to a QUBO and then
    returns the Ising representation (a `SparsePauliOp`) together with the
    associated energy offset.

    Args:
        knapsack_instance: The knapsack instance to convert.

    Returns:
        A tuple ``(hamiltonian, offset)`` where ``hamiltonian`` is a
        `qiskit.quantum_info.SparsePauliOp` describing the Ising operator and
        ``offset`` is the numeric energy offset (float).
    """
    knapsack = Knapsack(
        values=knapsack_instance.values,
        weights=knapsack_instance.weights,
        max_weight=knapsack_instance.capacity,
    )

    qprog: QuadraticProgram = knapsack.to_quadratic_program()
    qubo = QuadraticProgramToQubo(sum(knapsack_instance.values)).convert(qprog)

    return qubo.to_ising()


def knapsack_solver():
    """Script entrypoint that demonstrates mapping and prints the Hamiltonian.

    This function creates a sample :class:`KnapsackInstance`, converts it to an
    Ising Hamiltonian using :func:`map_hamiltonian`, and prints the resulting
    operator. It is intended for quick local demonstrations and testing.

    Note:
        The remainder of the quantum optimization pipeline (solvers, samplers,
        and result interpretation) is intentionally left as a TODO for
        integration in higher-level scripts or tests.
    """
    knapsack_instance = KnapsackInstance(
        values=[8, 5, 6, 9, 4, 7, 3, 10],
        weights=[4, 3, 5, 6, 2, 4, 1, 7],
        capacity=12,
    )

    hamiltonian, offset = map_hamiltonian(knapsack_instance)

    print(hamiltonian)

    # TODO : Add the rest of the quantum optimization pipeline.


if __name__ == "__main__":
    knapsack_solver()

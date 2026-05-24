"""Knapsack utilities and QUBO mapping helpers.

This module provides helper functions to evaluate knapsack objective and
penalty terms and a mapper that converts a knapsack instance into an Ising
Hamiltonian (QUBO) using Qiskit Optimization. It also exposes a small
script-style entrypoint for local demonstrations.
"""

from dataclasses import asdict, dataclass, is_dataclass
import itertools

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.applications import Knapsack
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator
from scipy.optimize import minimize, OptimizeResult

import logging
from argparse import ArgumentParser


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SEED = None  # Global seed for reproducibility, can be set from command-line arguments
OBJECTIVE_FUNC_VALUES = []


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


def cost_function(x: list[int], knapsack_instance: KnapsackInstance) -> float:
    """Compute the (negative) total value for a binary selection vector.

    The function returns the negative sum of selected item values so that a
    minimizer will prefer selections with larger total value.

    Args:
        x: Binary list where 1 indicates the item is selected and 0 otherwise.
        knapsack_instance: The knapsack instance containing item values and capacity.

    Returns:
        The cost as a float (negative total value).
    """
    cost = 0
    for i in range(len(x)):
        cost -= x[i] * knapsack_instance.values[i]

    return cost


def penalty_function(
    x: list[int], knapsack_instance: KnapsackInstance, penalty_value: float = None
) -> float:
    """Compute a quadratic penalty for capacity violation.

    The penalty is applied when the total weight of the selected items
    exceeds the knapsack capacity. It uses a quadratic term scaled by
    ``penalty_value`` to heavily penalize infeasible selections.

    Args:
        x: Binary list where 1 indicates the item is selected.
        knapsack_instance: The knapsack instance containing item weights and capacity.
        penalty_value: Scaling factor for the penalty (defaults to sum of
            global `VALUES`).

    Returns:
        The penalty term as a float.
    """
    if penalty_value is None:
        penalty_value = sum(knapsack_instance.values)

    penalty = penalty_value * np.square(
        max(
            0,
            sum(x[i] * knapsack_instance.weights[i] for i in range(len(x)))
            - knapsack_instance.capacity,
        )
    )

    return penalty


def objective_function(x: list[int], knapsack_instance: KnapsackInstance) -> float:
    """Compute the combined objective (cost + penalty) for a selection.

    Args:
        x: Binary selection list.
        knapsack_instance: The knapsack instance containing item values and capacity.

    Returns:
        The scalar objective value (cost + penalty).
    """
    return cost_function(x, knapsack_instance) + penalty_function(x, knapsack_instance)


def bruteforce(knapsack_instance: KnapsackInstance) -> tuple[list[int], float]:
    """Brute-force search for the optimal selection vector.

    This helper function exhaustively evaluates all possible binary selection
    vectors and returns the one with the lowest objective function value.

    Args:
        knapsack_instance: The knapsack instance to solve.

    Returns:
        The optimal binary selection vector as a list of integers (0 or 1) and its corresponding objective function value.
    """
    bestValue = float("inf")
    bestArray = []

    for test in itertools.product([0, 1], repeat=len(knapsack_instance.values)):
        logger.debug(f"Testing selection: {test}")
        test = list(test)
        newValue = objective_function(test, knapsack_instance)
        if newValue < bestValue:
            bestValue = newValue
            bestArray = test.copy()
            logger.debug(f"New best selection: {bestArray} with value {bestValue}")

    return bestArray, bestValue


def cost_function_estimator(
    params,
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: BaseEstimatorV2,
) -> float:
    """Estimate the cost function value for given parameters.

    This helper function takes a parameter vector, constructs the corresponding
    quantum state using the provided ansatz, and estimates the expectation
    value of the cost Hamiltonian using the given estimator.

    Args:
        params: Parameter vector for the ansatz.
        ansatz: The QAOA ansatz circuit.
        hamiltonian: The cost Hamiltonian as a SparsePauliOp.
        estimator: A Qiskit Estimator instance for evaluating expectation values.

    Returns:
        The estimated cost function value as a float.
    """
    pub = (ansatz, [hamiltonian], [params])

    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs

    OBJECTIVE_FUNC_VALUES.append(energy)

    return energy


def map_hamiltonian(
    knapsack_instance: KnapsackInstance, penalty_factor: float
) -> tuple[SparsePauliOp, float]:
    """Convert a `KnapsackInstance` into an Ising Hamiltonian.

    This helper constructs a Qiskit Optimization `Knapsack` application
    model, converts it to a `QuadraticProgram`, maps it to a QUBO and then
    returns the Ising representation (a `SparsePauliOp`) together with the
    associated energy offset.

    Args:
        knapsack_instance: The knapsack instance to convert.
        penalty_factor: The scaling factor for the penalty term.

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
    qubo = QuadraticProgramToQubo(penalty=penalty_factor).convert(qprog)

    return qubo.to_ising()


def pstate(knapsack_instance: KnapsackInstance) -> QuantumCircuit:
    """Construct a quantum circuit that prepares a reference state from a
    `KnapsackInstance`.

    Args:
        knapsack_instance: The knapsack instance to create the state for.

    Returns:
        A `QuantumCircuit` that prepares a reference state (e.g., all zeros).
    """
    num_qubits = len(knapsack_instance.values)
    circuit = QuantumCircuit(num_qubits)

    # TODO : Implement a state including the most valuable item

    return circuit


def _x0_parameters(num_params) -> np.ndarray:
    """Generate deterministic initial parameters for the optimizer.

    Args:
        num_params: Number of parameters to generate.

    Returns:
        np.ndarray: Seeded random initial parameter vector.
    """
    params = np.random.RandomState(seed=SEED).random(num_params)
    return params


def _pretty_result(result: OptimizeResult) -> str:
    """Format the optimization result for logging.

    Args:
        result: The optimization result from scipy.optimize.minimize.

    Returns:
        A formatted string summarizing the optimization outcome.
    """
    return (
        f"Success: {result.success},\n"
        f"Final Cost: {result.fun:.4f},\n"
        f"Optimal Parameters: {result.x}"
    )


def _plot_results():
    """Plot the optimization history of the objective function values."""
    plt.figure(figsize=(10, 6))
    plt.plot(OBJECTIVE_FUNC_VALUES, marker="o")
    plt.title("Objective Function Value During Optimization")
    plt.xlabel("Evaluation Number")
    plt.ylabel("Estimated Cost Function Value")
    plt.grid()
    plt.show()


def _json_default(value):
    """Convert common non-JSON types in experiment results to plain Python objects."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _save_results(results: dict):
    """Save the results dictionary to a JSON file.

    Args:
        results: The dictionary containing optimization results and metadata.
    """
    import json
    import datetime as dt
    from pathlib import Path

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not Path("results").exists():
        Path("results").mkdir()
    filename = f"knapsack_results_{timestamp}.json"
    with open(f"results/{filename}", "w") as f:
        json.dump(results, f, indent=4, default=_json_default)
    logger.info(f"Results saved to results/{filename}")


def process_result(optimization_result: OptimizeResult, results: dict):
    """Process the optimization result and update the results dictionary.

    Args:
        optimization_result: The optimization result from scipy.optimize.minimize.
        results: The dictionary to update with quantum optimization results.
    """
    results["quantum"]["final_cost"] = optimization_result.fun
    results["quantum"]["optimal_parameters"] = optimization_result.x.tolist()
    _save_results(results)


def knapsack_solver(args):
    """Script entrypoint that demonstrates mapping and prints the Hamiltonian.

    This function creates a sample :class:`KnapsackInstance`, converts it to an
    Ising Hamiltonian using :func:`map_hamiltonian`, and prints the resulting
    operator. It is intended for quick local demonstrations and testing.
    """
    SEED = args.seed

    knapsack_instance = KnapsackInstance(
        values=[8, 5, 6, 9],
        weights=[4, 3, 5, 6],
        capacity=10,
    )

    results = {
        "metadata": {
            "seed": SEED,
            "optimizer": args.optimizer,
            "max_iterations": args.iterations,
            "qaoa_layers": args.qaoa_layers,
            "use_reference_state": args.use_reference_state,
            "penalty_scaling": args.penalty_scaling,
            "knapsack_instance": asdict(knapsack_instance),
        },
        "classical": {
            "solution": None,
            "value": None,
        },
        "quantum": {
            "solution": None,
            "final_cost": None,
            "optimal_parameters": None,
            "optimization_log": {},
        },
    }

    classical_solution, classical_value = bruteforce(knapsack_instance)
    logger.info(f"Classical solution: {classical_solution}, Value: {classical_value}")
    results["classical"]["solution"] = classical_solution
    results["classical"]["value"] = classical_value

    hamiltonian, offset = map_hamiltonian(
        knapsack_instance,
        penalty_factor=args.penalty_scaling,
    )
    logger.info(f"Mapped Hamiltonian:\n{hamiltonian}\nOffset: {offset}")
    p_state = None

    if args.use_reference_state:
        p_state = pstate(knapsack_instance)
    ansatz = QAOAAnsatz(hamiltonian, reps=args.qaoa_layers, initial_state=p_state)

    logger.info(f"Number of qubits: {ansatz.num_qubits}")
    logger.info(f"Initial parameters: {ansatz.parameters}")

    estimator = StatevectorEstimator(seed=SEED)

    initial_params = _x0_parameters(ansatz.num_parameters)

    optimization_result = minimize(
        cost_function_estimator,
        x0=initial_params,
        args=(ansatz, hamiltonian, estimator),
        method=args.optimizer,
        options={"maxiter": args.iterations, "disp": True},
    )
    process_result(optimization_result, results)
    _plot_results()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Quantum resolution of the Knapsack problem using QAOA."
    )
    parser.add_argument(
        "--use-reference-state",
        action="store_true",
        help="Use a reference prepared state",
    )
    parser.add_argument(
        "--optimizer",
        choices=["COBYLA", "SLSQP", "ADAM"],
        default="COBYLA",
        help="Optimizer to use for parameter optimization",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for the optimizer",
    )
    parser.add_argument(
        "--qaoa-layers",
        type=int,
        default=1,
        help="Number of QAOA layers (reps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--penalty-scaling",
        type=float,
        help="Scaling factor for the penalty",
    )

    args = parser.parse_args()
    knapsack_solver(args)

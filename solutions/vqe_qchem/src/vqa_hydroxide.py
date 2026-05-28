import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize, OptimizeResult
from ham_sdk_conversion import map_pl_hamiltonian_to_qiskit
import pennylane as qml
import pprint as pp
import logging

SEED = 67

_PACKAGE_LOGGER_NAME = __package__ or __name__
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def _configure_package_logger() -> logging.Logger:
    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    package_logger.setLevel(logging.DEBUG)

    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        package_logger.addHandler(handler)

    # Keep this logging setup local to this package logger.
    package_logger.propagate = False
    return package_logger


logger = _configure_package_logger()


def classical_diagonalization(hamiltonian: SparsePauliOp) -> float:
    """Classically diagonalize the Hamiltonian to find the ground state energy.

    Args:
        hamiltonian: Hamiltonian operator to diagonalize.

    Returns:
        float: Ground state energy of the Hamiltonian.
    """
    # Convert the SparsePauliOp to a dense matrix
    dense_matrix = hamiltonian.to_matrix()

    # Diagonalize the matrix to find eigenvalues
    eigenvalues, _ = np.linalg.eigh(dense_matrix)

    # The ground state energy is the minimum eigenvalue
    ground_state_energy = np.min(eigenvalues)

    return ground_state_energy

def vqe_loop(input_hamiltonian: SparsePauliOp) -> OptimizeResult:
    """Run a VQE optimization using only the most basic Qiskit components.

    Args:
        input_hamiltonian: Hamiltonian operator to minimize.

    Returns:
        scipy.optimize.OptimizeResult: Optimization result returned by ``minimize``.

    Example:
        >>> from hamiltonian import Hamiltonian
        >>> result = vqe_bare(Hamiltonian.H2_STO6G_REDUX.value)
    """

    # Define the ansatz
    ansatz = n_local(
        num_qubits = input_hamiltonian.num_qubits, # type: ignore
        rotation_blocks = ["rx", "rz"],
        entanglement_blocks = ["cx"],
        reps = 1,
        entanglement = "linear",
    )

    # Build the Hamiltonian
    hamiltonian = input_hamiltonian

    # Set up the estimator
    estimator = StatevectorEstimator(seed=SEED)

    # Define the cost function for optimization
    def cost_function(
        params, ansatz=ansatz, hamiltonian=hamiltonian, estimator=estimator
    ) -> float:
        pub_estimate = (
            ansatz,
            [hamiltonian],
            [params],
        )
        result = estimator.run(
            pubs=[pub_estimate],
        ).result()
        energy = result[0].data.evs[0]

        return energy

    # Run the optimization
    initial_params = np.random.RandomState(seed=SEED).rand(ansatz.num_parameters)  # type: ignore

    optimization_result = minimize(
        cost_function,
        x0=initial_params,
        args=(ansatz, hamiltonian, estimator),
        method="COBYLA",
        options={"maxiter": 150},
        callback=lambda xk: logger.debug(f"Current parameters: {xk}, Current energy: {cost_function(xk)}"),
    )

    return optimization_result

def hydroxide_solver():
    """Run the VQE optimization for the hydroxide ion (OH-).

    Returns:
        scipy.optimize.OptimizeResult: Optimization result returned by ``minimize``.
    """
    hydroxide_dataset = qml.data.load("qchem", molname="OH-", bondlength=0.964, basis="STO-3G")

    logger.info(f"Tapered Hamiltonian: {hydroxide_dataset[0].tapered_hamiltonian}")
    hamiltonian = map_pl_hamiltonian_to_qiskit(hydroxide_dataset[0].tapered_hamiltonian)

    logger.info(f"Classical diagonalization energy: {classical_diagonalization(hamiltonian)}")

    # Run the VQE optimization
    result = vqe_loop(hamiltonian)

    logger.info(f"VQE optimization result: {result}")



if __name__ == "__main__":
    hydroxide_solver()
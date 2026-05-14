# Toy QeMCMC for Maximum Independent Set

This repository contains a reproducible reference solution for `challenges/QeMCMC-challenge.md`.

It implements two small Maximum Independent Set instances, exact brute-force verification, classical random-start Metropolis MCMC, warm-started Metropolis MCMC, and a toy Qiskit/Aer proposal kernel inside a classical Metropolis accept/reject loop.

This implementation is a toy reproduction of the workflow structure, not a reproduction of hardware-scale results and not a demonstration of quantum advantage.

## Structure

```text
notebooks/qemcmc_mis_reference_solution.ipynb
scripts/run_experiments.py
src/qemcmc_mis/
tests/
results/data/
results/figures/
```

## Setup

Use Python 3.11 or 3.12 for Qiskit compatibility.

```bash
uv sync --extra dev
```

Without `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib networkx qiskit qiskit-aer jupyter tqdm pytest ruff
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run Tests

```bash
pytest tests/
```

The Qiskit smoke test is skipped automatically when `qiskit` or `qiskit-aer` is not installed.

## Run Experiments

Reduced feasibility run:

```bash
python scripts/run_experiments.py --iterations 50 --repetitions 5
```

Classical-only run when Qiskit is not installed:

```bash
python scripts/run_experiments.py --iterations 50 --repetitions 5 --skip-qemcmc
```

Fuller local run:

```bash
python scripts/run_experiments.py --iterations 200 --repetitions 20
```

Outputs are written to:

- `results/data/*.csv`
- `results/figures/*.png`

## Notebook

Open `notebooks/qemcmc_mis_reference_solution.ipynb` for a readable walkthrough of the assignment: graph definitions, exact optima, sampler traces, repeated-run comparisons, plots, discussion, and limitations.

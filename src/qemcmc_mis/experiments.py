"""Experiment orchestration for repeated MIS sampler comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from qemcmc_mis.classical_mcmc import (
    greedy_independent_set_start,
    random_bitstring,
    run_classical_mcmc,
)
from qemcmc_mis.mis import bitstring_to_str, brute_force_mis
from qemcmc_mis.quantum_proposal import run_qemcmc

METHODS = ("classical_random", "classical_warm", "qemcmc")


def run_single_method(
    graph: nx.Graph,
    method: str,
    iterations: int,
    beta: float,
    penalty: float,
    seed: int,
    theta: float = 0.7,
    gamma: float = 0.4,
) -> pd.DataFrame:
    """Run one sampler/method with explicit seeding."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")

    rng = np.random.default_rng(seed)
    if method == "classical_warm":
        initial = greedy_independent_set_start(graph)
        initialization = "greedy"
    else:
        initial = random_bitstring(graph.number_of_nodes(), rng)
        initialization = "random"

    if method == "qemcmc":
        df = run_qemcmc(graph, initial, iterations, beta, penalty, theta, gamma, rng)
    else:
        df = run_classical_mcmc(graph, initial, iterations, beta, penalty, rng)

    df = df.copy()
    df["method"] = method
    df["seed"] = seed
    df["initialization"] = initialization
    df["initial_state"] = [initial] * len(df)
    df["initial_state_str"] = bitstring_to_str(initial)
    return df


def summarize_run(
    graph: nx.Graph,
    graph_name: str,
    method: str,
    run_id: int,
    trace_df: pd.DataFrame,
    penalty: float,
) -> dict:
    """Summarize a single trace against the exact MIS optimum."""
    brute = brute_force_mis(graph, penalty)
    optimal_strings = {
        bitstring_to_str(state) for state in brute["optimal_valid_bitstrings"]
    }
    optimum_rows = trace_df[trace_df["best_state_str"].isin(optimal_strings)]
    first_optimum = (
        float(optimum_rows["iteration"].min()) if len(optimum_rows) else np.nan
    )

    final = trace_df.iloc[-1]
    return {
        "graph_name": graph_name,
        "method": method,
        "run_id": run_id,
        "found_optimum": final["best_state_str"] in optimal_strings,
        "final_cost": float(final["cost"]),
        "best_cost": float(final["best_cost"]),
        "best_size": int(final["best_size"]),
        "best_is_valid": bool(final["best_is_valid"]),
        "first_optimum_iteration": first_optimum,
        "unique_states_visited": int(trace_df["state_str"].nunique()),
        "valid_sample_fraction": float(trace_df["is_valid"].mean()),
    }


def run_repeated_experiments(
    graph: nx.Graph,
    graph_name: str,
    methods: list[str],
    repetitions: int,
    iterations: int,
    beta: float,
    penalty: float,
    base_seed: int,
    theta: float = 0.7,
    gamma: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-iteration traces and one-row-per-run summaries."""
    traces = []
    summaries = []
    for method_index, method in enumerate(methods):
        for run_id in range(repetitions):
            seed = base_seed + 10_000 * method_index + run_id
            df = run_single_method(
                graph, method, iterations, beta, penalty, seed, theta, gamma
            )
            df = df.copy()
            df["graph_name"] = graph_name
            df["run_id"] = run_id
            traces.append(df)
            summaries.append(
                summarize_run(graph, graph_name, method, run_id, df, penalty)
            )

    return pd.concat(traces, ignore_index=True), pd.DataFrame(summaries)

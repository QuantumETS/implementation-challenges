"""Run reproducible toy QeMCMC experiments and save data/figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qemcmc_mis.experiments import run_repeated_experiments  # noqa: E402
from qemcmc_mis.graphs import get_problem_graphs  # noqa: E402
from qemcmc_mis.mis import brute_force_mis  # noqa: E402
from qemcmc_mis.plotting import (  # noqa: E402
    plot_best_cost_convergence,
    plot_best_size_convergence,
    plot_final_cost_distribution,
    plot_graph,
    plot_sampled_bitstring_histogram,
    plot_sampled_cost_histogram,
    plot_success_rates,
    plot_valid_invalid_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--skip-qemcmc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = ROOT / "results" / "data"
    figures_dir = ROOT / "results" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    methods = ["classical_random", "classical_warm"]
    if not args.skip_qemcmc:
        methods.append("qemcmc")

    all_traces = []
    all_summaries = []
    for graph_name, graph in get_problem_graphs().items():
        penalty = graph.number_of_nodes()
        traces, summary = run_repeated_experiments(
            graph=graph,
            graph_name=graph_name,
            methods=methods,
            repetitions=args.repetitions,
            iterations=args.iterations,
            beta=args.beta,
            penalty=penalty,
            base_seed=args.base_seed,
        )
        all_traces.append(traces)
        all_summaries.append(summary)

        traces.to_csv(data_dir / f"{graph_name}_traces.csv", index=False)
        summary.to_csv(data_dir / f"{graph_name}_summary.csv", index=False)

        brute = brute_force_mis(graph, penalty)
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_graph(graph, graph_name, ax=ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{graph_name}_graph.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        plot_best_cost_convergence(traces, graph_name, brute["best_cost"], ax=ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{graph_name}_best_cost.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        plot_best_size_convergence(traces, graph_name, ax=ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{graph_name}_best_size.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        plot_final_cost_distribution(summary, graph_name, ax=ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{graph_name}_final_cost.png", dpi=160)
        plt.close(fig)

    traces_df = pd.concat(all_traces, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    traces_df.to_csv(data_dir / "all_traces.csv", index=False)
    summary_df.to_csv(data_dir / "all_summary.csv", index=False)

    first_trace = traces_df[
        (traces_df["graph_name"] == "graph_5")
        & (traces_df["method"] == methods[0])
        & (traces_df["run_id"] == 0)
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_sampled_bitstring_histogram(first_trace, "graph_5 sampled bitstrings", ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "graph_5_sampled_bitstrings.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_sampled_cost_histogram(first_trace, "graph_5 sampled costs", ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "graph_5_sampled_costs.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_success_rates(summary_df, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "success_rates.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_valid_invalid_fraction(summary_df, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "valid_fraction.png", dpi=160)
    plt.close(fig)

    print(summary_df.groupby(["graph_name", "method"])["found_optimum"].mean())


if __name__ == "__main__":
    main()

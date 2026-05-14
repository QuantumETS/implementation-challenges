"""Matplotlib plotting helpers for MIS/QeMCMC experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_graph(graph: nx.Graph, title: str, ax=None):
    """Plot a problem graph."""
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    pos = nx.spring_layout(graph, seed=7)
    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        node_color="#2a9d8f",
        edge_color="#334155",
        font_color="white",
        node_size=700,
    )
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_energy_landscape_5(states_df: pd.DataFrame, title: str, ax=None):
    """Plot the full 5-node energy landscape."""
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    ordered = states_df.sort_values("state_str")
    colors = ordered["is_valid"].map({True: "#2a9d8f", False: "#e76f51"})
    ax.bar(ordered["state_str"], ordered["cost"], color=colors)
    ax.set_title(title)
    ax.set_xlabel("bitstring")
    ax.set_ylabel("cost")
    ax.tick_params(axis="x", rotation=90)
    return ax


def _mean_by_iteration(traces_df: pd.DataFrame, graph_name: str, value: str):
    return (
        traces_df[traces_df["graph_name"] == graph_name]
        .groupby(["method", "iteration"], as_index=False)[value]
        .mean()
    )


def plot_best_cost_convergence(
    traces_df: pd.DataFrame,
    graph_name: str,
    optimal_cost: float | None = None,
    ax=None,
):
    """Plot mean best cost over repetitions per method."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    data = _mean_by_iteration(traces_df, graph_name, "best_cost")
    for method, group in data.groupby("method"):
        ax.plot(group["iteration"], group["best_cost"], label=method)
    if optimal_cost is not None:
        ax.axhline(optimal_cost, color="#111827", linestyle="--", label="exact optimum")
    ax.set_title(f"{graph_name}: best cost convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best cost")
    ax.legend()
    return ax


def plot_best_size_convergence(traces_df: pd.DataFrame, graph_name: str, ax=None):
    """Plot mean best independent-set size over repetitions per method."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    data = _mean_by_iteration(traces_df, graph_name, "best_size")
    for method, group in data.groupby("method"):
        ax.plot(group["iteration"], group["best_size"], label=method)
    ax.set_title(f"{graph_name}: best size convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best independent-set size")
    ax.legend()
    return ax


def plot_final_cost_distribution(summary_df: pd.DataFrame, graph_name: str, ax=None):
    """Plot final best-cost distribution by method."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    data = summary_df[summary_df["graph_name"] == graph_name]
    methods = list(data["method"].unique())
    ax.boxplot([data.loc[data["method"] == method, "best_cost"] for method in methods])
    ax.set_xticklabels(methods, rotation=20)
    ax.set_title(f"{graph_name}: final best-cost distribution")
    ax.set_ylabel("best cost")
    return ax


def plot_success_rates(summary_df: pd.DataFrame, ax=None):
    """Plot optimum-found rate by graph and method."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    rates = (
        summary_df.groupby(["graph_name", "method"])["found_optimum"]
        .mean()
        .unstack("method")
    )
    rates.plot(kind="bar", ax=ax)
    ax.set_title("success rate by method")
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=0)
    return ax


def plot_sampled_bitstring_histogram(
    trace_df: pd.DataFrame, title: str, top_k: int = 15, ax=None
):
    """Plot the most frequently sampled bitstrings."""
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    counts = trace_df["state_str"].value_counts().head(top_k).sort_values()
    ax.barh(counts.index, counts.values, color="#457b9d")
    ax.set_title(title)
    ax.set_xlabel("samples")
    return ax


def plot_sampled_cost_histogram(trace_df: pd.DataFrame, title: str, ax=None):
    """Plot sampled cost frequencies."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    ax.hist(trace_df["cost"], bins=20, color="#8ab17d", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("sampled cost")
    ax.set_ylabel("frequency")
    return ax


def plot_valid_invalid_fraction(summary_df: pd.DataFrame, ax=None):
    """Plot mean valid sampled fraction by graph and method."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    rates = (
        summary_df.groupby(["graph_name", "method"])["valid_sample_fraction"]
        .mean()
        .unstack("method")
    )
    rates.plot(kind="bar", ax=ax)
    ax.set_title("valid sampled fraction")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=0)
    return ax

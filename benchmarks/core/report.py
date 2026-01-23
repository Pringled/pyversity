"""Plot generation from benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _setup_plot_style() -> None:
    """Set up seaborn styling for plots."""
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")


STRATEGIES = ["mmr", "msd", "dpp", "ssd"]

# Relevance floor: keep configs with at least this fraction of baseline nDCG
RELEVANCE_FLOOR = 0.95


def _compute_relevance_budgeted_scores(
    all_data: list[dict],
) -> dict[str, dict[str, dict[str, float | str]]]:
    """
    Compute best configs per strategy under a relevance floor constraint.

    For each dataset, finds strategies that maximize ILAD, ILMD, and combined diversity
    while maintaining >= 95% of baseline (lambda=0) nDCG.

    Returns dict mapping dataset -> goal -> {strategy, lambda, ndcg, ilad, ilmd, score}
    """
    datasets = sorted(set(p["dataset"] for p in all_data))
    results: dict[str, dict[str, dict[str, float | str]]] = {}

    for dataset in datasets:
        dataset_points = [p for p in all_data if p["dataset"] == dataset]

        # Find baseline nDCG (lambda=0, or minimum lambda for each strategy)
        baseline_ndcg = 0.0
        for strategy in STRATEGIES:
            strategy_points = [p for p in dataset_points if p["strategy"] == strategy]
            if strategy_points:
                min_lambda_point = min(strategy_points, key=lambda x: x["lambda"])
                baseline_ndcg = max(baseline_ndcg, min_lambda_point["ndcg"])

        if baseline_ndcg == 0:
            continue

        ndcg_floor = RELEVANCE_FLOOR * baseline_ndcg

        # Find baseline (lambda=0) and max for normalization
        ilad_baseline = (
            min(p["ilad"] for p in dataset_points if p["lambda"] == 0.0)
            if any(p["lambda"] == 0.0 for p in dataset_points)
            else min(p["ilad"] for p in dataset_points)
        )
        ilmd_baseline = (
            min(p["ilmd"] for p in dataset_points if p["lambda"] == 0.0)
            if any(p["lambda"] == 0.0 for p in dataset_points)
            else min(p["ilmd"] for p in dataset_points)
        )
        ilad_max = max(p["ilad"] for p in dataset_points)
        ilmd_max = max(p["ilmd"] for p in dataset_points)

        # Filter to feasible configs (above relevance floor)
        feasible = [p for p in dataset_points if p["ndcg"] >= ndcg_floor]

        if not feasible:
            continue

        results[dataset] = {}

        # Goal 1: Max ILAD (tie-break by higher nDCG, lower lambda)
        best_ilad = max(feasible, key=lambda x: (x["ilad"], x["ndcg"], -x["lambda"]))
        results[dataset]["max_ilad"] = {
            "strategy": best_ilad["strategy"],
            "lambda": best_ilad["lambda"],
            "ndcg": best_ilad["ndcg"],
            "ndcg_vs_baseline": best_ilad["ndcg"] / baseline_ndcg,
            "ilad": best_ilad["ilad"],
            "ilmd": best_ilad["ilmd"],
        }

        # Goal 2: Max ILMD (tie-break by higher nDCG, lower lambda)
        best_ilmd = max(feasible, key=lambda x: (x["ilmd"], x["ndcg"], -x["lambda"]))
        results[dataset]["max_ilmd"] = {
            "strategy": best_ilmd["strategy"],
            "lambda": best_ilmd["lambda"],
            "ndcg": best_ilmd["ndcg"],
            "ndcg_vs_baseline": best_ilmd["ndcg"] / baseline_ndcg,
            "ilad": best_ilmd["ilad"],
            "ilmd": best_ilmd["ilmd"],
        }

        # Goal 3: Best combined (geometric mean of normalized gains relative to baseline)
        def combined_score(point: dict) -> float:
            ilad_range = ilad_max - ilad_baseline
            ilmd_range = ilmd_max - ilmd_baseline
            if ilad_range == 0 or ilmd_range == 0:
                return 0.0
            ilad_gain = (point["ilad"] - ilad_baseline) / ilad_range
            ilmd_gain = (point["ilmd"] - ilmd_baseline) / ilmd_range
            return (ilad_gain * ilmd_gain) ** 0.5  # Geometric mean

        # Tie-break by higher nDCG, lower lambda
        best_combined = max(feasible, key=lambda p: (combined_score(p), p["ndcg"], -p["lambda"]))
        results[dataset]["best_combined"] = {
            "strategy": best_combined["strategy"],
            "lambda": best_combined["lambda"],
            "ndcg": best_combined["ndcg"],
            "ndcg_vs_baseline": best_combined["ndcg"] / baseline_ndcg,
            "ilad": best_combined["ilad"],
            "ilmd": best_combined["ilmd"],
            "score": combined_score(best_combined),
        }

    return results


def _get_dataset_baseline_and_bounds(
    dataset_points: list[dict],
) -> tuple[float, float, float, float, float] | None:
    """Get baseline nDCG and normalization bounds (baseline, not min) for a dataset."""
    baseline_ndcg = 0.0
    for strategy in STRATEGIES:
        strategy_points = [p for p in dataset_points if p["strategy"] == strategy]
        if strategy_points:
            min_lambda_point = min(strategy_points, key=lambda x: x["lambda"])
            baseline_ndcg = max(baseline_ndcg, min_lambda_point["ndcg"])

    if baseline_ndcg == 0:
        return None

    # Use baseline (lambda=0) values, not min
    has_baseline = any(p["lambda"] == 0.0 for p in dataset_points)
    if has_baseline:
        ilad_baseline = float(min(p["ilad"] for p in dataset_points if p["lambda"] == 0.0))
        ilmd_baseline = float(min(p["ilmd"] for p in dataset_points if p["lambda"] == 0.0))
    else:
        ilad_baseline = float(min(p["ilad"] for p in dataset_points))
        ilmd_baseline = float(min(p["ilmd"] for p in dataset_points))

    ilad_max = float(max(p["ilad"] for p in dataset_points))
    ilmd_max = float(max(p["ilmd"] for p in dataset_points))

    if ilad_max == ilad_baseline or ilmd_max == ilmd_baseline:
        return None

    return baseline_ndcg, ilad_baseline, ilad_max, ilmd_baseline, ilmd_max


def _compute_strategy_scorecard(all_data: list[dict]) -> dict[str, dict[str, float]]:
    """
    Compute per-strategy aggregate scores across all datasets.

    Returns combined score, ILAD/ILMD at best operating point, nDCG retention, and typical diversity.
    """
    datasets = sorted(set(p["dataset"] for p in all_data))
    strategy_scores: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    strategy_ilads: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    strategy_ilmds: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    strategy_ndcg_ret: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    strategy_lambdas: dict[str, list[float]] = {s: [] for s in STRATEGIES}

    for dataset in datasets:
        dataset_points = [p for p in all_data if p["dataset"] == dataset]
        bounds = _get_dataset_baseline_and_bounds(dataset_points)
        if bounds is None:
            continue

        baseline_ndcg, ilad_baseline, ilad_max, ilmd_baseline, ilmd_max = bounds
        ndcg_floor = RELEVANCE_FLOOR * baseline_ndcg
        ilad_range = ilad_max - ilad_baseline
        ilmd_range = ilmd_max - ilmd_baseline

        def combined_score(point: dict) -> float:
            ilad_gain = (point["ilad"] - ilad_baseline) / ilad_range
            ilmd_gain = (point["ilmd"] - ilmd_baseline) / ilmd_range
            return (ilad_gain * ilmd_gain) ** 0.5

        for strategy in STRATEGIES:
            strategy_points = [p for p in dataset_points if p["strategy"] == strategy]
            feasible = [p for p in strategy_points if p["ndcg"] >= ndcg_floor]
            if not feasible:
                continue

            # Select best point with tie-breaker
            best_point = max(feasible, key=lambda p: (combined_score(p), p["ndcg"], -p["lambda"]))
            strategy_scores[strategy].append(combined_score(best_point))
            strategy_ilads[strategy].append(best_point["ilad"])
            strategy_ilmds[strategy].append(best_point["ilmd"])
            strategy_ndcg_ret[strategy].append(best_point["ndcg"] / baseline_ndcg)
            strategy_lambdas[strategy].append(best_point["lambda"])

    # Aggregate across datasets
    scorecard: dict[str, dict[str, float]] = {}
    for strategy in STRATEGIES:
        if strategy_scores[strategy]:
            scores = strategy_scores[strategy]
            ilads = strategy_ilads[strategy]
            ilmds = strategy_ilmds[strategy]
            ndcg_rets = strategy_ndcg_ret[strategy]
            lambdas = strategy_lambdas[strategy]
            scorecard[strategy] = {
                "combined_score": sum(scores) / len(scores),
                "ilad_at_best": sum(ilads) / len(ilads),
                "ilmd_at_best": sum(ilmds) / len(ilmds),
                "ndcg_retention": sum(ndcg_rets) / len(ndcg_rets),
                "typical_diversity": sum(lambdas) / len(lambdas),
            }

    return scorecard


def generate_pareto_plot(all_data: list[dict], output_path: Path, diversity_metric: str = "ilad") -> None:
    """Generate Pareto frontier plot showing relevance vs diversity tradeoff."""
    import matplotlib.pyplot as plt

    _setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    datasets = sorted(set(point["dataset"] for point in all_data))
    strategies = ["mmr", "msd", "dpp", "ssd"]
    colors = {"mmr": "#e74c3c", "msd": "#2ecc71", "dpp": "#3498db", "ssd": "#9b59b6"}

    name_map = {
        "ml-32m": "MovieLens-32M",
        "lastfm": "Last.FM",
        "amazon-video-games": "Amazon Video Games",
        "goodreads": "Goodreads",
    }

    metric_label = "ILAD (Avg Diversity)" if diversity_metric == "ilad" else "ILMD (Min Diversity)"

    markers = {"mmr": "o", "msd": "s", "dpp": "^", "ssd": "D"}

    for ax, dataset in zip(axes, datasets):
        for strategy in strategies:
            strategy_points = [p for p in all_data if p["dataset"] == dataset and p["strategy"] == strategy]
            strategy_points = sorted(strategy_points, key=lambda x: x["lambda"])

            if strategy_points:
                x_vals = [p[diversity_metric] for p in strategy_points]
                y_vals = [p["ndcg"] for p in strategy_points]
                ax.plot(
                    x_vals,
                    y_vals,
                    color=colors[strategy],
                    marker=markers[strategy],
                    label=strategy.upper(),
                    markersize=7,
                    linewidth=2.5,
                    alpha=0.8,
                )

        ax.set_xlabel(f"{metric_label} →", fontsize=10)
        ax.set_ylabel("nDCG@10 (Relevance) →", fontsize=10)
        ax.set_title(name_map.get(dataset, dataset), fontsize=12, fontweight="bold")
        # ILAD data is mostly on the right, ILMD data more spread out
        legend_loc = "upper left" if diversity_metric == "ilad" else "upper right"
        ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0.0, 1.05)

    if diversity_metric == "ilad":
        title = "Relevance vs Average Pairwise Diversity (nDCG@10 vs ILAD)"
    else:
        title = "Relevance vs Minimum Pairwise Diversity (nDCG@10 vs ILMD)"
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_latency_plot(output_path: Path) -> None:
    """Generate latency scaling plot using synthetic benchmark."""
    import matplotlib.pyplot as plt

    from benchmarks.core.latency import run_latency_benchmark

    _setup_plot_style()

    logger.info("Running synthetic latency benchmark...")
    results = run_latency_benchmark()

    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ["mmr", "msd", "dpp", "ssd"]
    colors = {"mmr": "#e74c3c", "msd": "#2ecc71", "dpp": "#3498db", "ssd": "#9b59b6"}
    markers = {"mmr": "o", "msd": "s", "dpp": "^", "ssd": "D"}

    for strategy in strategies:
        points = [r for r in results if r["strategy"] == strategy]
        points = sorted(points, key=lambda x: x["n_candidates"])

        x_vals = [p["n_candidates"] for p in points]
        y_vals = [p["latency_ms"] for p in points]

        ax.plot(
            x_vals,
            y_vals,
            color=colors[strategy],
            marker=markers[strategy],
            label=strategy.upper(),
            markersize=8,
            linewidth=2.5,
        )

    ax.set_xlabel("Number of Candidates (n)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Latency Scaling by Strategy (k=10, d=256)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _log_relevance_budgeted_analysis(all_data: list[dict]) -> None:
    """Log relevance-budgeted analysis tables."""
    logger.info(f"\n=== Relevance-Budgeted Analysis (≥{int(RELEVANCE_FLOOR*100)}% baseline nDCG) ===")

    # Per-dataset recommendations
    recommendations = _compute_relevance_budgeted_scores(all_data)

    name_map = {
        "ml-32m": "MovieLens-32M",
        "lastfm": "Last.FM",
        "amazon-video-games": "Amazon Video Games",
        "goodreads": "Goodreads",
    }

    logger.info("\nBest configs per dataset (maintaining ≥95% baseline relevance):")
    logger.info("| Dataset | Goal | Strategy | λ | nDCG vs Base | ILAD | ILMD |")
    logger.info("|---------|------|----------|---|--------------|------|------|")

    for dataset, goals in recommendations.items():
        display_name = name_map.get(dataset, dataset)[:16]
        for goal, config in goals.items():
            goal_display = {"max_ilad": "Max ILAD", "max_ilmd": "Max ILMD", "best_combined": "Best Overall"}[goal]
            strategy_name = str(config["strategy"]).upper()
            logger.info(
                f"| {display_name:16} | {goal_display:12} | {strategy_name:8} | "
                f"{config['lambda']:.1f} | {config['ndcg_vs_baseline']:.1%} | "
                f"{config['ilad']:.3f} | {config['ilmd']:.3f} |"
            )

    # Strategy scorecard
    scorecard = _compute_strategy_scorecard(all_data)
    logger.info("\n=== Strategy Scorecard (averaged across datasets) ===")
    logger.info("| Strategy | Combined | ILAD | ILMD | nDCG Ret | Typical diversity |")
    logger.info("|----------|----------|------|------|----------|-------------------|")

    sorted_strategies = sorted(scorecard.items(), key=lambda x: x[1]["combined_score"], reverse=True)
    for strategy, scores in sorted_strategies:
        logger.info(
            f"| {strategy.upper():8} | {scores['combined_score']:.3f}    | "
            f"{scores['ilad_at_best']:.2f} | {scores['ilmd_at_best']:.2f} | "
            f"{scores['ndcg_retention']:.1%}    | {scores['typical_diversity']:.1f}               |"
        )

    # Summary
    logger.info("\n=== Key Findings ===")
    if sorted_strategies:
        best_strategy = sorted_strategies[0][0].upper()
        logger.info(f"Best overall (balanced ILAD+ILMD): {best_strategy}")

    # Count wins per goal
    goal_wins: dict[str, dict[str, int]] = {"max_ilad": {}, "max_ilmd": {}, "best_combined": {}}
    for goals in recommendations.values():
        for goal, config in goals.items():
            strategy_str = str(config["strategy"])
            goal_wins[goal][strategy_str] = goal_wins[goal].get(strategy_str, 0) + 1

    for goal, wins in goal_wins.items():
        if wins:
            winner = max(wins.items(), key=lambda x: x[1])
            goal_display = {"max_ilad": "Max ILAD", "max_ilmd": "Max ILMD", "best_combined": "Best Overall"}[goal]
            logger.info(f"  {goal_display}: {winner[0].upper()} ({winner[1]}/{len(recommendations)} datasets)")


def generate_report(results_dir: Path) -> None:
    """Generate plots from JSON results. Main findings are in benchmarks/README.md."""
    results = []
    for json_path in results_dir.glob("*.json"):
        with open(json_path) as f:
            results.append(json.load(f))

    if not results:
        logger.warning("No results found.")
        return

    all_data: list[dict] = []
    for dataset_result in results:
        dataset = dataset_result["dataset"]
        for run in dataset_result["results"]:
            all_data.append(
                {
                    "dataset": dataset,
                    "strategy": run["strategy"],
                    "lambda": run["diversity"],
                    "ndcg": run["ndcg@10"],
                    "mrr": run["mrr"],
                    "ilad": run["ilad"],
                    "ilmd": run["ilmd"],
                }
            )

    # Generate ILAD plot (average diversity)
    pareto_ilad_path = results_dir / "pareto_ilad.png"
    generate_pareto_plot(all_data, pareto_ilad_path, diversity_metric="ilad")
    logger.debug(f"Saved: {pareto_ilad_path}")

    # Generate ILMD plot (minimum diversity)
    pareto_ilmd_path = results_dir / "pareto_ilmd.png"
    generate_pareto_plot(all_data, pareto_ilmd_path, diversity_metric="ilmd")
    logger.debug(f"Saved: {pareto_ilmd_path}")

    latency_path = results_dir / "latency.png"
    generate_latency_plot(latency_path)
    logger.debug(f"Saved: {latency_path}")

    # Relevance-budgeted analysis
    _log_relevance_budgeted_analysis(all_data)

    logger.info(f"\nReport generated: {results_dir}")

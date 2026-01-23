"""Plot generation from benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STRATEGIES = ["mmr", "msd", "dpp", "ssd"]

# Diversity regions for analysis (based on observed ranges across datasets)
# ILAD observed: 0.46-0.99, ILMD observed: 0.00-0.73
ILAD_REGIONS = {
    "low": (0.45, 0.60),
    "moderate": (0.60, 0.75),
    "high": (0.75, 1.0),
}
ILMD_REGIONS = {
    "low": (0.0, 0.20),
    "moderate": (0.20, 0.45),
    "high": (0.45, 0.75),
}


def _compute_regional_winners(
    all_data: list[dict],
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Count dataset wins for each strategy in each diversity region.

    A "win" means having the highest average nDCG for that region on a specific dataset.
    Returns dict mapping metric -> region -> strategy -> win_count.
    """
    results: dict[str, dict[str, dict[str, int]]] = {"ilad": {}, "ilmd": {}}
    datasets = sorted(set(p["dataset"] for p in all_data))

    for metric, regions in [("ilad", ILAD_REGIONS), ("ilmd", ILMD_REGIONS)]:
        for region_name, (low, high) in regions.items():
            win_counts: dict[str, int] = {s: 0 for s in STRATEGIES}

            for dataset in datasets:
                # Get points for this dataset in this region
                region_points = [p for p in all_data if p["dataset"] == dataset and low <= p[metric] < high]
                if not region_points:
                    continue

                # Compute avg nDCG per strategy for this dataset+region
                strategy_ndcg: dict[str, list[float]] = {s: [] for s in STRATEGIES}
                for point in region_points:
                    strategy_ndcg[point["strategy"]].append(point["ndcg"])

                # Find winner for this dataset
                best_strategy = None
                best_ndcg = -1.0
                for strategy, ndcgs in strategy_ndcg.items():
                    if ndcgs:
                        avg = sum(ndcgs) / len(ndcgs)
                        if avg > best_ndcg:
                            best_ndcg = avg
                            best_strategy = strategy

                if best_strategy:
                    win_counts[best_strategy] += 1

            results[metric][region_name] = win_counts

    return results


def generate_pareto_plot(all_data: list[dict], output_path: Path, diversity_metric: str = "ilad") -> None:
    """Generate Pareto frontier plot showing relevance vs diversity tradeoff."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    datasets = sorted(set(point["dataset"] for point in all_data))
    strategies = ["mmr", "msd", "dpp", "ssd"]
    colors = {"mmr": "#e74c3c", "msd": "#2ecc71", "dpp": "#3498db", "ssd": "#9b59b6"}

    name_map = {
        "ml-32m": "MovieLens-32M",
        "lastfm": "Last.FM",
        "amazon-product-reviews-video-games": "Amazon Video Games",
        "goodreads-rating": "Goodreads",
    }

    metric_label = "ILAD (Avg Diversity)" if diversity_metric == "ilad" else "ILMD (Min Diversity)"

    for ax, dataset in zip(axes, datasets):
        ax.axvspan(0.3, 0.5, alpha=0.1, color="blue", label="_Low")
        ax.axvspan(0.5, 0.7, alpha=0.1, color="green", label="_Moderate")
        ax.axvspan(0.7, 0.9, alpha=0.1, color="orange", label="_High")
        ax.axvspan(0.9, 1.0, alpha=0.1, color="red", label="_Max")

        for strategy in strategies:
            strategy_points = [p for p in all_data if p["dataset"] == dataset and p["strategy"] == strategy]
            strategy_points = sorted(strategy_points, key=lambda x: x["lambda"])

            if strategy_points:
                x_vals = [p[diversity_metric] for p in strategy_points]
                y_vals = [p["ndcg"] for p in strategy_points]
                ax.plot(
                    x_vals, y_vals, "o-", color=colors[strategy], label=strategy.upper(), markersize=7, linewidth=2.5
                )

        ax.set_xlabel(f"{metric_label} →", fontsize=10)
        ax.set_ylabel("nDCG@10 (Relevance) →", fontsize=10)
        ax.set_title(name_map.get(dataset, dataset), fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0.0, 1.05)

    first_ax = axes[0]
    y_top = first_ax.get_ylim()[1]
    first_ax.text(0.4, y_top * 0.95, "Low", ha="center", fontsize=8, color="blue", alpha=0.7)
    first_ax.text(0.6, y_top * 0.95, "Med", ha="center", fontsize=8, color="green", alpha=0.7)
    first_ax.text(0.8, y_top * 0.95, "High", ha="center", fontsize=8, color="orange", alpha=0.7)
    first_ax.text(0.95, y_top * 0.95, "Max", ha="center", fontsize=8, color="red", alpha=0.7)

    title_suffix = "(Average Diversity)" if diversity_metric == "ilad" else "(Minimum Diversity)"
    plt.suptitle(f"Relevance vs Diversity Tradeoff {title_suffix}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_latency_plot(output_path: Path) -> None:
    """Generate latency scaling plot using synthetic benchmark."""
    import matplotlib.pyplot as plt

    from benchmarks.core.latency import run_latency_benchmark

    logger.info("Running synthetic latency benchmark...")
    results = run_latency_benchmark()

    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ["mmr", "msd", "dpp", "ssd"]
    colors = {"mmr": "#e74c3c", "msd": "#2ecc71", "dpp": "#3498db", "ssd": "#9b59b6"}
    linestyles = {"mmr": "-", "msd": "--", "dpp": "-.", "ssd": "-"}
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
            linestyle=linestyles[strategy],
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


def _log_regional_analysis(regional: dict, num_datasets: int) -> None:
    """Log regional analysis tables."""
    logger.info(f"\nRegional Analysis (dataset wins out of {num_datasets} datasets):")
    logger.info("\nILAD (Average Diversity):")
    logger.info("| Region   | Range     | MMR | MSD | DPP | SSD | Best   |")
    logger.info("|----------|-----------|-----|-----|-----|-----|--------|")
    for region in ["low", "moderate", "high"]:
        if region in regional["ilad"]:
            wins = regional["ilad"][region]
            low, high = ILAD_REGIONS[region]
            best = max(wins.items(), key=lambda x: x[1])[0] if wins else "-"
            logger.info(
                f"| {region:8} | {low:.1f}-{high:.1f}   | "
                f"{wins.get('mmr', 0):3d} | {wins.get('msd', 0):3d} | "
                f"{wins.get('dpp', 0):3d} | {wins.get('ssd', 0):3d} | {best.upper():6} |"
            )

    logger.info("\nILMD (Minimum Diversity):")
    logger.info("| Region   | Range       | MMR | MSD | DPP | SSD | Best   |")
    logger.info("|----------|-------------|-----|-----|-----|-----|--------|")
    for region in ["low", "moderate", "high"]:
        if region in regional["ilmd"]:
            wins = regional["ilmd"][region]
            low, high = ILMD_REGIONS[region]
            best = max(wins.items(), key=lambda x: x[1])[0] if wins else "-"
            logger.info(
                f"| {region:8} | {low:.2f}-{high:.2f} | "
                f"{wins.get('mmr', 0):3d} | {wins.get('msd', 0):3d} | "
                f"{wins.get('dpp', 0):3d} | {wins.get('ssd', 0):3d} | {best.upper():6} |"
            )

    # Summary: total wins across all regions
    logger.info("\nTotal Wins Summary:")
    total_wins: dict[str, int] = {s: 0 for s in STRATEGIES}
    for metric in ["ilad", "ilmd"]:
        for region_wins in regional[metric].values():
            for strategy, count in region_wins.items():
                total_wins[strategy] += count

    sorted_strategies = sorted(total_wins.items(), key=lambda x: x[1], reverse=True)
    logger.info("| Strategy | Total Wins |")
    logger.info("|----------|------------|")
    for strategy, count in sorted_strategies:
        logger.info(f"| {strategy.upper():8} | {count:10d} |")


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

    # Regional analysis - count wins per dataset
    regional = _compute_regional_winners(all_data)
    num_datasets = len(set(p["dataset"] for p in all_data))
    _log_regional_analysis(regional, num_datasets)

    logger.info(f"\nReport generated: {results_dir}")

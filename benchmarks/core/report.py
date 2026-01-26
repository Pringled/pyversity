from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from benchmarks.core.latency import run_latency_benchmark

logger = logging.getLogger(__name__)

# Set up seaborn styling
sns.set_theme(style="whitegrid", palette="muted")


STRATEGIES = ["mmr", "msd", "dpp", "ssd"]

# Relevance floors: keep configs with at least this fraction of baseline nDCG
RELEVANCE_FLOORS = [0.99, 0.95]  # Strict and default


def _compute_relevance_budgeted_scores(
    all_data: list[dict],
    relevance_floor: float = 0.95,
) -> dict[str, dict[str, dict[str, float | str]]]:
    """
    Compute best configs per strategy under a relevance floor constraint.

    For each dataset, finds strategies that maximize ILAD, ILMD, and combined diversity
    while maintaining >= relevance_floor of baseline (lambda=0) nDCG.

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

        ndcg_floor = relevance_floor * baseline_ndcg

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
            """2-way score: geometric mean of ILAD and ILMD gains."""
            ilad_range = ilad_max - ilad_baseline
            ilmd_range = ilmd_max - ilmd_baseline
            if ilad_range == 0 or ilmd_range == 0:
                return 0.0
            # Clamp gains to [0, 1] to handle noise/non-monotonic behavior
            ilad_gain = max(0.0, min(1.0, (point["ilad"] - ilad_baseline) / ilad_range))
            ilmd_gain = max(0.0, min(1.0, (point["ilmd"] - ilmd_baseline) / ilmd_range))
            return (ilad_gain * ilmd_gain) ** 0.5  # Geometric mean

        def combined_score_3way(point: dict) -> float:
            """3-way score: includes nDCG gain (rewards relevance improvement)."""
            ilad_range = ilad_max - ilad_baseline
            ilmd_range = ilmd_max - ilmd_baseline
            if ilad_range == 0 or ilmd_range == 0:
                return 0.0
            ilad_gain = max(0.0, min(1.0, (point["ilad"] - ilad_baseline) / ilad_range))
            ilmd_gain = max(0.0, min(1.0, (point["ilmd"] - ilmd_baseline) / ilmd_range))
            # nDCG gain: ratio vs baseline, clamped to [0, 2] then normalized to [0, 1]
            ndcg_ratio = point["ndcg"] / baseline_ndcg if baseline_ndcg > 0 else 1.0
            ndcg_gain = max(0.0, min(1.0, (ndcg_ratio - 0.5) / 1.0))  # 0.5-1.5 -> 0-1
            return (ilad_gain * ilmd_gain * ndcg_gain) ** (1 / 3)  # Cubic root for 3-way

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

        # Goal 4: Best 3-way score (includes nDCG improvement)
        best_3way = max(feasible, key=lambda p: (combined_score_3way(p), p["ndcg"], -p["lambda"]))
        results[dataset]["best_3way"] = {
            "strategy": best_3way["strategy"],
            "lambda": best_3way["lambda"],
            "ndcg": best_3way["ndcg"],
            "ndcg_vs_baseline": best_3way["ndcg"] / baseline_ndcg,
            "ilad": best_3way["ilad"],
            "ilmd": best_3way["ilmd"],
            "score": combined_score_3way(best_3way),
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


def _aggregate_strategy_metrics(strategy_data: dict[str, list[float]]) -> dict[str, float]:
    """Aggregate per-dataset metrics into averages."""
    return {key: sum(values) / len(values) if values else 0.0 for key, values in strategy_data.items()}


def _compute_combined_score_2way(
    point: dict, ilad_baseline: float, ilad_range: float, ilmd_baseline: float, ilmd_range: float
) -> float:
    """Compute 2-way combined score (ILAD × ILMD)."""
    if ilad_range <= 0 or ilmd_range <= 0:
        return 0.0
    ilad_gain = max(0.0, min(1.0, (point["ilad"] - ilad_baseline) / ilad_range))
    ilmd_gain = max(0.0, min(1.0, (point["ilmd"] - ilmd_baseline) / ilmd_range))
    return (ilad_gain * ilmd_gain) ** 0.5


def _update_accumulator(
    acc: dict[str, list[float]],
    point: dict,
    baseline_ndcg: float,
    ilad_baseline: float,
    ilad_range: float,
    ilmd_baseline: float,
    ilmd_range: float,
) -> None:
    """Update accumulator with a point's metrics."""
    acc["scores"].append(_compute_combined_score_2way(point, ilad_baseline, ilad_range, ilmd_baseline, ilmd_range))
    acc["ilads"].append(point["ilad"])
    acc["ilmds"].append(point["ilmd"])
    acc["ndcg_ret"].append(point["ndcg"] / baseline_ndcg if baseline_ndcg > 0 else 0)
    acc["lambdas"].append(point["lambda"])
    if ilad_baseline > 0:
        acc["ilad_pct"].append((point["ilad"] - ilad_baseline) / ilad_baseline * 100)
    if ilmd_baseline > 0:
        acc["ilmd_pct"].append((point["ilmd"] - ilmd_baseline) / ilmd_baseline * 100)
    acc["ndcg_gain"].append((point["ndcg"] / baseline_ndcg - 1) * 100 if baseline_ndcg > 0 else 0)


def _get_bounds_for_dataset(
    dataset_points: list[dict], feasible_all: list[dict], relevance_floor: float | None, rank_by_ndcg: bool
) -> tuple[float, float, float, float] | None:
    """Get normalization bounds for a dataset (supports feasible-max normalization)."""
    bounds = _get_dataset_baseline_and_bounds(dataset_points)
    if bounds is None:
        return None

    baseline_ndcg, ilad_baseline, ilad_max_global, ilmd_baseline, ilmd_max_global = bounds

    # For diversity ranking with floor, use feasible-max normalization
    if relevance_floor is not None and not rank_by_ndcg and feasible_all:
        ilad_max = max(p["ilad"] for p in feasible_all)
        ilmd_max = max(p["ilmd"] for p in feasible_all)
    else:
        ilad_max = ilad_max_global
        ilmd_max = ilmd_max_global

    return ilad_baseline, ilad_max - ilad_baseline, ilmd_baseline, ilmd_max - ilmd_baseline


def _select_best_point(
    feasible: list[dict],
    rank_by_ndcg: bool,
    ilad_baseline: float,
    ilad_range: float,
    ilmd_baseline: float,
    ilmd_range: float,
) -> dict:
    """Select the best point based on ranking mode."""
    if rank_by_ndcg:
        return max(feasible, key=lambda p: (p["ndcg"], -p["lambda"]))
    else:
        return max(
            feasible,
            key=lambda p: (
                _compute_combined_score_2way(p, ilad_baseline, ilad_range, ilmd_baseline, ilmd_range),
                p["ndcg"],
                -p["lambda"],
            ),
        )


def _compute_strategy_scorecard(
    all_data: list[dict], relevance_floor: float | None = None, rank_by_ndcg: bool = False
) -> dict[str, dict[str, float]]:
    """Compute per-strategy aggregate scores across all datasets."""
    datasets = sorted(set(p["dataset"] for p in all_data))

    accumulators: dict[str, dict[str, list[float]]] = {
        s: {
            "scores": [],
            "ilads": [],
            "ilmds": [],
            "ilad_pct": [],
            "ilmd_pct": [],
            "ndcg_ret": [],
            "ndcg_gain": [],
            "lambdas": [],
        }
        for s in STRATEGIES
    }

    for dataset in datasets:
        dataset_points = [p for p in all_data if p["dataset"] == dataset]
        bounds = _get_dataset_baseline_and_bounds(dataset_points)
        if bounds is None:
            continue

        baseline_ndcg = bounds[0]
        ndcg_floor = relevance_floor * baseline_ndcg if relevance_floor else 0
        feasible_all = [p for p in dataset_points if p["ndcg"] >= ndcg_floor] if relevance_floor else dataset_points

        if not feasible_all:
            continue

        norm_bounds = _get_bounds_for_dataset(dataset_points, feasible_all, relevance_floor, rank_by_ndcg)
        if norm_bounds is None:
            continue

        ilad_baseline, ilad_range, ilmd_baseline, ilmd_range = norm_bounds

        for strategy in STRATEGIES:
            strategy_points = [p for p in dataset_points if p["strategy"] == strategy]
            feasible = [p for p in strategy_points if p["ndcg"] >= ndcg_floor] if relevance_floor else strategy_points

            if not feasible:
                continue

            best_point = _select_best_point(
                feasible, rank_by_ndcg, ilad_baseline, ilad_range, ilmd_baseline, ilmd_range
            )
            _update_accumulator(
                accumulators[strategy], best_point, baseline_ndcg, ilad_baseline, ilad_range, ilmd_baseline, ilmd_range
            )

    scorecard: dict[str, dict[str, float]] = {}
    for strategy, acc in accumulators.items():
        if acc["scores"]:
            scorecard[strategy] = {
                "diversity_score": sum(acc["scores"]) / len(acc["scores"]),
                "ilad_at_best": sum(acc["ilads"]) / len(acc["ilads"]),
                "ilmd_at_best": sum(acc["ilmds"]) / len(acc["ilmds"]),
                "ilad_pct_gain": sum(acc["ilad_pct"]) / len(acc["ilad_pct"]) if acc["ilad_pct"] else 0.0,
                "ilmd_pct_gain": sum(acc["ilmd_pct"]) / len(acc["ilmd_pct"]) if acc["ilmd_pct"] else 0.0,
                "ndcg_retention": sum(acc["ndcg_ret"]) / len(acc["ndcg_ret"]),
                "ndcg_pct_gain": sum(acc["ndcg_gain"]) / len(acc["ndcg_gain"]),
                "typical_diversity": sum(acc["lambdas"]) / len(acc["lambdas"]),
            }

    return scorecard


def _compute_sweet_spots(all_data: list[dict]) -> dict[str, dict[str, float]]:
    """
    Find the optimal operating point (sweet spot) for each strategy.

    The sweet spot maximizes combined diversity while nDCG is at or above baseline.
    This shows the "free lunch" region where you get diversity without losing relevance.
    """
    datasets = sorted(set(p["dataset"] for p in all_data))
    strategy_sweet_spots: dict[str, list[dict]] = {s: [] for s in STRATEGIES}

    for dataset in datasets:
        dataset_points = [p for p in all_data if p["dataset"] == dataset]
        bounds = _get_dataset_baseline_and_bounds(dataset_points)
        if bounds is None:
            continue

        baseline_ndcg, ilad_baseline, ilad_max, ilmd_baseline, ilmd_max = bounds
        ilad_range = ilad_max - ilad_baseline
        ilmd_range = ilmd_max - ilmd_baseline

        def combined_diversity(point: dict) -> float:
            ilad_gain = max(0.0, min(1.0, (point["ilad"] - ilad_baseline) / ilad_range))
            ilmd_gain = max(0.0, min(1.0, (point["ilmd"] - ilmd_baseline) / ilmd_range))
            return (ilad_gain * ilmd_gain) ** 0.5

        for strategy in STRATEGIES:
            strategy_points = [p for p in dataset_points if p["strategy"] == strategy]
            # Find points where nDCG >= baseline (no loss or gain)
            no_loss = [p for p in strategy_points if p["ndcg"] >= baseline_ndcg]

            if no_loss:
                # Find point with best diversity that doesn't lose relevance
                best = max(no_loss, key=lambda p: (combined_diversity(p), p["ndcg"], -p["lambda"]))
                strategy_sweet_spots[strategy].append(
                    {
                        "lambda": best["lambda"],
                        "ndcg_gain": (best["ndcg"] / baseline_ndcg - 1) * 100,
                        "ilad": best["ilad"],
                        "ilmd": best["ilmd"],
                        "ilad_gain": (best["ilad"] - ilad_baseline) / ilad_baseline * 100 if ilad_baseline > 0 else 0,
                        "ilmd_gain": (best["ilmd"] - ilmd_baseline) / ilmd_baseline * 100 if ilmd_baseline > 0 else 0,
                    }
                )

    # Aggregate across datasets
    sweet_spots: dict[str, dict[str, float]] = {}
    for strategy in STRATEGIES:
        spots = strategy_sweet_spots[strategy]
        if spots:
            sweet_spots[strategy] = {
                "typical_diversity": sum(s["lambda"] for s in spots) / len(spots),
                "avg_ndcg_gain": sum(s["ndcg_gain"] for s in spots) / len(spots),
                "avg_ilad": sum(s["ilad"] for s in spots) / len(spots),
                "avg_ilmd": sum(s["ilmd"] for s in spots) / len(spots),
                "avg_ilad_gain": sum(s["ilad_gain"] for s in spots) / len(spots),
                "avg_ilmd_gain": sum(s["ilmd_gain"] for s in spots) / len(spots),
                "datasets_with_sweet_spot": float(len(spots)),
            }

    return sweet_spots


def generate_pareto_plot(all_data: list[dict], output_path: Path, diversity_metric: str = "ilad") -> None:
    """Generate Pareto frontier plot showing relevance vs diversity tradeoff."""
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

    # Get best operating points for each dataset (use default 95% floor)
    budgeted_scores_95 = _compute_relevance_budgeted_scores(all_data, relevance_floor=0.95)
    budgeted_scores_99 = _compute_relevance_budgeted_scores(all_data, relevance_floor=0.99)

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

        # Add stars for best operating points at both floors
        goal_key = "max_ilad" if diversity_metric == "ilad" else "max_ilmd"

        # Filled star for 95% floor
        if dataset in budgeted_scores_95:
            best = budgeted_scores_95[dataset].get(goal_key)
            if best:
                x_star = best["ilad"] if diversity_metric == "ilad" else best["ilmd"]
                y_star = best["ndcg"]
                strategy = str(best["strategy"])
                ax.scatter(
                    [x_star],
                    [y_star],
                    marker="*",
                    s=200,
                    c=colors[strategy],
                    edgecolors="black",
                    linewidths=1,
                    zorder=10,
                )

        # Diamond for 99% floor
        if dataset in budgeted_scores_99:
            best = budgeted_scores_99[dataset].get(goal_key)
            if best:
                x_diamond = best["ilad"] if diversity_metric == "ilad" else best["ilmd"]
                y_diamond = best["ndcg"]
                strategy = str(best["strategy"])
                ax.scatter(
                    [x_diamond],
                    [y_diamond],
                    marker="D",
                    s=100,
                    c=colors[strategy],
                    edgecolors="black",
                    linewidths=1,
                    zorder=10,
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
            alpha=0.8,
        )

    ax.set_xlabel("Number of Candidates (n)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Latency Scaling by Strategy (k=10, d=256)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _log_per_strategy_diversity_tables(all_data: list[dict]) -> None:
    """Log per-strategy tables showing metrics across diversity values (averaged over datasets)."""
    datasets = sorted(set(p["dataset"] for p in all_data))
    lambda_values = sorted(set(p["lambda"] for p in all_data))

    logger.info("\n=== Per-Strategy Diversity Sweep (Averaged Across Datasets) ===")

    for strategy in STRATEGIES:
        logger.info(f"\n#### {strategy.upper()}")
        logger.info("")
        logger.info("| `diversity` | nDCG Retention | ILAD | ILMD |")
        logger.info("|:-----------:|:--------------:|:----:|:----:|")

        for lam in lambda_values:
            ndcg_rets = []
            ilads = []
            ilmds = []
            ilad_pct_gains = []
            ilmd_pct_gains = []

            for dataset in datasets:
                dataset_points = [p for p in all_data if p["dataset"] == dataset]
                bounds = _get_dataset_baseline_and_bounds(dataset_points)
                if bounds is None:
                    continue

                baseline_ndcg, ilad_baseline, _, ilmd_baseline, _ = bounds

                # Find point for this strategy/lambda
                point = next(
                    (p for p in dataset_points if p["strategy"] == strategy and p["lambda"] == lam),
                    None,
                )
                if point is None:
                    continue

                ndcg_rets.append(point["ndcg"] / baseline_ndcg if baseline_ndcg > 0 else 0)
                ilads.append(point["ilad"])
                ilmds.append(point["ilmd"])

                if ilad_baseline > 0:
                    ilad_pct_gains.append((point["ilad"] - ilad_baseline) / ilad_baseline * 100)
                if ilmd_baseline > 0:
                    ilmd_pct_gains.append((point["ilmd"] - ilmd_baseline) / ilmd_baseline * 100)

            if ndcg_rets:
                avg_ndcg_ret = sum(ndcg_rets) / len(ndcg_rets)
                avg_ilad = sum(ilads) / len(ilads)
                avg_ilmd = sum(ilmds) / len(ilmds)
                avg_ilad_gain = sum(ilad_pct_gains) / len(ilad_pct_gains) if ilad_pct_gains else 0
                avg_ilmd_gain = sum(ilmd_pct_gains) / len(ilmd_pct_gains) if ilmd_pct_gains else 0

                logger.info(
                    f"| {lam:.1f}         | {avg_ndcg_ret:.1%}          | "
                    f"{avg_ilad:.2f} (+{avg_ilad_gain:.0f}%) | "
                    f"{avg_ilmd:.2f} (+{avg_ilmd_gain:.0f}%) |"
                )


def _log_per_dataset_configs(recommendations: dict, floor_pct: int) -> None:
    """Log per-dataset config table."""
    name_map = {
        "ml-32m": "MovieLens-32M",
        "lastfm": "Last.FM",
        "amazon-video-games": "Amazon Video Games",
        "goodreads": "Goodreads",
    }

    logger.info(f"\nBest configs per dataset (maintaining ≥{floor_pct}% baseline relevance):")
    logger.info("| Dataset | Goal | Strategy | λ | nDCG vs Base | ILAD | ILMD |")
    logger.info("|---------|------|----------|---|--------------|------|------|")

    for dataset, goals in recommendations.items():
        display_name = name_map.get(dataset, dataset)[:16]
        for goal, config in goals.items():
            if goal == "best_3way":
                continue
            goal_display = {"max_ilad": "Max ILAD", "max_ilmd": "Max ILMD", "best_combined": "Best Overall"}[goal]
            strategy_name = str(config["strategy"]).upper()
            logger.info(
                f"| {display_name:16} | {goal_display:12} | {strategy_name:8} | "
                f"{config['lambda']:.1f} | {config['ndcg_vs_baseline']:.1%} | "
                f"{config['ilad']:.3f} | {config['ilmd']:.3f} |"
            )


def _log_scorecard_table(
    scorecard: dict[str, dict[str, float]], title: str, rank_by_ndcg: bool = False
) -> list[tuple[str, dict]]:
    """Log scorecard table and return sorted strategies."""
    logger.info(f"\n=== {title} ===")

    if rank_by_ndcg:
        logger.info("| Strategy | nDCG Δ | ILAD (+%) | ILMD (+%) | `diversity` |")
        logger.info("|----------|:------:|:---------:|:---------:|:-----------:|")
        sort_key = lambda x: x[1]["ndcg_pct_gain"]
    else:
        logger.info("| Strategy | Diversity Score | nDCG Δ | ILAD (+%) | ILMD (+%) | `diversity` |")
        logger.info("|----------|:---------------:|:------:|:---------:|:---------:|:-----------:|")
        sort_key = lambda x: x[1]["diversity_score"]

    sorted_strategies = sorted(scorecard.items(), key=sort_key, reverse=True)

    for strategy, scores in sorted_strategies:
        ndcg_delta = scores["ndcg_pct_gain"]
        ndcg_str = f"+{ndcg_delta:.1f}%" if ndcg_delta >= 0 else f"{ndcg_delta:.1f}%"

        if rank_by_ndcg:
            logger.info(
                f"| **{strategy.upper()}** | {ndcg_str} | "
                f"{scores['ilad_at_best']:.2f} (+{scores['ilad_pct_gain']:.0f}%) | "
                f"{scores['ilmd_at_best']:.2f} (+{scores['ilmd_pct_gain']:.0f}%) | "
                f"{scores['typical_diversity']:.1f} |"
            )
        else:
            logger.info(
                f"| **{strategy.upper()}** | {scores['diversity_score']:.3f} | "
                f"{ndcg_str} | "
                f"{scores['ilad_at_best']:.2f} (+{scores['ilad_pct_gain']:.0f}%) | "
                f"{scores['ilmd_at_best']:.2f} (+{scores['ilmd_pct_gain']:.0f}%) | "
                f"{scores['typical_diversity']:.1f} |"
            )

    return sorted_strategies


def _log_per_floor_analysis(all_data: list[dict], floor: float) -> None:
    """Log analysis for a single relevance floor (diversity leaderboard)."""
    floor_pct = int(floor * 100)

    # Diversity leaderboard: maximize diversity under floor constraint
    scorecard = _compute_strategy_scorecard(all_data, relevance_floor=floor, rank_by_ndcg=False)
    title = f"Diversity Leaderboard (≥{floor_pct}% baseline nDCG)"
    sorted_strategies = _log_scorecard_table(scorecard, title, rank_by_ndcg=False)

    if sorted_strategies:
        best_strategy = sorted_strategies[0][0].upper()
        best_scores = sorted_strategies[0][1]
        logger.info(f"\nBest diversity under {floor_pct}% floor: {best_strategy}")
        logger.info(f"  Diversity score: {best_scores['diversity_score']:.3f}")
        logger.info(f"  nDCG: {'+' if best_scores['ndcg_pct_gain'] >= 0 else ''}{best_scores['ndcg_pct_gain']:.1f}%")
        logger.info(f"  ILAD: +{best_scores['ilad_pct_gain']:.0f}%")
        logger.info(f"  ILMD: +{best_scores['ilmd_pct_gain']:.0f}%")


def _log_relevance_budgeted_analysis(all_data: list[dict]) -> None:
    """Log the three main tables: accuracy leaderboard + two diversity leaderboards."""
    # Table 1: Accuracy leaderboard (best nDCG, no constraints)
    logger.info("\n" + "=" * 60)
    logger.info("TABLE 1: ACCURACY LEADERBOARD")
    logger.info("Best nDCG per strategy (selecting λ that maximizes nDCG)")
    logger.info("=" * 60)

    accuracy_scorecard = _compute_strategy_scorecard(all_data, relevance_floor=None, rank_by_ndcg=True)
    _log_scorecard_table(accuracy_scorecard, "Best Relevance (Accuracy)", rank_by_ndcg=True)

    # Table 2 & 3: Diversity leaderboards under floor constraints
    for floor in RELEVANCE_FLOORS:
        floor_pct = int(floor * 100)
        logger.info("\n" + "=" * 60)
        logger.info(f"TABLE {RELEVANCE_FLOORS.index(floor) + 2}: DIVERSITY LEADERBOARD (≥{floor_pct}% nDCG)")
        logger.info(f"Best diversity per strategy under ≥{floor_pct}% baseline relevance")
        logger.info("Ranked by geometric mean of normalized ILAD/ILMD gains")
        logger.info("Uses feasible-max normalization for fair comparison")
        logger.info("=" * 60)

        _log_per_floor_analysis(all_data, floor)

    # Per-strategy diversity sweep tables (for detailed results)
    _log_per_strategy_diversity_tables(all_data)


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

    # Per-strategy diversity sweep tables
    _log_per_strategy_diversity_tables(all_data)

    logger.info(f"\nReport generated: {results_dir}")

"""Markdown report and plot generation from benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STRATEGIES = ["mmr", "msd", "dpp", "ssd"]


def _compute_retention_table(results: list[dict], retention_pct: float = 0.95) -> dict:
    """Compute diversity achieved at given relevance retention threshold."""
    table: dict[str, dict[str, dict[str, float]]] = {}

    for dataset_result in results:
        dataset = dataset_result["dataset"]
        table[dataset] = {}

        for strategy in STRATEGIES:
            runs = [run for run in dataset_result["results"] if run["strategy"] == strategy]
            baseline = next((run for run in runs if run["diversity"] == 0.0), None)

            if not baseline:
                continue

            threshold = baseline["ndcg@10"] * retention_pct
            valid = [run for run in runs if run["ndcg@10"] >= threshold and run["diversity"] > 0]

            if valid:
                best = max(valid, key=lambda run: run["ilad"])
                table[dataset][strategy] = {
                    "ilad": best["ilad"],
                    "lambda": best["diversity"],
                    "ndcg": best["ndcg@10"],
                    "retention": best["ndcg@10"] / baseline["ndcg@10"],
                }

    return table


def _extract_all_points(results: list[dict]) -> list[dict]:
    """Extract all data points from results into flat list."""
    all_points = []
    for dataset_result in results:
        dataset = dataset_result["dataset"]
        for run in dataset_result["results"]:
            all_points.append(
                {
                    "dataset": dataset,
                    "strategy": run["strategy"],
                    "ndcg": run["ndcg@10"],
                    "ilad": run["ilad"],
                }
            )
    return all_points


def _find_regional_winners(
    datasets: list[str],
    all_points: list[dict],
    regions: dict[str, tuple[float, float]],
) -> tuple[dict, dict]:
    """Find winner per region per dataset and aggregate wins."""
    regional_data: dict[str, dict[str, dict[str, str | float]]] = {}
    region_wins: dict[str, dict[str, int]] = {region: {} for region in regions}

    for dataset in datasets:
        dataset_points = [p for p in all_points if p["dataset"] == dataset]
        regional_data[dataset] = {}

        for region_name, (lo, hi) in regions.items():
            in_region = [p for p in dataset_points if lo <= p["ilad"] < hi]
            if in_region:
                best = max(in_region, key=lambda x: x["ndcg"])
                regional_data[dataset][region_name] = {
                    "winner": best["strategy"],
                    "ndcg": best["ndcg"],
                    "ilad": best["ilad"],
                }
                winner = str(best["strategy"])
                region_wins[region_name][winner] = region_wins[region_name].get(winner, 0) + 1

    return regional_data, region_wins


def _compute_region_averages(
    datasets: list[str],
    all_points: list[dict],
    regions: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Compute average best nDCG per strategy per region across datasets."""
    all_strategies = {"ssd", "dpp", "msd", "mmr"}
    region_best_per_dataset: dict[str, dict[str, dict[str, float]]] = {r: {} for r in regions}

    for dataset in datasets:
        dataset_points = [p for p in all_points if p["dataset"] == dataset]
        for region_name, (lo, hi) in regions.items():
            in_region = [p for p in dataset_points if lo <= p["ilad"] < hi]
            strategy_best: dict[str, float] = {}
            for point in in_region:
                s = point["strategy"]
                if s not in strategy_best or point["ndcg"] > strategy_best[s]:
                    strategy_best[s] = point["ndcg"]
            if strategy_best:
                region_best_per_dataset[region_name][dataset] = strategy_best

    region_avg: dict[str, dict[str, float]] = {}
    for region, dataset_bests in region_best_per_dataset.items():
        valid = [d for d in dataset_bests.values() if set(d.keys()) == all_strategies]
        if not valid:
            valid = list(dataset_bests.values())

        strategy_scores: dict[str, list[float]] = {}
        for ds_bests in valid:
            for s, score in ds_bests.items():
                strategy_scores.setdefault(s, []).append(score)
        region_avg[region] = {s: sum(v) / len(v) for s, v in strategy_scores.items() if v}

    return region_avg


def _compute_regional_winners(results: list[dict]) -> dict:
    """Compute which strategy achieves best nDCG in each diversity region."""
    regions = {
        "Low (0.3-0.5)": (0.3, 0.5),
        "Moderate (0.5-0.7)": (0.5, 0.7),
        "High (0.7-0.9)": (0.7, 0.9),
        "Maximum (0.9+)": (0.9, 1.0),
    }

    all_points = _extract_all_points(results)
    datasets = sorted(set(p["dataset"] for p in all_points))

    regional_data, region_wins = _find_regional_winners(datasets, all_points, regions)
    region_avg = _compute_region_averages(datasets, all_points, regions)

    return {"per_dataset": regional_data, "aggregate": region_wins, "avg_ndcg": region_avg}


def _format_retention_table(retention_table: dict) -> list[str]:
    """Format the retention table as markdown lines."""
    lines = [
        "## Diversity at 95% Relevance Retention\n",
        "How much diversity (ILAD) can each strategy achieve while maintaining ≥95% of baseline nDCG?\n",
        "| Dataset | MMR | MSD | DPP | SSD | Winner |",
        "|---------|:---:|:---:|:---:|:---:|:------:|",
    ]

    for dataset in sorted(retention_table.keys()):
        row = [dataset]
        best_ilad = 0
        winner = ""

        for strategy in STRATEGIES:
            if strategy in retention_table[dataset]:
                ilad = retention_table[dataset][strategy]["ilad"]
                row.append(f"{ilad:.3f}")
                if ilad > best_ilad:
                    best_ilad = ilad
                    winner = strategy.upper()
            else:
                row.append("-")

        row.append(f"**{winner}**")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def _format_regional_analysis(regional_data: dict) -> list[str]:
    """Format the regional winner analysis as markdown."""
    lines = [
        "## Best Strategy by Diversity Level\n",
        "Which strategy achieves highest relevance (nDCG) at each diversity (ILAD) level?\n",
        "| Dataset | Low (0.3-0.5) | Moderate (0.5-0.7) | High (0.7-0.9) | Maximum (0.9+) |",
        "|---------|:-------------:|:------------------:|:--------------:|:--------------:|",
    ]

    region_order = ["Low (0.3-0.5)", "Moderate (0.5-0.7)", "High (0.7-0.9)", "Maximum (0.9+)"]

    for dataset in sorted(regional_data["per_dataset"].keys()):
        row = [dataset]
        for region in region_order:
            if region in regional_data["per_dataset"][dataset]:
                winner = regional_data["per_dataset"][dataset][region]["winner"].upper()
                row.append(f"**{winner}**")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    # Add summary using best nDCG per region (averaged across datasets)
    lines.append("")
    lines.append("### Summary: Best nDCG per Strategy by Region\n")
    lines.append("For each region, what's the best nDCG each strategy can achieve (averaged across datasets)?\n")
    lines.append("| Region | SSD | DPP | MSD | MMR | Best |")
    lines.append("|--------|:---:|:---:|:---:|:---:|:----:|")

    avg_ndcg = regional_data.get("avg_ndcg", {})

    for region in region_order:
        region_avgs = avg_ndcg.get(region, {})
        row = [region]
        best_strategy = ""
        best_val = -1.0
        for s in ["ssd", "dpp", "msd", "mmr"]:
            val = region_avgs.get(s, 0)
            if val > 0:
                row.append(f"{val:.3f}")
                if val > best_val:
                    best_val = val
                    best_strategy = s.upper()
            else:
                row.append("-")
        row.append(f"**{best_strategy}**" if best_strategy else "-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def _trapezoidal_area(points: list[dict]) -> float:
    """Compute area under curve using trapezoidal rule."""
    if len(points) < 2:
        return 0.0
    sorted_pts = sorted(points, key=lambda p: p["ilad"])
    area = 0.0
    for i in range(len(sorted_pts) - 1):
        dx = sorted_pts[i + 1]["ilad"] - sorted_pts[i]["ilad"]
        avg_y = (sorted_pts[i]["ndcg"] + sorted_pts[i + 1]["ndcg"]) / 2
        area += dx * avg_y
    return area


def _find_overlap_range(strategy_points: dict[str, list[dict]]) -> tuple[float, float]:
    """Find ILAD range where all strategies have data points."""
    max_mins = []
    min_maxs = []
    for points in strategy_points.values():
        if points:
            ilads = [p["ilad"] for p in points]
            max_mins.append(min(ilads))
            min_maxs.append(max(ilads))

    if max_mins and min_maxs:
        return max(max_mins), min(min_maxs)
    return 0.0, 1.0


def _compute_pareto_area(results: list[dict]) -> dict:
    """
    Compute area under each strategy's Pareto curve per dataset.

    Uses trapezoidal integration, sorted by ILAD.
    Higher area = better overall relevance-diversity tradeoff.
    """
    strategies = ["mmr", "msd", "dpp", "ssd"]

    # Collect points per dataset per strategy
    dataset_strategy_points: dict[str, dict[str, list[dict]]] = {}
    for dataset_result in results:
        dataset = dataset_result["dataset"]
        dataset_strategy_points[dataset] = {s: [] for s in strategies}
        for run in dataset_result["results"]:
            dataset_strategy_points[dataset][run["strategy"]].append({"ndcg": run["ndcg@10"], "ilad": run["ilad"]})

    full_areas: dict[str, dict[str, float]] = {s: {} for s in strategies}
    overlap_areas: dict[str, dict[str, float]] = {s: {} for s in strategies}

    for dataset, strategy_points in dataset_strategy_points.items():
        overlap_lo, overlap_hi = _find_overlap_range(strategy_points)
        for s, points in strategy_points.items():
            full_areas[s][dataset] = _trapezoidal_area(points)
            overlap_pts = [p for p in points if overlap_lo <= p["ilad"] <= overlap_hi]
            overlap_areas[s][dataset] = _trapezoidal_area(overlap_pts)

    # Average across datasets
    avg_full = {s: sum(full_areas[s].values()) / len(full_areas[s]) if full_areas[s] else 0.0 for s in strategies}
    avg_overlap = {
        s: sum(overlap_areas[s].values()) / len(overlap_areas[s]) if overlap_areas[s] else 0.0 for s in strategies
    }

    return {
        "full_per_dataset": full_areas,
        "overlap_per_dataset": overlap_areas,
        "avg_full": avg_full,
        "avg_overlap": avg_overlap,
    }


def _compute_pareto_dominance(results: list[dict]) -> dict:
    """
    Compute Pareto frontier dominance for each strategy.

    A point is on the Pareto frontier if no other point has both higher nDCG
    and higher ILAD. Returns count of frontier points per strategy per dataset.
    """
    strategies = ["mmr", "msd", "dpp", "ssd"]

    # Collect all points per dataset
    dataset_points: dict[str, list[dict]] = {}
    for dataset_result in results:
        dataset = dataset_result["dataset"]
        dataset_points[dataset] = []
        for run in dataset_result["results"]:
            dataset_points[dataset].append(
                {
                    "strategy": run["strategy"],
                    "ndcg": run["ndcg@10"],
                    "ilad": run["ilad"],
                }
            )

    # Compute Pareto frontier for each dataset
    frontier_counts: dict[str, dict[str, int]] = {s: {} for s in strategies}
    total_frontier: dict[str, int] = {s: 0 for s in strategies}

    for dataset, points in dataset_points.items():
        for point in points:
            is_dominated = False
            for other in points:
                if other is point:
                    continue
                # Check if 'other' dominates 'point' (better or equal on both, strictly better on at least one)
                if other["ndcg"] >= point["ndcg"] and other["ilad"] >= point["ilad"]:
                    if other["ndcg"] > point["ndcg"] or other["ilad"] > point["ilad"]:
                        is_dominated = True
                        break

            if not is_dominated:
                strategy = point["strategy"]
                frontier_counts[strategy][dataset] = frontier_counts[strategy].get(dataset, 0) + 1
                total_frontier[strategy] += 1

    # Compute percentage of all frontier points
    total_points = sum(total_frontier.values())
    frontier_pct: dict[str, float] = {
        s: (count / total_points * 100) if total_points > 0 else 0.0 for s, count in total_frontier.items()
    }

    return {
        "per_dataset": frontier_counts,
        "total": total_frontier,
        "percentage": frontier_pct,
    }


def _compute_multi_metric_summary(results: list[dict]) -> dict:
    """
    Compute Pareto area for multiple metric combinations.

    Analyzes: nDCG vs ILAD, MRR vs ILAD, nDCG vs ILMD, MRR vs ILMD
    """
    strategies = ["mmr", "msd", "dpp", "ssd"]
    metric_pairs = [
        ("ndcg@10", "ilad", "nDCG vs ILAD"),
        ("mrr", "ilad", "MRR vs ILAD"),
        ("ndcg@10", "ilmd", "nDCG vs ILMD"),
        ("mrr", "ilmd", "MRR vs ILMD"),
    ]

    def _compute_area(points: list[dict], x_key: str, y_key: str) -> float:
        """Compute area under curve using trapezoidal rule."""
        if len(points) < 2:
            return 0.0
        sorted_pts = sorted(points, key=lambda p: p[x_key])
        area = 0.0
        for i in range(len(sorted_pts) - 1):
            dx = sorted_pts[i + 1][x_key] - sorted_pts[i][x_key]
            avg_y = (sorted_pts[i][y_key] + sorted_pts[i + 1][y_key]) / 2
            area += dx * avg_y
        return area

    summary: dict[str, dict[str, float]] = {label: {} for _, _, label in metric_pairs}

    for rel_key, div_key, label in metric_pairs:
        strategy_areas: dict[str, list[float]] = {s: [] for s in strategies}

        for dataset_result in results:
            for s in strategies:
                points = [
                    {rel_key: run[rel_key], div_key: run[div_key]}
                    for run in dataset_result["results"]
                    if run["strategy"] == s
                ]
                area = _compute_area(points, div_key, rel_key)
                strategy_areas[s].append(area)

        # Average across datasets
        for s in strategies:
            summary[label][s] = sum(strategy_areas[s]) / len(strategy_areas[s]) if strategy_areas[s] else 0.0

    return summary


def _format_multi_metric_table(multi_metric: dict) -> list[str]:
    """Format multi-metric summary as markdown table."""
    lines = [
        "### Multi-Metric Comparison\n",
        "Area under curve for different relevance-diversity metric combinations:\n",
        "| Metrics | SSD | DPP | MSD | MMR | Best |",
        "|---------|:---:|:---:|:---:|:---:|:----:|",
    ]

    for label, areas in multi_metric.items():
        row = [label]
        best_s = ""
        best_val = -1.0
        for s in ["ssd", "dpp", "msd", "mmr"]:
            val = areas.get(s, 0)
            row.append(f"{val:.4f}")
            if val > best_val:
                best_val = val
                best_s = s.upper()
        row.append(f"**{best_s}**")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("*Higher area = better overall tradeoff between that relevance and diversity metric.*")
    lines.append("")
    return lines


def _format_pareto_summary(pareto_data: dict, area_data: dict) -> list[str]:
    """Format Pareto dominance and area summary as markdown."""
    lines = [
        "## Overall Performance Summary\n",
        "### Pareto Frontier Analysis\n",
        "Points on the Pareto frontier represent optimal relevance-diversity tradeoffs.\n",
        "| Strategy | Frontier Points | Share | Avg Area (Full) | Avg Area (Overlap) |",
        "|----------|----------------:|------:|----------------:|-------------------:|",
    ]

    total = pareto_data["total"]
    pct = pareto_data["percentage"]
    avg_full = area_data["avg_full"]
    avg_overlap = area_data["avg_overlap"]

    # Sort by overlap area descending (fairest comparison)
    for strategy in sorted(total.keys(), key=lambda s: avg_overlap.get(s, 0), reverse=True):
        lines.append(
            f"| **{strategy.upper()}** | {total[strategy]} | {pct[strategy]:.1f}% | "
            f"{avg_full.get(strategy, 0):.4f} | {avg_overlap.get(strategy, 0):.4f} |"
        )

    lines.extend(
        [
            "",
            "*Full Area*: Area under each strategy's curve across all ILAD values it can reach.",
            "*Overlap Area*: Area only in the ILAD range where all strategies compete (fairest comparison).",
            "",
        ]
    )
    return lines


def _format_full_results(results: list[dict]) -> list[str]:
    """Format full results as collapsible markdown."""
    lines = [
        "<details>",
        "<summary>Full Results by Dataset (click to expand)</summary>\n",
    ]

    for dataset_result in sorted(results, key=lambda x: x["dataset"]):
        dataset = dataset_result["dataset"]
        lines.append(f"### {dataset}\n")
        lines.append("| Strategy | λ | nDCG@10 | MRR | ILAD | ILMD |")
        lines.append("|----------|--:|--------:|----:|-----:|-----:|")

        for run in sorted(dataset_result["results"], key=lambda x: (x["strategy"], x["diversity"])):
            lines.append(
                f"| {run['strategy'].upper()} | {run['diversity']:.1f} | "
                f"{run['ndcg@10']:.4f} | {run['mrr']:.4f} | {run['ilad']:.3f} | {run['ilmd']:.3f} |"
            )
        lines.append("")

    lines.append("</details>\n")
    return lines


def generate_markdown(results: list[dict]) -> str:
    """Generate markdown report content from benchmark results."""
    retention_table = _compute_retention_table(results, retention_pct=0.95)
    regional_data = _compute_regional_winners(results)

    md_lines = [
        "# Pyversity Benchmark Results\n",
        "Comparison of diversification strategies across recommendation datasets.\n",
        "## Datasets\n",
        "| Dataset | Source | Description |",
        "|---------|--------|-------------|",
        "| MovieLens-32M | GroupLens | 32M ratings from MovieLens |",
        "| Last.FM | HetRec 2011 | Music listening data |",
        "| Amazon Video Games | McAuley Lab | Product reviews |",
        "| Goodreads | McAuley Lab | Book ratings |",
        "",
        "## Strategies\n",
        "| Strategy | Description | Complexity |",
        "|----------|-------------|------------|",
        "| **MMR** | Maximal Marginal Relevance | O(k·n) |",
        "| **MSD** | Max-Sum Diversification | O(k·n) |",
        "| **DPP** | Determinantal Point Process | O(k²·n) |",
        "| **SSD** | Sliding Spectrum Decomposition | O(k²·n) |",
        "",
    ]

    pareto_data = _compute_pareto_dominance(results)
    area_data = _compute_pareto_area(results)
    multi_metric = _compute_multi_metric_summary(results)
    md_lines.extend(_format_pareto_summary(pareto_data, area_data))
    md_lines.extend(_format_multi_metric_table(multi_metric))
    md_lines.extend(_format_regional_analysis(regional_data))
    md_lines.extend(_format_retention_table(retention_table))
    md_lines.extend(_format_full_results(results))

    return "\n".join(md_lines)


def generate_pareto_plot(all_data: list[dict], output_path: Path, diversity_metric: str = "ilad") -> None:
    """
    Generate Pareto frontier plot showing relevance vs diversity tradeoff.

    Args:
    ----
        all_data: List of benchmark results.
        output_path: Path to save the plot.
        diversity_metric: 'ilad' for average diversity, 'ilmd' for minimum diversity.

    """
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


def generate_report(results_dir: Path) -> None:
    """Generate markdown report and plots from JSON results."""
    results = []
    for json_path in results_dir.glob("*.json"):
        with open(json_path) as f:
            results.append(json.load(f))

    if not results:
        logger.warning("No results found.")
        return

    all_data = []
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

    md_content = generate_markdown(results)
    report_path = results_dir / "RESULTS.md"
    report_path.write_text(md_content)
    logger.debug(f"Saved: {report_path}")

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

    logger.info(f"Report generated: {results_dir}")

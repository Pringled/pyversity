"""Report generation for benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STRATEGIES = ["mmr", "msd", "dpp", "ssd"]


def _compute_retention_table(results: list[dict], retention_pct: float = 0.95) -> dict:
    """Compute diversity achieved at given relevance retention threshold."""
    table: dict[str, dict] = {}

    for dataset_result in results:
        dataset = dataset_result["dataset"]
        table[dataset] = {}

        for strategy in STRATEGIES:
            runs = [r for r in dataset_result["results"] if r["strategy"] == strategy]
            baseline = next((r for r in runs if r["diversity"] == 0.0), None)

            if not baseline:
                continue

            threshold = baseline["ndcg@10"] * retention_pct
            valid = [r for r in runs if r["ndcg@10"] >= threshold and r["diversity"] > 0]

            if valid:
                best = max(valid, key=lambda r: r["ilad"])
                table[dataset][strategy] = {
                    "ilad": best["ilad"],
                    "lambda": best["diversity"],
                    "ndcg": best["ndcg@10"],
                    "retention": best["ndcg@10"] / baseline["ndcg@10"],
                }

    return table


def _compute_regional_winners(results: list[dict]) -> dict:
    """Compute which strategy wins in each diversity region."""
    # Define diversity regions
    regions = {
        "Low (0.3-0.5)": (0.3, 0.5),
        "Moderate (0.5-0.7)": (0.5, 0.7),
        "High (0.7-0.9)": (0.7, 0.9),
        "Maximum (0.9+)": (0.9, 1.0),
    }

    # Collect all points
    all_points = []
    for dataset_result in results:
        dataset = dataset_result["dataset"]
        for r in dataset_result["results"]:
            all_points.append(
                {
                    "dataset": dataset,
                    "strategy": r["strategy"],
                    "ndcg": r["ndcg@10"],
                    "ilad": r["ilad"],
                }
            )

    # Find winner for each dataset/region
    regional_data: dict[str, dict] = {}
    datasets = sorted(set(p["dataset"] for p in all_points))

    for dataset in datasets:
        ds_points = [p for p in all_points if p["dataset"] == dataset]
        regional_data[dataset] = {}

        for region_name, (lo, hi) in regions.items():
            in_region = [p for p in ds_points if lo <= p["ilad"] < hi]
            if in_region:
                best = max(in_region, key=lambda x: x["ndcg"])
                regional_data[dataset][region_name] = {
                    "winner": best["strategy"],
                    "ndcg": best["ndcg"],
                    "ilad": best["ilad"],
                }

    # Aggregate wins per region
    region_wins: dict[str, dict[str, int]] = {r: {} for r in regions}
    for dataset, regions_dict in regional_data.items():
        for region_name, data in regions_dict.items():
            winner = data["winner"]
            region_wins[region_name][winner] = region_wins[region_name].get(winner, 0) + 1

    return {"per_dataset": regional_data, "aggregate": region_wins}


def _compute_f1_scores(results: list[dict]) -> tuple[dict, dict]:
    """Compute normalized F1 scores (harmonic mean of nDCG and ILAD)."""
    best_per_dataset: dict[str, dict] = {}
    all_scores: list[dict] = []

    for dataset_result in results:
        dataset = dataset_result["dataset"]
        best_per_dataset[dataset] = {}

        # Get min/max for normalization within dataset
        ndcgs = [r["ndcg@10"] for r in dataset_result["results"]]
        ilads = [r["ilad"] for r in dataset_result["results"]]
        ndcg_min, ndcg_max = min(ndcgs), max(ndcgs)
        ilad_min, ilad_max = min(ilads), max(ilads)

        for strategy in STRATEGIES:
            runs = [r for r in dataset_result["results"] if r["strategy"] == strategy and 0 < r["diversity"] < 1]

            best_f1: float = 0.0
            best_run = None

            for r in runs:
                ndcg_norm = (r["ndcg@10"] - ndcg_min) / (ndcg_max - ndcg_min) if ndcg_max > ndcg_min else 0
                ilad_norm = (r["ilad"] - ilad_min) / (ilad_max - ilad_min) if ilad_max > ilad_min else 0

                if ndcg_norm + ilad_norm > 0:
                    f1 = 2 * ndcg_norm * ilad_norm / (ndcg_norm + ilad_norm)
                else:
                    f1 = 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_run = r

            if best_run:
                best_per_dataset[dataset][strategy] = {
                    "f1": best_f1,
                    "ndcg": best_run["ndcg@10"],
                    "ilad": best_run["ilad"],
                    "lambda": best_run["diversity"],
                }
                all_scores.append({"strategy": strategy, "f1": best_f1})

    # Compute average F1 per strategy
    strategy_avg: dict[str, float] = {}
    for strategy in STRATEGIES:
        scores: list[float] = [s["f1"] for s in all_scores if s["strategy"] == strategy]
        strategy_avg[strategy] = sum(scores) / len(scores) if scores else 0.0

    return best_per_dataset, strategy_avg


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


def _format_ranking_table(f1_ranking: dict) -> list[str]:
    """Format the F1 ranking table as markdown lines."""
    lines = [
        "## Overall Strategy Ranking\n",
        "Ranking by average F1 score (harmonic mean of normalized nDCG and ILAD):\n",
        "| Rank | Strategy | Avg F1 | Interpretation |",
        "|:----:|----------|:------:|----------------|",
    ]

    sorted_ranking = sorted(f1_ranking.items(), key=lambda x: -x[1])
    interpretations = {
        "msd": "Best diversity with acceptable relevance",
        "dpp": "Good balance of both metrics",
        "mmr": "Conservative, relevance-focused",
        "ssd": "Conservative, relevance-focused",
    }
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}

    for i, (strategy, score) in enumerate(sorted_ranking):
        medal = medals.get(i, f"{i + 1}.")
        lines.append(f"| {medal} | **{strategy.upper()}** | {score:.3f} | {interpretations.get(strategy, '')} |")

    lines.append("")
    return lines


def _format_f1_details(f1_per_dataset: dict) -> list[str]:
    """Format the detailed F1 table as markdown lines."""
    lines = [
        "## Best F1 Score per Dataset\n",
        "| Dataset | Strategy | λ | nDCG@10 | ILAD | F1 |",
        "|---------|----------|--:|--------:|-----:|---:|",
    ]

    for dataset in sorted(f1_per_dataset.keys()):
        for strategy in STRATEGIES:
            if strategy in f1_per_dataset[dataset]:
                d = f1_per_dataset[dataset][strategy]
                lines.append(
                    f"| {dataset} | {strategy.upper()} | {d['lambda']:.1f} | "
                    f"{d['ndcg']:.4f} | {d['ilad']:.3f} | {d['f1']:.3f} |"
                )

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

    # Add summary row
    lines.append("")
    lines.append("### Summary: Wins per Strategy by Region\n")
    lines.append("| Region | SSD | DPP | MSD | MMR | Recommendation |")
    lines.append("|--------|:---:|:---:|:---:|:---:|----------------|")

    recommendations = {
        "Low (0.3-0.5)": "Use **SSD** for light diversification",
        "Moderate (0.5-0.7)": "Use **MSD** or **DPP** for moderate diversity",
        "High (0.7-0.9)": "Use **MSD** for heavy diversification",
        "Maximum (0.9+)": "Use **MSD** for maximum diversity",
    }

    for region in region_order:
        wins = regional_data["aggregate"].get(region, {})
        row = [region]
        for s in ["ssd", "dpp", "msd", "mmr"]:
            count = wins.get(s, 0)
            row.append(str(count) if count > 0 else "-")
        row.append(recommendations.get(region, ""))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
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
        lines.append("| Strategy | λ | nDCG@10 | MRR | ILAD | Latency |")
        lines.append("|----------|--:|--------:|----:|-----:|--------:|")

        for r in sorted(dataset_result["results"], key=lambda x: (x["strategy"], x["diversity"])):
            lines.append(
                f"| {r['strategy'].upper()} | {r['diversity']:.1f} | "
                f"{r['ndcg@10']:.4f} | {r['mrr']:.4f} | {r['ilad']:.3f} | "
                f"{r['latency_ms']:.2f}ms |"
            )
        lines.append("")

    lines.append("</details>\n")
    return lines


def generate_markdown(results: list[dict]) -> str:
    """Generate markdown report content from benchmark results."""
    # Compute analysis
    retention_table = _compute_retention_table(results, retention_pct=0.95)
    f1_per_dataset, f1_ranking = _compute_f1_scores(results)
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

    # Add analysis tables
    md_lines.extend(_format_regional_analysis(regional_data))
    md_lines.extend(_format_retention_table(retention_table))
    md_lines.extend(_format_ranking_table(f1_ranking))
    md_lines.extend(_format_f1_details(f1_per_dataset))
    md_lines.extend(_format_full_results(results))

    return "\n".join(md_lines)


def generate_pareto_plot(all_data: list[dict], output_path: Path) -> None:
    """Generate Pareto frontier plot showing relevance vs diversity tradeoff."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    datasets = sorted(set(d["dataset"] for d in all_data))
    strategies = ["mmr", "msd", "dpp", "ssd"]
    colors = {"mmr": "#e74c3c", "msd": "#2ecc71", "dpp": "#3498db", "ssd": "#9b59b6"}

    # Clean dataset names for titles
    name_map = {
        "ml-32m": "MovieLens-32M",
        "lastfm": "Last.FM",
        "amazon-product-reviews-video-games": "Amazon Video Games",
        "goodreads-rating": "Goodreads",
    }

    for ax, dataset in zip(axes, datasets):
        # Add shaded regions for diversity levels
        ax.axvspan(0.3, 0.5, alpha=0.1, color="blue", label="_Low")
        ax.axvspan(0.5, 0.7, alpha=0.1, color="green", label="_Moderate")
        ax.axvspan(0.7, 0.9, alpha=0.1, color="orange", label="_High")
        ax.axvspan(0.9, 1.0, alpha=0.1, color="red", label="_Max")

        for strategy in strategies:
            points = [d for d in all_data if d["dataset"] == dataset and d["strategy"] == strategy]
            points = sorted(points, key=lambda x: x["lambda"])

            if points:
                x = [p["ilad"] for p in points]
                y = [p["ndcg"] for p in points]
                ax.plot(x, y, "o-", color=colors[strategy], label=strategy.upper(), markersize=7, linewidth=2.5)

        ax.set_xlabel("ILAD (Diversity) →", fontsize=10)
        ax.set_ylabel("nDCG@10 (Relevance) →", fontsize=10)
        ax.set_title(name_map.get(dataset, dataset), fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0.25, 1.05)

    # Add region labels to first subplot
    ax0 = axes[0]
    y_top = ax0.get_ylim()[1]
    ax0.text(0.4, y_top * 0.95, "Low", ha="center", fontsize=8, color="blue", alpha=0.7)
    ax0.text(0.6, y_top * 0.95, "Med", ha="center", fontsize=8, color="green", alpha=0.7)
    ax0.text(0.8, y_top * 0.95, "High", ha="center", fontsize=8, color="orange", alpha=0.7)
    ax0.text(0.95, y_top * 0.95, "Max", ha="center", fontsize=8, color="red", alpha=0.7)

    plt.suptitle("Relevance vs Diversity Tradeoff by Strategy", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_latency_plot(all_data: list[dict], output_path: Path) -> None:
    """Generate latency comparison bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = sorted(set(d["dataset"] for d in all_data))
    strategies = ["mmr", "msd", "dpp", "ssd"]
    colors = dict(zip(strategies, sns.color_palette("husl", len(strategies))))

    # Average latency per strategy/dataset at λ=0.6
    latency_data = {}
    for d in all_data:
        if d["lambda"] == 0.6:
            key = (d["dataset"], d["strategy"])
            latency_data[key] = d["latency_ms"]

    if not latency_data:
        plt.close()
        return

    x_labels: list[str] = []
    bar_data: dict[str, list[float]] = {s: [] for s in strategies}

    for dataset in datasets:
        x_labels.append(dataset)
        for strategy in strategies:
            val = latency_data.get((dataset, strategy), 0)
            bar_data[strategy].append(val)

    x = np.arange(len(x_labels))
    width = 0.2

    for i, (strategy, values) in enumerate(bar_data.items()):
        ax.bar(x + i * width, values, width, label=strategy.upper(), color=colors[strategy])

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Comparison at λ=0.6")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(results_dir: Path) -> None:
    """Generate full report from JSON results in a directory."""
    results = []
    for path in results_dir.glob("*.json"):
        with open(path) as f:
            results.append(json.load(f))

    if not results:
        logger.warning("No results found.")
        return

    # Collect all data points
    all_data = []
    for dataset_result in results:
        dataset = dataset_result["dataset"]
        for r in dataset_result["results"]:
            all_data.append(
                {
                    "dataset": dataset,
                    "strategy": r["strategy"],
                    "lambda": r["diversity"],
                    "ndcg": r["ndcg@10"],
                    "mrr": r["mrr"],
                    "ilad": r["ilad"],
                    "latency_ms": r["latency_ms"],
                }
            )

    # Generate markdown
    md_content = generate_markdown(results)
    report_path = results_dir / "RESULTS.md"
    report_path.write_text(md_content)
    logger.debug(f"Saved: {report_path}")

    # Generate plots
    pareto_path = results_dir / "pareto.png"
    generate_pareto_plot(all_data, pareto_path)
    logger.debug(f"Saved: {pareto_path}")

    latency_path = results_dir / "latency.png"
    generate_latency_plot(all_data, latency_path)
    logger.debug(f"Saved: {latency_path}")

    logger.info(f"Report generated: {results_dir}")

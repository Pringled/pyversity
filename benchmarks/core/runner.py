from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyversity
from scipy import sparse
from tqdm import tqdm

from benchmarks.core.config import BenchmarkConfig
from benchmarks.core.data import InteractionData, load_dataset
from benchmarks.core.embeddings import compute_similarity_matrix, generate_embeddings, get_candidates
from benchmarks.core.metrics import ilad, ilmd, mrr, ndcg

logger = logging.getLogger(__name__)


def run_benchmark(config: BenchmarkConfig) -> dict:
    """Run benchmark suite and return results dictionary."""
    if config.dataset is None:
        msg = "config.dataset must be specified"
        raise ValueError(msg)

    dataset_name = config.dataset if isinstance(config.dataset, str) else config.dataset.name
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"{dataset_name}.json"

    # Load existing results if resuming
    existing_runs: dict[int, list[dict]] = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        if "per_run_results" in existing:
            existing_runs = {int(k): v for k, v in existing["per_run_results"].items()}
            logger.info(f"Resuming: found {len(existing_runs)} existing runs")

    # Load and prepare data (once, shared across runs)
    logger.debug("[1/4] Loading dataset...")
    data = load_dataset(config.dataset, config.min_interactions, config.rating_threshold)

    logger.debug("[2/4] Generating embeddings...")
    embeddings = generate_embeddings(data, dim=config.embedding_dim, seed=config.seed)

    logger.debug("[3/4] Computing similarity matrix...")
    similarity = compute_similarity_matrix(embeddings, top_k=100)
    logger.debug(f"Similarity matrix: {similarity.shape}, nnz={similarity.nnz:,}")

    # Run multiple times with different seeds for robustness
    per_run_results: dict[int, list[dict]] = existing_runs.copy()

    for run_idx in range(config.n_runs):
        # Skip if already completed
        if run_idx in per_run_results:
            logger.info(f"Skipping run {run_idx + 1}/{config.n_runs} (already completed)")
            continue

        run_seed = config.seed + run_idx
        rng = np.random.default_rng(run_seed)

        # Sample users (different sample per run)
        user_counts = np.bincount(data.user_ids, minlength=data.n_users)
        eligible = np.where(user_counts >= 2)[0]
        sampled = rng.choice(eligible, min(len(eligible), config.sample_users), replace=False)

        run_desc = f"Run {run_idx + 1}/{config.n_runs}" if config.n_runs > 1 else "Users"
        logger.debug(f"[4/4] {run_desc}: Evaluating {len(sampled)} users...")

        # Run evaluation for this run
        run_results = []
        for user_id in tqdm(sampled, desc=run_desc):
            user_results = _evaluate_user(user_id, data, embeddings, similarity, config, rng)
            run_results.extend(user_results)

        # Save this run immediately (for resumability)
        per_run_results[run_idx] = _aggregate_single_run(run_results)
        _save_intermediate(output_path, dataset_name, data, config, per_run_results)

    # Aggregate results across all runs
    aggregated = _aggregate_across_runs(per_run_results)

    # Build final output
    output = {
        "dataset": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "n_interactions": len(data.user_ids),
            "sample_users": config.sample_users,
            "n_runs": config.n_runs,
            "total_evaluations": config.sample_users * config.n_runs,
            "k": config.k,
            "embedding_dim": config.embedding_dim,
            "seed": config.seed,
        },
        "per_run_results": {str(k): v for k, v in per_run_results.items()},
        "results": aggregated,
    }

    # Save final results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.debug(f"Saved results to: {output_path}")

    # Print summary table (keep as print for nice formatting)
    _print_summary(aggregated)

    return output


def _save_intermediate(
    output_path: Path, dataset_name: str, data: InteractionData, config: BenchmarkConfig, per_run_results: dict
) -> None:
    """Save intermediate results after each run for resumability."""
    intermediate = {
        "dataset": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "n_interactions": len(data.user_ids),
            "sample_users": config.sample_users,
            "n_runs": config.n_runs,
            "k": config.k,
            "embedding_dim": config.embedding_dim,
            "seed": config.seed,
        },
        "per_run_results": {str(k): v for k, v in per_run_results.items()},
        "results": _aggregate_across_runs(per_run_results),
    }
    with open(output_path, "w") as f:
        json.dump(intermediate, f, indent=2)
    logger.debug(f"Saved intermediate results ({len(per_run_results)} runs completed)")


def _evaluate_user(
    user_id: int,
    data: InteractionData,
    embeddings: np.ndarray,
    similarity: sparse.csr_matrix,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> list[dict]:
    """Evaluate all strategies for a single user using leave-one-out."""
    # Get user's items
    mask = data.user_ids == user_id
    profile = data.item_ids[mask]

    if len(profile) < 2:
        return []

    # Leave-one-out split
    test_idx = rng.integers(len(profile))
    test_items = np.array([profile[test_idx]])
    profile_items = np.delete(profile, test_idx)

    # Generate candidates
    candidate_ids, relevance_scores = get_candidates(
        profile_items, similarity, config.topk_similar_per_item, config.max_candidates
    )

    if len(candidate_ids) < config.k:
        return []

    candidate_embeddings = embeddings[candidate_ids]
    results = []

    for strategy in config.strategies:
        for diversity in config.diversity_values:
            result = pyversity.diversify(
                embeddings=candidate_embeddings,
                scores=relevance_scores,
                k=config.k,
                strategy=strategy,
                diversity=diversity,
            )

            selected = candidate_ids[result.indices]
            results.append(
                {
                    "strategy": strategy.value,
                    "diversity": diversity,
                    "mrr": mrr(selected, test_items),
                    "ndcg@10": ndcg(selected, test_items, k=10),
                    "ilad": ilad(selected, embeddings),
                    "ilmd": ilmd(selected, embeddings),
                }
            )

    return results


def _aggregate_single_run(results: list[dict]) -> list[dict]:
    """Aggregate per-user results from a single run into means."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in results:
        key = (r["strategy"], r["diversity"])
        groups[key].append(r)

    aggregated = []
    for (strategy, diversity), group in sorted(groups.items()):
        agg = {"strategy": strategy, "diversity": diversity}
        for metric in ["mrr", "ndcg@10", "ilad", "ilmd"]:
            values = [r[metric] for r in group]
            agg[metric] = float(np.mean(values))
        aggregated.append(agg)

    return aggregated


def _aggregate_across_runs(per_run_results: dict[int, list[dict]]) -> list[dict]:
    """Aggregate results across multiple runs, computing mean and std."""
    if not per_run_results:
        return []

    # Group by (strategy, diversity) across runs
    groups: dict[tuple[str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for run_results in per_run_results.values():
        for row in run_results:
            key = (row["strategy"], row["diversity"])
            for metric in ["mrr", "ndcg@10", "ilad", "ilmd"]:
                groups[key][metric].append(row[metric])

    aggregated = []
    for (strategy, diversity), metrics in sorted(groups.items()):
        agg = {"strategy": strategy, "diversity": diversity}
        for metric in ["mrr", "ndcg@10", "ilad", "ilmd"]:
            values = metrics[metric]
            agg[metric] = float(np.mean(values))
            agg[f"{metric}_std"] = float(np.std(values))
        aggregated.append(agg)

    return aggregated


def _print_summary(results: list[dict]) -> None:
    """Print summary table (uses print for table formatting)."""
    print("\n" + "=" * 55)  # noqa: T201
    print("RESULTS SUMMARY")  # noqa: T201
    print("=" * 55)  # noqa: T201
    print(f"{'Strategy':<10} {'λ':>5} {'nDCG@10':>10} {'MRR':>10} {'ILAD':>10}")  # noqa: T201
    print("-" * 55)  # noqa: T201

    for row in results:
        print(  # noqa: T201
            f"{row['strategy']:<10} {row['diversity']:>5.1f} "
            f"{row['ndcg@10']:>10.4f} {row['mrr']:>10.4f} {row['ilad']:>10.4f}"
        )

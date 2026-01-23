"""Core benchmark execution logic."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

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
    """
    Run the benchmark suite.

    Args:
    ----
        config: Benchmark configuration

    Returns:
    -------
        Results dictionary with config and per-strategy metrics

    """
    if config.dataset is None:
        raise ValueError("config.dataset must be specified")

    rng = np.random.default_rng(config.seed)

    # Load and prepare data
    logger.debug("[1/4] Loading dataset...")
    data = load_dataset(config.dataset, config.min_interactions, config.rating_threshold)

    logger.debug("[2/4] Generating embeddings...")
    embeddings = generate_embeddings(data, dim=config.embedding_dim, seed=config.seed)

    logger.debug("[3/4] Computing similarity matrix...")
    similarity = compute_similarity_matrix(embeddings, top_k=100)
    logger.debug(f"Similarity matrix: {similarity.shape}, nnz={similarity.nnz:,}")

    # Sample users
    user_counts = np.bincount(data.user_ids, minlength=data.n_users)
    eligible = np.where(user_counts >= 2)[0]
    sampled = rng.choice(eligible, min(len(eligible), config.sample_users), replace=False)
    logger.debug(f"[4/4] Evaluating {len(sampled)} users...")

    # Run evaluation
    all_results = []
    for user_id in tqdm(sampled, desc="Users"):
        user_results = _evaluate_user(user_id, data, embeddings, similarity, config, rng)
        all_results.extend(user_results)

    # Aggregate results
    aggregated = _aggregate_results(all_results, config)

    # Build output
    dataset_name = config.dataset if isinstance(config.dataset, str) else config.dataset.name
    output = {
        "dataset": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "n_interactions": len(data.user_ids),
            "sample_users": len(sampled),
            "k": config.k,
            "embedding_dim": config.embedding_dim,
            "seed": config.seed,
        },
        "results": aggregated,
    }

    # Save
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"{dataset_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.debug(f"Saved results to: {output_path}")

    # Print summary table (keep as print for nice formatting)
    _print_summary(aggregated)

    return output


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


def _aggregate_results(results: list[dict], config: BenchmarkConfig) -> list[dict]:
    """Aggregate per-user results by strategy/diversity into mean and std."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (r["strategy"], r["diversity"])
        groups[key].append(r)

    aggregated = []
    for (strategy, diversity), group in sorted(groups.items()):
        agg = {"strategy": strategy, "diversity": diversity}
        for metric in ["mrr", "ndcg@10", "ilad", "ilmd"]:
            values = [r[metric] for r in group]
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

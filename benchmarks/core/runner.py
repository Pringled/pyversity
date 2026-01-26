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

    # Load raw data (once, shared across runs)
    logger.info("[1/4] Loading dataset...")
    data = load_dataset(config.dataset, config.min_interactions, config.rating_threshold)

    # Run multiple times with different seeds for robustness
    per_run_results: dict[int, list[dict]] = existing_runs.copy()

    for run_idx in range(config.n_runs):
        # Skip if already completed
        if run_idx in per_run_results:
            logger.info(f"Skipping run {run_idx + 1}/{config.n_runs} (already completed)")
            continue

        run_seed = config.seed + run_idx
        rng = np.random.default_rng(run_seed)

        # Sample users and their held-out items FIRST (before training)
        user_counts = np.bincount(data.user_ids, minlength=data.n_users)
        eligible = np.where(user_counts >= 2)[0]
        sampled_users = rng.choice(eligible, min(len(eligible), config.sample_users), replace=False)

        # For each sampled user, select their held-out item
        holdout_map: dict[int, int] = {}  # user_id -> held-out item_id
        for user_id in sampled_users:
            user_mask = data.user_ids == user_id
            user_items = data.item_ids[user_mask]
            holdout_idx = rng.integers(len(user_items))
            holdout_map[user_id] = user_items[holdout_idx]

        # Create training data by removing held-out edges (leakage-free)
        logger.info(
            f"[2/4] Run {run_idx + 1}/{config.n_runs}: Creating training data (removing {len(holdout_map)} held-out edges)..."
        )
        train_data = _create_training_data(data, holdout_map)

        # Train embeddings on training data only (no leakage)
        logger.info(f"[2/4] Run {run_idx + 1}/{config.n_runs}: Generating embeddings...")
        embeddings = generate_embeddings(train_data, dim=config.embedding_dim, seed=run_seed)

        logger.info(f"[3/4] Run {run_idx + 1}/{config.n_runs}: Computing similarity matrix...")
        similarity = compute_similarity_matrix(embeddings, top_k=100)

        run_desc = f"Run {run_idx + 1}/{config.n_runs}" if config.n_runs > 1 else "Users"
        logger.info(f"[4/4] {run_desc}: Evaluating {len(sampled_users)} users...")

        # Run evaluation for this run (use train_data for profile to avoid leakage)
        run_results = []
        for user_id in tqdm(sampled_users, desc=run_desc):
            test_item = holdout_map[user_id]
            user_results = _evaluate_user(user_id, test_item, train_data, data, embeddings, similarity, config)
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
    logger.info(f"Saved results to: {output_path}")

    # Print summary table (keep as print for nice formatting)
    _print_summary(aggregated)

    return output


def _create_training_data(data: InteractionData, holdout_map: dict[int, int]) -> InteractionData:
    """Create training data by removing held-out edges (vectorized for speed)."""
    # Build lookup array: holdout_item[user] = item to hold out, or -1 if not in holdout
    holdout_item = np.full(data.n_users, -1, dtype=data.item_ids.dtype)
    for user_id, item_id in holdout_map.items():
        holdout_item[user_id] = item_id

    # Vectorized mask: remove exactly the (user, holdout_item[user]) edge for users in holdout_map
    is_holdout_user = holdout_item[data.user_ids] != -1
    is_holdout_edge = is_holdout_user & (data.item_ids == holdout_item[data.user_ids])
    keep_mask = ~is_holdout_edge

    return InteractionData(
        user_ids=data.user_ids[keep_mask],
        item_ids=data.item_ids[keep_mask],
        n_users=data.n_users,
        n_items=data.n_items,
    )


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
    test_item: int,
    train_data: InteractionData,
    original_data: InteractionData,
    embeddings: np.ndarray,
    similarity: sparse.csr_matrix,
    config: BenchmarkConfig,
) -> list[dict]:
    """
    Evaluate all strategies for a single user using leave-one-out.

    Uses next-item prediction: test_item is eligible as a candidate,
    but other already-interacted items are excluded.
    """
    # Get user's profile from training data (held-out item already removed)
    mask = train_data.user_ids == user_id
    profile_items = train_data.item_ids[mask]
    test_items = np.array([test_item])

    if len(profile_items) < 1:
        return []

    # Get all items user has interacted with (from original data)
    # Exclude these from candidates, EXCEPT the test_item (which we want to retrieve)
    orig_mask = original_data.user_ids == user_id
    all_seen_items = set(original_data.item_ids[orig_mask])
    exclude_items = all_seen_items - {test_item}

    # Generate candidates using training profile, excluding seen items (except test)
    candidate_ids, relevance_scores = get_candidates(
        profile_items, similarity, config.topk_similar_per_item, config.max_candidates, exclude_items
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

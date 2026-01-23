from __future__ import annotations

import time

import numpy as np
import pyversity


def run_latency_benchmark(
    candidate_sizes: list[int] | None = None,
    k: int = 10,
    embedding_dim: int = 256,
    n_runs: int = 10,
    seed: int = 42,
) -> list[dict]:
    """
    Run synthetic latency benchmark with random embeddings.

    Measures how latency scales with number of candidates for each strategy.
    Uses 256d embeddings by default (common for modern embedding models).
    """
    if candidate_sizes is None:
        candidate_sizes = [100, 250, 500, 1000, 2000, 5000]

    rng = np.random.default_rng(seed)
    strategies = [
        pyversity.Strategy.MMR,
        pyversity.Strategy.MSD,
        pyversity.Strategy.DPP,
        pyversity.Strategy.SSD,
    ]

    results = []
    for num_candidates in candidate_sizes:
        embeddings = rng.standard_normal((num_candidates, embedding_dim)).astype(np.float32)
        scores = rng.uniform(0, 1, num_candidates).astype(np.float32)

        for strategy in strategies:
            latencies = []
            for _ in range(n_runs):
                start = time.perf_counter()
                pyversity.diversify(
                    embeddings=embeddings,
                    scores=scores,
                    k=k,
                    strategy=strategy,
                    diversity=0.5,
                )
                latencies.append((time.perf_counter() - start) * 1000)

            results.append(
                {
                    "n_candidates": num_candidates,
                    "strategy": strategy.value,
                    "latency_ms": float(np.median(latencies)),
                    "latency_std": float(np.std(latencies)),
                }
            )

    return results

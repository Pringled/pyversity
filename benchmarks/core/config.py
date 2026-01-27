from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyversity import Strategy


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Dataset (name from registry or DatasetInfo)
    dataset: str | None = None
    min_interactions: int = 5
    rating_threshold: float | None = None  # None = use dataset default

    # Embeddings
    embedding_dim: int = 64

    # Candidate generation
    topk_similar_per_item: int = 50
    max_candidates: int = 1000

    # Evaluation
    k: int = 10
    sample_users: int = 2000

    # Strategies
    strategies: list[Strategy] = field(default_factory=lambda: [Strategy.MMR, Strategy.MSD, Strategy.DPP, Strategy.SSD])
    diversity_values: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    # Output
    output_dir: Path = Path("benchmarks/results")

    # Reproducibility
    seed: int = 42
    n_runs: int = 10  # Number of runs with different seeds for robustness

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        self.output_dir = Path(self.output_dir)

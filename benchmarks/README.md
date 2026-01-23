# Pyversity Benchmarks

This directory contains comprehensive benchmarks for **MMR**, **MSD**, **DPP**, and **SSD** across 4 recommendation datasets.

For each dataset, we use a leave-one-out evaluation protocol (meaning we hold out one item
per user as the test set), generate candidate items based on similar items, then rerank with each
diversification strategy. We measure relevance (nDCG, MRR) and diversity (ILAD) to trace the
Pareto frontier of the relevance-diversity tradeoff.

In addition, we measure latency of each strategy as the number of candidates scales.

> **Note:** We don't benchmark COVER (coverage-based diversification) because it optimizes a different
> objective (topic/category coverage) and requires explicit item taxonomies not available in standard
> collaborative filtering datasets.

## Table of Contents

- [Key Findings](#key-findings)
- [Relevance-Diversity Tradeoff](#relevance-diversity-tradeoff)
- [Latency](#latency)
- [Usage](#usage)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Citations](#citations)

## Key Findings

**Different strategies excel at different diversity levels:**

| Diversity Level | Winner | λ | Notes |
|-----------------|--------|---|-------|
| Low (ILAD 0.3-0.5) | **SSD** | 0.6-0.8 | Highest relevance at light diversification |
| Moderate (ILAD 0.5-0.7) | **MSD/DPP** | 0.4-0.6 | Good middle-ground |
| High (ILAD 0.7-0.9) | **MSD** | 0.4-0.6 | Best for heavy diversification |
| Maximum (ILAD 0.9+) | **MSD** | 0.2-0.4 | Dominates at extreme diversity |

## Relevance-Diversity Tradeoff

![Relevance vs Diversity Tradeoff](results/pareto.png)

## Latency

All strategies are extremely fast in practice. The plot below shows latency scaling with the number of candidates (k=10, d=256):

![Latency vs Candidates](results/latency.png)

**Key observations:**
- **All strategies are very fast**: Even with 10,000 candidates, all strategies complete in under 100ms
- **Typical use case**: With 100 candidates (common for reranking), all strategies complete in <1ms
- **MMR/MSD/DPP** are nearly identical in speed for practical candidate sizes
- **SSD** is slower due to Gram-Schmidt orthogonalization—scales with embedding dimension d

| Strategy | 100 candidates | 1,000 candidates | 10,000 candidates |
|----------|----------------|------------------|-------------------|
| MMR | ~0.1ms | ~1ms | ~10ms |
| MSD | ~0.1ms | ~1ms | ~10ms |
| DPP | ~0.1ms | ~2ms | ~20ms |
| SSD | ~0.5ms | ~5ms | ~80ms |

*Measured with k=10 items selected, d=256 dimensional embeddings.*

## Usage

```bash
# Download datasets
python -m benchmarks download

# Run benchmarks
python -m benchmarks run

# Generate report
python -m benchmarks report
```

<details>
<summary>Detailed Results</summary>

### Strategies

| Strategy | Description | Complexity |
|----------|-------------|------------|
| **MMR** | Maximal Marginal Relevance | O(k·n) |
| **MSD** | Max-Sum Diversification | O(k·n) |
| **DPP** | Determinantal Point Process | O(k²·n) |
| **SSD** | Sliding Spectrum Decomposition | O(k²·n·d) |

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| nDCG@10 | Relevance | Normalized Discounted Cumulative Gain |
| MRR | Relevance | Mean Reciprocal Rank |
| ILAD | Diversity | Intra-List Average Distance |

### Per-Dataset Winners

| Dataset | Low (0.3-0.5) | Moderate (0.5-0.7) | High (0.7-0.9) | Max (0.9+) |
|---------|:-------------:|:------------------:|:--------------:|:----------:|
| MovieLens-32M | SSD | DPP | MSD | - |
| Last.FM | SSD | MSD | MSD | MSD |
| Amazon-VG | - | - | MMR | MSD |
| Goodreads | SSD | MSD | MSD | MSD |

### Full Results

See [`results/RESULTS.md`](results/RESULTS.md) for complete tables.

</details>

## Datasets

| Dataset | Domain | Interactions | Source |
|---------|--------|--------------|--------|
| MovieLens-32M | Movies | 32M ratings | [GroupLens](https://grouplens.org/datasets/movielens/32m/) |
| Last.FM | Music | 92K plays | [HetRec 2011](http://ir.ii.uam.es/hetrec2011/) |
| Amazon Video Games | Games | 47K reviews | [UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets.html) |
| Goodreads | Books | 869K ratings | [UCSD](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/) |

## Methodology

- **Evaluation**: Leave-one-out protocol with 2,000 sampled users per dataset
- **Embeddings**: 64-dim SVD on item co-occurrence matrix
- **Candidates**: Top-100 similar items per profile item
- **Selection**: k=20 items selected by each strategy
- **λ sweep**: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

<details>
<summary>Programmatic API</summary>

```python
from benchmarks import BenchmarkConfig, run_benchmark
from pyversity import Strategy

config = BenchmarkConfig(
    dataset_path="local/data/ml-32m",
    sample_users=1000,
    strategies=[Strategy.MMR, Strategy.DPP, Strategy.MSD, Strategy.SSD],
    diversity_values=[0.0, 0.3, 0.5, 0.7, 1.0],
)
results = run_benchmark(config)
```

</details>

## Citations

```bibtex
@article{harper2015movielens,
  title={The MovieLens Datasets: History and Context},
  author={Harper, F Maxwell and Konstan, Joseph A},
  journal={ACM TiiS}, year={2015}
}

@inproceedings{cantador2011hetrec,
  title={2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems},
  author={Cantador, Iv{\'a}n and Brusilovsky, Peter and Kuflik, Tsvi},
  booktitle={RecSys}, year={2011}
}

@inproceedings{ni2019amazon,
  title={Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects},
  author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
  booktitle={EMNLP}, year={2019}
}

@inproceedings{wan2018goodreads,
  title={Item Recommendation on Monotonic Behavior Chains},
  author={Wan, Mengting and McAuley, Julian},
  booktitle={RecSys}, year={2018}
}
```

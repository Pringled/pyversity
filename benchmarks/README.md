# Pyversity Benchmarks

This directory contains comprehensive benchmarks for **MMR**, **MSD**, **DPP**, and **SSD** across 4 recommendation datasets.

For each dataset, we use a leave-one-out evaluation protocol (meaning we hold out one item
per user as the test set), generate candidate items based on similar items, then rerank with each
diversification strategy. We measure relevance (nDCG) and diversity (ILAD, ILMD) to trace the
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

We evaluate using two diversity metrics that capture different aspects:
- **ILAD** (Intra-List Average Distance): Measures overall diversity spread
- **ILMD** (Intra-List Minimum Distance): Ensures no very similar item pairs (worst-case diversity)

### Dataset Wins by Region

We count how many datasets each strategy achieves the highest nDCG in each diversity region:

| Metric | Region   | Range       | MMR | MSD | DPP | SSD | Best    |
|--------|----------|-------------|:---:|:---:|:---:|:---:|---------|
| ILAD   | Low      | 0.3-0.5     | 0   | 1   | 1   | 1   | Tied    |
| ILAD   | Moderate | 0.5-0.7     | 0   | 2   | 1   | 0   | **MSD** |
| ILAD   | High     | 0.7-0.9     | 1   | 3   | 0   | 0   | **MSD** |
| ILMD   | Low      | 0.05-0.15   | 1   | 0   | 2   | 0   | **DPP** |
| ILMD   | Moderate | 0.15-0.25   | 1   | 0   | 1   | 0   | Tied    |
| ILMD   | High     | 0.25-0.35   | 1   | 0   | 3   | 0   | **DPP** |
| | | **Total** | **4** | **6** | **8** | **1** | **DPP** |

**Key insight:**
- **DPP** wins overall—best at balancing relevance with ILMD (no similar pairs)
- **MSD** dominates ILAD—best when you want maximum variety
- **MMR** is a solid baseline with wins across regions
- **SSD** shines in sequence-aware settings (see note below)

> **Note on SSD:** These benchmarks evaluate single-batch diversification. SSD is designed for
> **sequence-aware** diversification with `recent_embeddings`—it rewards novelty relative to
> recently shown items. For content feeds, infinite scroll, or conversational RAG where you
> maintain a sliding window of recent items, SSD may outperform these results.

### Recommendations

| Goal | Strategy | diversity | Why |
|-----------|----------|-----------|-----|
| Maximum variety | **MSD** | 0.7-1.0 | Dominates ILAD across all regions |
| Avoid similar pairs | **DPP** | 0.5-0.8 | Best ILMD wins, balances relevance |
| Balanced tradeoff | **DPP** | 0.4-0.6 | Most total wins (8), good on both metrics |
| Sequence-aware feeds | **SSD** | 0.3-0.5 | Use with `recent_embeddings` |
| Simple baseline | **MMR** | 0.3-0.5 | Easy to implement and tune |

*`diversity=0` prioritizes relevance, `diversity=1` prioritizes diversity.*

## Relevance-Diversity Tradeoff

### ILAD (Average Diversity)

![Relevance vs ILAD Tradeoff](results/pareto_ilad.png)

*Higher ILAD = more overall variety in recommendations*

### ILMD (Minimum Diversity)

![Relevance vs ILMD Tradeoff](results/pareto_ilmd.png)

*Higher ILMD = no very similar pairs (stricter diversity guarantee)*

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
| nDCG@k | Relevance | Normalized Discounted Cumulative Gain |
| ILAD | Diversity | Intra-List Average Distance (mean pairwise) |
| ILMD | Diversity | Intra-List Minimum Distance (worst-case pair) |

### Per-Dataset Winners (by ILAD region)

| Dataset | Low (0.3-0.5) | Moderate (0.5-0.7) | High (0.7-0.9) |
|---------|:-------------:|:------------------:|:--------------:|
| MovieLens-32M | DPP | DPP | MSD |
| Last.FM | SSD | MSD | MSD |
| Amazon-VG | - | - | MMR |
| Goodreads | MSD | MSD | MSD |

*Raw JSON results are in `results/*.json`*

</details>

## Datasets

| Dataset | Domain | Interactions | Source |
|---------|--------|--------------|--------|
| MovieLens-32M | Movies | 32M ratings | [GroupLens](https://grouplens.org/datasets/movielens/32m/) |
| Last.FM | Music | 92K plays | [HetRec 2011](http://ir.ii.uam.es/hetrec2011/) |
| Amazon Video Games | Games | 47K reviews | [UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets.html) |
| Goodreads | Books | 869K ratings | [UCSD](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/) |

## Methodology

### Experimental Setup

- **Evaluation**: Leave-one-out protocol with 2,000 sampled users per dataset
- **Embeddings**: 64-dim SVD on item co-occurrence matrix
- **Candidates**: Top-100 similar items per profile item
- **Selection**: k=20 items selected by each strategy
- **diversity sweep**: 0.0, 0.1, 0.2, ..., 0.9, 1.0

### How We Compute "Wins"

For each diversity region (e.g., ILAD 0.5-0.7), we:

1. **Filter** each strategy's results to points within that region
2. **Find best nDCG** achieved by each strategy in that region
3. **Compare across strategies** per dataset—the strategy with highest nDCG "wins" that dataset
4. **Count wins** across all 4 datasets to determine the overall leader

This approach measures: *"At a given diversity level, which strategy maintains the best relevance?"*

We use three regions per metric:
- **Low**: Where diversification begins (ILAD 0.3-0.5, ILMD 0.05-0.15)
- **Moderate**: Balanced tradeoff zone (ILAD 0.5-0.7, ILMD 0.15-0.25)
- **High**: Strong diversity (ILAD 0.7-0.9, ILMD 0.25-0.35)

*Note: Not all strategies reach all regions on all datasets. A "-" in the per-dataset table means no data points in that region.*

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

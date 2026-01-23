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

We use a **relevance-budgeted** approach: for each strategy, we find the best config that maintains **≥95% of baseline relevance**.

### Strategy Scorecard

| Strategy | Combined Score | Best ILAD | Best ILMD |
|----------|:--------------:|:---------:|:---------:|
| **DPP**  | **0.439**      | 0.75      | **0.26**  |
| SSD      | 0.236          | 0.62      | 0.17      |
| MSD      | 0.226          | **0.78**  | 0.15      |
| MMR      | 0.213          | 0.58      | 0.18      |

*Combined Score = geometric mean of normalized ILAD and ILMD gains. Best ILAD/ILMD = average across datasets while maintaining ≥95% baseline relevance.*

### Best Configs Per Goal (≥95% baseline nDCG)

| Goal | Winner | Wins | Why |
|------|--------|:----:|-----|
| **Best Overall** | **DPP** | 4/4 | Highest combined diversity (both ILAD and ILMD) |
| **Max ILAD** (overall variety) | **MSD** | 3/4 | Optimizes sum of pairwise distances |
| **Max ILMD** (no similar pairs) | **DPP** | 4/4 | Best worst-case diversity guarantee |

### Recommendations

| Goal | Strategy | `diversity` | Notes |
|------|----------|:-----------:|-------|
| **Best overall balance** | **DPP** | 0.8-0.9 | Wins both overall and ILMD |
| **Maximum variety** | **MSD** | 0.5-0.6 | Best ILAD while keeping relevance |
| **Avoid similar pairs** | **DPP** | 0.8-0.9 | Strictest duplicate avoidance |
| **Sequence-aware feeds** | **SSD** | 0.8-0.9 | Use with `recent_embeddings` |
| **Simple baseline** | **MMR** | 0.7-0.8 | Easy to implement, competitive |

*`diversity=0` prioritizes relevance, `diversity=1` prioritizes diversity.*

> **Note on SSD:** These benchmarks evaluate single-batch diversification. SSD is designed for
> **sequence-aware** diversification with `recent_embeddings`—it rewards novelty relative to
> recently shown items. For content feeds, infinite scroll, or conversational RAG where you
> maintain a sliding window of recent items, SSD may outperform these results.

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

### Per-Dataset Best Configs (≥95% baseline nDCG)

| Dataset | Max ILAD | Max ILMD | Best Overall |
|---------|:--------:|:--------:|:------------:|
| MovieLens-32M | DPP (`diversity`=0.9) | DPP (`diversity`=0.9) | DPP (`diversity`=0.9) |
| Last.FM | MSD (`diversity`=0.6) | DPP (`diversity`=0.8) | DPP (`diversity`=0.8) |
| Amazon-VG | MSD (`diversity`=0.5) | DPP (`diversity`=0.9) | DPP (`diversity`=0.9) |
| Goodreads | MSD (`diversity`=0.5) | DPP (`diversity`=0.8) | DPP (`diversity`=0.8) |

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

### Relevance-Budgeted Evaluation

We use a **relevance floor** approach to ensure fair comparison:

1. **Baseline**: For each dataset, compute baseline nDCG at `diversity=0` (no diversification)
2. **Filter**: Keep only configs where nDCG ≥ 95% of baseline (deployable configurations)
3. **Compare**: Among feasible configs, find which strategy achieves:
   - **Max ILAD**: Best overall diversity spread
   - **Max ILMD**: Best worst-case diversity (no similar pairs)
   - **Best Combined**: Geometric mean of normalized ILAD and ILMD gains

The **Combined Score** normalizes gains per dataset:
- `ILAD_gain = (ILAD - ILAD_min) / (ILAD_max - ILAD_min)`
- `ILMD_gain = (ILMD - ILMD_min) / (ILMD_max - ILMD_min)`
- `Combined = sqrt(ILAD_gain × ILMD_gain)`

This ensures a strategy must improve *both* metrics to score well—high only if both improve.

### Diversity Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **ILAD** | Average | Mean pairwise distance—measures overall variety |
| **ILMD** | Minimum | Minimum pairwise distance—ensures no similar pairs |

Both are computed as `1 - cosine_similarity` between item embeddings.

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

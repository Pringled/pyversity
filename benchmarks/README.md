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
- [Detailed Results](#detailed-results)
- [Usage](#usage)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Citations](#citations)

## Key Findings

We use a **relevance-budgeted** approach: for each strategy, we find the best config that maintains a minimum percentage of baseline relevance (see [Methodology](#methodology) for details).

### Overall Results

We report results at two relevance floors:
- **99%**: Near-zero relevance loss, production-safe
- **95%**: Balanced tradeoff with more diversity headroom

#### 99% Relevance Floor

| Strategy | Combined | ILAD | ILMD | `diversity` | Best For |
|----------|:--------:|:----:|:----:|:-----------:|----------|
| **DPP**  | **0.273** | 0.59 (+51%) | **0.24 (+104%)** | 0.7 | Overall balance, avoiding similar pairs |
| MMR      | 0.224    | 0.52 (+32%) | 0.20 (+67%)      | 0.7 | Simple baseline |
| SSD      | 0.195    | 0.54 (+38%) | 0.18 (+55%)      | 0.8 | Sequence-aware feeds |
| MSD      | 0.166    | **0.61 (+54%)** | 0.16 (+36%)  | 0.4 | Maximum variety (ILAD) |

#### 95% Relevance Floor

| Strategy | Combined | ILAD | ILMD | `diversity` | Best For |
|----------|:--------:|:----:|:----:|:-----------:|----------|
| **DPP**  | **0.389** | 0.69 (+79%) | **0.31 (+178%)** | 0.9 | Overall balance, avoiding similar pairs |
| SSD      | 0.254    | 0.60 (+52%) | 0.23 (+88%)      | 0.9 | Sequence-aware feeds |
| MMR      | 0.247    | 0.57 (+41%) | 0.24 (+97%)      | 0.8 | Simple baseline |
| MSD      | 0.228    | **0.68 (+71%)** | 0.20 (+47%)  | 0.5 | Maximum variety (ILAD) |

*Combined Score = geometric mean of normalized ILAD and ILMD gains (higher = better). Percentages show improvement vs baseline (diversity=0). `diversity` shows the typical parameter value to achieve these results.*

**DPP wins at both thresholds** by balancing both ILAD (variety) and ILMD (worst-case diversity). At 95% floor, DPP achieves +79% ILAD and +178% ILMD improvement—a substantial diversity boost for just 5% relevance cost.

### Recommendations

| Goal | Strategy | `diversity` | Notes |
|------|----------|:-----------:|-------|
| **Best overall balance** | **DPP** | 0.8-0.9 | Wins both overall and ILMD |
| **Maximum diversity** | **MSD** | 0.4-0.5 (99%) / 0.5-0.6 (95%) | Best ILAD while keeping relevance |
| **Avoid similar pairs** | **DPP** | 0.7-0.9 | Best worst-case diversity (highest ILMD) |
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

*Higher ILAD = more overall variety in recommendations. ★ marks the best ILAD point across all strategies (≥95% baseline relevance).*

### ILMD (Minimum Diversity)

![Relevance vs ILMD Tradeoff](results/pareto_ilmd.png)

*Higher ILMD = better worst-case diversity (fewer similar pairs). ★ marks the best ILMD point across all strategies (≥95% baseline relevance).*

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

## Detailed Results

<details>
<summary>Per-Dataset Best Configs (99% Relevance Floor)</summary>

| Dataset | Max ILAD | Max ILMD | Best Overall |
|---------|:--------:|:--------:|:------------:|
| MovieLens-32M | DPP (`diversity`=0.8) | DPP (`diversity`=0.8) | DPP (`diversity`=0.8) |
| Last.FM | DPP (`diversity`=0.7) | DPP (`diversity`=0.7) | DPP (`diversity`=0.7) |
| Amazon-VG | MSD (`diversity`=0.4) | MMR (`diversity`=0.7) | MMR (`diversity`=0.7) |
| Goodreads | MSD (`diversity`=0.4) | DPP (`diversity`=0.7) | DPP (`diversity`=0.7) |

</details>

<details>
<summary>Per-Dataset Best Configs (95% Relevance Floor)</summary>

| Dataset | Max ILAD | Max ILMD | Best Overall |
|---------|:--------:|:--------:|:------------:|
| MovieLens-32M | DPP (`diversity`=0.9) | DPP (`diversity`=0.9) | DPP (`diversity`=0.9) |
| Last.FM | MSD (`diversity`=0.6) | DPP (`diversity`=0.8) | DPP (`diversity`=0.8) |
| Amazon-VG | MSD (`diversity`=0.5) | MMR (`diversity`=0.7) | MSD (`diversity`=0.5) |
| Goodreads | MSD (`diversity`=0.5) | DPP (`diversity`=0.8) | DPP (`diversity`=0.8) |

</details>

<details>
<summary>Per-Dataset Detailed Metrics (95% floor)</summary>

The tables below show best achievable metrics per strategy while maintaining ≥95% of baseline nDCG.

#### MovieLens-32M

| Strategy | Best `diversity` | nDCG | ILAD | ILMD | Combined |
|----------|:----------------:|:----:|:----:|:----:|:--------:|
| **DPP**  | 0.9              | 0.052 | 0.67 | 0.31 | **0.46** |
| SSD      | 0.9              | 0.053 | 0.60 | 0.22 | 0.32     |
| MSD      | 0.6              | 0.055 | 0.67 | 0.13 | 0.25     |
| MMR      | 0.8              | 0.055 | 0.56 | 0.17 | 0.27     |

#### Last.FM

| Strategy | Best `diversity` | nDCG | ILAD | ILMD | Combined |
|----------|:----------------:|:----:|:----:|:----:|:--------:|
| **DPP**  | 0.8              | 0.151 | 0.61 | 0.22 | **0.37** |
| SSD      | 0.9              | 0.152 | 0.52 | 0.14 | 0.18     |
| MSD      | 0.6              | 0.154 | 0.63 | 0.13 | 0.22     |
| MMR      | 0.8              | 0.153 | 0.52 | 0.16 | 0.19     |

#### Amazon Video Games

| Strategy | Best `diversity` | nDCG | ILAD | ILMD | Combined |
|----------|:----------------:|:----:|:----:|:----:|:--------:|
| **MSD**  | 0.5              | 0.269 | 0.95 | 0.48 | **0.50** |
| DPP      | 0.9              | 0.279 | 0.89 | 0.52 | 0.48     |
| SSD      | 0.9              | 0.271 | 0.81 | 0.35 | 0.38     |
| MMR      | 0.7              | 0.289 | 0.88 | 0.53 | 0.27     |

#### Goodreads

| Strategy | Best `diversity` | nDCG | ILAD | ILMD | Combined |
|----------|:----------------:|:----:|:----:|:----:|:--------:|
| **DPP**  | 0.8              | 0.027 | 0.60 | 0.17 | **0.24** |
| SSD      | 0.9              | 0.026 | 0.47 | 0.10 | 0.10     |
| MSD      | 0.5              | 0.027 | 0.67 | 0.11 | 0.13     |
| MMR      | 0.7              | 0.028 | 0.44 | 0.11 | 0.12     |

</details>

<details>
<summary>Per-Strategy Diversity Sweep (Averaged Across Datasets)</summary>

Shows how each strategy's metrics change as you increase `diversity`, averaged across all 4 datasets.

> **Note:** nDCG Retention can exceed 100% and may not decrease monotonically. This happens because
> diversification can sometimes *improve* relevance by breaking ties among similar items or surfacing
> items that better match the held-out test item. This effect is dataset-dependent and typically
> occurs at moderate diversity levels before eventually declining at high diversity.

#### MMR

| `diversity` | nDCG Retention | ILAD | ILMD |
|:-----------:|:--------------:|:----:|:----:|
| 0.0         | 100.0%          | 0.42 (+0%) | 0.12 (+0%) |
| 0.1         | 100.9%          | 0.42 (+1%) | 0.13 (+2%) |
| 0.2         | 101.7%          | 0.43 (+3%) | 0.13 (+5%) |
| 0.3         | 102.9%          | 0.44 (+4%) | 0.14 (+9%) |
| 0.4         | 103.4%          | 0.45 (+7%) | 0.15 (+15%) |
| 0.5         | 103.0%          | 0.46 (+10%) | 0.16 (+23%) |
| 0.6         | 103.5%          | 0.48 (+16%) | 0.18 (+36%) |
| 0.7         | 103.1%          | 0.52 (+24%) | 0.22 (+60%) |
| 0.8         | 98.1%          | 0.58 (+41%) | 0.28 (+110%) |
| 0.9         | 85.0%          | 0.71 (+79%) | 0.42 (+247%) |
| 1.0         | 46.3%          | 0.92 (+142%) | 0.76 (+655%) |

#### MSD

| `diversity` | nDCG Retention | ILAD | ILMD |
|:-----------:|:--------------:|:----:|:----:|
| 0.0         | 100.0%          | 0.42 (+0%) | 0.12 (+0%) |
| 0.1         | 102.9%          | 0.46 (+10%) | 0.13 (+5%) |
| 0.2         | 103.2%          | 0.50 (+22%) | 0.14 (+11%) |
| 0.3         | 103.1%          | 0.55 (+34%) | 0.16 (+19%) |
| 0.4         | 101.6%          | 0.60 (+48%) | 0.17 (+30%) |
| 0.5         | 98.1%          | 0.66 (+65%) | 0.20 (+45%) |
| 0.6         | 92.0%          | 0.72 (+84%) | 0.23 (+64%) |
| 0.7         | 82.7%          | 0.80 (+107%) | 0.26 (+93%) |
| 0.8         | 72.6%          | 0.87 (+129%) | 0.31 (+138%) |
| 0.9         | 56.7%          | 0.93 (+145%) | 0.40 (+224%) |
| 1.0         | 43.4%          | 0.95 (+153%) | 0.53 (+373%) |

#### DPP

| `diversity` | nDCG Retention | ILAD | ILMD |
|:-----------:|:--------------:|:----:|:----:|
| 0.0         | 100.0%          | 0.42 (+0%) | 0.12 (+0%) |
| 0.1         | 104.8%          | 0.46 (+13%) | 0.15 (+26%) |
| 0.2         | 105.2%          | 0.47 (+15%) | 0.15 (+29%) |
| 0.3         | 105.2%          | 0.48 (+17%) | 0.16 (+33%) |
| 0.4         | 105.2%          | 0.49 (+19%) | 0.16 (+38%) |
| 0.5         | 105.1%          | 0.50 (+24%) | 0.17 (+46%) |
| 0.6         | 105.2%          | 0.52 (+30%) | 0.19 (+57%) |
| 0.7         | 106.9%          | 0.56 (+41%) | 0.21 (+77%) |
| 0.8         | 103.0%          | 0.64 (+65%) | 0.25 (+118%) |
| 0.9         | 92.4%          | 0.80 (+111%) | 0.39 (+275%) |
| 1.0         | 44.6%          | 0.92 (+144%) | 0.74 (+623%) |

#### SSD

| `diversity` | nDCG Retention | ILAD | ILMD |
|:-----------:|:--------------:|:----:|:----:|
| 0.0         | 100.0%          | 0.42 (+0%) | 0.12 (+0%) |
| 0.1         | 100.7%          | 0.42 (+1%) | 0.12 (+1%) |
| 0.2         | 101.5%          | 0.43 (+2%) | 0.13 (+2%) |
| 0.3         | 102.4%          | 0.43 (+3%) | 0.13 (+4%) |
| 0.4         | 102.7%          | 0.43 (+4%) | 0.13 (+6%) |
| 0.5         | 104.2%          | 0.44 (+6%) | 0.14 (+10%) |
| 0.6         | 105.6%          | 0.46 (+10%) | 0.14 (+16%) |
| 0.7         | 105.7%          | 0.48 (+16%) | 0.15 (+25%) |
| 0.8         | 106.5%          | 0.53 (+31%) | 0.18 (+47%) |
| 0.9         | 92.6%          | 0.76 (+98%) | 0.29 (+146%) |
| 1.0         | 44.0%          | 0.94 (+149%) | 0.73 (+611%) |

*Percentages show gain vs baseline (diversity=0).*

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

- **Evaluation**: Leave-one-out protocol with up to 2,000 sampled users per dataset (all users if fewer)
- **Embeddings**: 64-dim SVD on item co-occurrence matrix
- **Candidates**: Top-100 similar items per profile item
- **Selection**: k=10 items selected by each strategy
- **diversity sweep**: 0.0, 0.1, 0.2, ..., 0.9, 1.0

### Relevance-Budgeted Evaluation

We use a **relevance floor** approach to ensure fair comparison:

1. **Baseline**: For each dataset, compute baseline nDCG at `diversity=0` (no diversification)
2. **Filter**: Keep only configs where nDCG meets the relevance floor
3. **Compare**: Among feasible configs, find which strategy achieves:
   - **Max ILAD**: Best overall diversity spread
   - **Max ILMD**: Best worst-case diversity (fewer similar pairs)
   - **Best Combined**: Geometric mean of normalized ILAD and ILMD gains

We report results at two relevance floors:
- **99%**: Near-zero relevance loss, production-safe
- **95%**: Balanced tradeoff with more diversity headroom

The **Combined Score** normalizes gains relative to baseline (λ=0):
- `ILAD_gain = (ILAD - ILAD_baseline) / (ILAD_max - ILAD_baseline)`
- `ILMD_gain = (ILMD - ILMD_baseline) / (ILMD_max - ILMD_baseline)`
- `Combined = sqrt(ILAD_gain × ILMD_gain)`

This ensures a strategy must improve *both* metrics to score well—high only if both improve.

### Diversity Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **ILAD** | Average | Mean pairwise distance—measures overall variety |
| **ILMD** | Minimum | Min pairwise distance—higher = fewer similar pairs |

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

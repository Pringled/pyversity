# Pyversity Benchmark Results

Comparison of diversification strategies across recommendation datasets.

## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| MovieLens-32M | GroupLens | 32M ratings from MovieLens |
| Last.FM | HetRec 2011 | Music listening data |
| Amazon Video Games | McAuley Lab | Product reviews |
| Goodreads | McAuley Lab | Book ratings |

## Strategies

| Strategy | Description | Complexity |
|----------|-------------|------------|
| **MMR** | Maximal Marginal Relevance | O(k·n) |
| **MSD** | Max-Sum Diversification | O(k·n) |
| **DPP** | Determinantal Point Process | O(k²·n) |
| **SSD** | Sliding Spectrum Decomposition | O(k²·n) |

## Best Strategy by Diversity Level

Which strategy achieves highest relevance (nDCG) at each diversity (ILAD) level?

| Dataset | Low (0.3-0.5) | Moderate (0.5-0.7) | High (0.7-0.9) | Maximum (0.9+) |
|---------|:-------------:|:------------------:|:--------------:|:--------------:|
| amazon-product-reviews-video-games | - | - | **MMR** | **MSD** |
| goodreads-rating | **SSD** | **MSD** | **MSD** | **MSD** |
| lastfm | **SSD** | **MSD** | **MSD** | **MSD** |
| ml-32m | **SSD** | **DPP** | **MSD** | - |

### Summary: Wins per Strategy by Region

| Region | SSD | DPP | MSD | MMR | Recommendation |
|--------|:---:|:---:|:---:|:---:|----------------|
| Low (0.3-0.5) | 3 | - | - | - | Use **SSD** for light diversification |
| Moderate (0.5-0.7) | - | 1 | 2 | - | Use **MSD** or **DPP** for moderate diversity |
| High (0.7-0.9) | - | - | 3 | 1 | Use **MSD** for heavy diversification |
| Maximum (0.9+) | - | - | 3 | - | Use **MSD** for maximum diversity |

## Diversity at 95% Relevance Retention

How much diversity (ILAD) can each strategy achieve while maintaining ≥95% of baseline nDCG?

| Dataset | MMR | MSD | DPP | SSD | Winner |
|---------|:---:|:---:|:---:|:---:|:------:|
| amazon-product-reviews-video-games | 0.851 | 0.947 | 0.866 | 0.832 | **MSD** |
| goodreads-rating | 0.427 | 0.724 | 0.673 | 0.462 | **MSD** |
| lastfm | 0.485 | 0.837 | 0.733 | 0.523 | **MSD** |
| ml-32m | 0.389 | 0.536 | 0.582 | 0.415 | **DPP** |

## Overall Strategy Ranking

Ranking by average F1 score (harmonic mean of normalized nDCG and ILAD):

| Rank | Strategy | Avg F1 | Interpretation |
|:----:|----------|:------:|----------------|
| 🥇 | **MSD** | 0.790 | Best diversity with acceptable relevance |
| 🥈 | **DPP** | 0.661 | Good balance of both metrics |
| 🥉 | **MMR** | 0.397 | Conservative, relevance-focused |
| 4. | **SSD** | 0.334 | Conservative, relevance-focused |

## Best F1 Score per Dataset

| Dataset | Strategy | λ | nDCG@10 | ILAD | F1 |
|---------|----------|--:|--------:|-----:|---:|
| amazon-product-reviews-video-games | MMR | 0.8 | 0.2557 | 0.935 | 0.697 |
| amazon-product-reviews-video-games | MSD | 0.4 | 0.2852 | 0.947 | 0.834 |
| amazon-product-reviews-video-games | DPP | 0.8 | 0.2982 | 0.866 | 0.593 |
| amazon-product-reviews-video-games | SSD | 0.8 | 0.2987 | 0.832 | 0.425 |
| goodreads-rating | MMR | 0.8 | 0.0251 | 0.553 | 0.476 |
| goodreads-rating | MSD | 0.6 | 0.0244 | 0.832 | 0.779 |
| goodreads-rating | DPP | 0.8 | 0.0272 | 0.673 | 0.683 |
| goodreads-rating | SSD | 0.8 | 0.0288 | 0.462 | 0.310 |
| lastfm | MMR | 0.8 | 0.1590 | 0.485 | 0.195 |
| lastfm | MSD | 0.6 | 0.1537 | 0.837 | 0.821 |
| lastfm | DPP | 0.8 | 0.1506 | 0.733 | 0.681 |
| lastfm | SSD | 0.8 | 0.1585 | 0.523 | 0.298 |
| ml-32m | MMR | 0.8 | 0.0606 | 0.389 | 0.219 |
| ml-32m | MSD | 0.6 | 0.0512 | 0.660 | 0.727 |
| ml-32m | DPP | 0.8 | 0.0616 | 0.582 | 0.688 |
| ml-32m | SSD | 0.8 | 0.0634 | 0.415 | 0.303 |

<details>
<summary>Full Results by Dataset (click to expand)</summary>

### amazon-product-reviews-video-games

| Strategy | λ | nDCG@10 | MRR | ILAD | Latency |
|----------|--:|--------:|----:|-----:|--------:|
| DPP | 0.0 | 0.2790 | 0.2403 | 0.774 | 0.03ms |
| DPP | 0.2 | 0.2904 | 0.2497 | 0.798 | 0.21ms |
| DPP | 0.4 | 0.2917 | 0.2506 | 0.806 | 0.20ms |
| DPP | 0.6 | 0.2937 | 0.2532 | 0.824 | 0.20ms |
| DPP | 0.8 | 0.2982 | 0.2576 | 0.866 | 0.21ms |
| DPP | 1.0 | 0.1711 | 0.1696 | 0.973 | 0.20ms |
| MMR | 0.0 | 0.2790 | 0.2403 | 0.774 | 0.20ms |
| MMR | 0.2 | 0.2893 | 0.2477 | 0.785 | 0.17ms |
| MMR | 0.4 | 0.2961 | 0.2546 | 0.806 | 0.16ms |
| MMR | 0.6 | 0.3037 | 0.2592 | 0.851 | 0.15ms |
| MMR | 0.8 | 0.2557 | 0.2287 | 0.935 | 0.16ms |
| MMR | 1.0 | 0.1727 | 0.1722 | 0.972 | 0.16ms |
| MSD | 0.0 | 0.2790 | 0.2403 | 0.774 | 0.17ms |
| MSD | 0.2 | 0.2916 | 0.2467 | 0.881 | 0.17ms |
| MSD | 0.4 | 0.2852 | 0.2428 | 0.947 | 0.16ms |
| MSD | 0.6 | 0.2511 | 0.2258 | 0.975 | 0.16ms |
| MSD | 0.8 | 0.1946 | 0.1864 | 0.986 | 0.17ms |
| MSD | 1.0 | 0.1648 | 0.1644 | 0.990 | 0.16ms |
| SSD | 0.0 | 0.2790 | 0.2403 | 0.774 | 0.01ms |
| SSD | 0.2 | 0.2814 | 0.2423 | 0.776 | 1.88ms |
| SSD | 0.4 | 0.2845 | 0.2451 | 0.782 | 1.80ms |
| SSD | 0.6 | 0.2911 | 0.2501 | 0.793 | 1.72ms |
| SSD | 0.8 | 0.2987 | 0.2575 | 0.832 | 1.64ms |
| SSD | 1.0 | 0.1644 | 0.1653 | 0.980 | 1.66ms |

### goodreads-rating

| Strategy | λ | nDCG@10 | MRR | ILAD | Latency |
|----------|--:|--------:|----:|-----:|--------:|
| DPP | 0.0 | 0.0280 | 0.0235 | 0.356 | 0.05ms |
| DPP | 0.2 | 0.0290 | 0.0236 | 0.414 | 0.39ms |
| DPP | 0.4 | 0.0286 | 0.0236 | 0.438 | 0.40ms |
| DPP | 0.6 | 0.0284 | 0.0234 | 0.490 | 0.35ms |
| DPP | 0.8 | 0.0272 | 0.0221 | 0.673 | 0.35ms |
| DPP | 1.0 | 0.0111 | 0.0114 | 0.879 | 0.35ms |
| MMR | 0.0 | 0.0280 | 0.0235 | 0.356 | 0.28ms |
| MMR | 0.2 | 0.0284 | 0.0236 | 0.367 | 0.24ms |
| MMR | 0.4 | 0.0288 | 0.0236 | 0.388 | 0.23ms |
| MMR | 0.6 | 0.0275 | 0.0229 | 0.427 | 0.24ms |
| MMR | 0.8 | 0.0251 | 0.0211 | 0.553 | 0.23ms |
| MMR | 1.0 | 0.0127 | 0.0125 | 0.863 | 0.23ms |
| MSD | 0.0 | 0.0280 | 0.0235 | 0.356 | 0.24ms |
| MSD | 0.2 | 0.0288 | 0.0234 | 0.572 | 0.24ms |
| MSD | 0.4 | 0.0277 | 0.0219 | 0.724 | 0.24ms |
| MSD | 0.6 | 0.0244 | 0.0198 | 0.832 | 0.24ms |
| MSD | 0.8 | 0.0182 | 0.0159 | 0.898 | 0.24ms |
| MSD | 1.0 | 0.0115 | 0.0113 | 0.927 | 0.24ms |
| SSD | 0.0 | 0.0280 | 0.0235 | 0.356 | 0.01ms |
| SSD | 0.2 | 0.0284 | 0.0238 | 0.360 | 3.68ms |
| SSD | 0.4 | 0.0284 | 0.0237 | 0.369 | 3.61ms |
| SSD | 0.6 | 0.0293 | 0.0236 | 0.387 | 3.53ms |
| SSD | 0.8 | 0.0288 | 0.0236 | 0.462 | 3.53ms |
| SSD | 1.0 | 0.0115 | 0.0116 | 0.888 | 3.55ms |

### lastfm

| Strategy | λ | nDCG@10 | MRR | ILAD | Latency |
|----------|--:|--------:|----:|-----:|--------:|
| DPP | 0.0 | 0.1582 | 0.1315 | 0.426 | 0.05ms |
| DPP | 0.2 | 0.1592 | 0.1326 | 0.471 | 0.40ms |
| DPP | 0.4 | 0.1593 | 0.1329 | 0.490 | 0.41ms |
| DPP | 0.6 | 0.1568 | 0.1320 | 0.542 | 0.36ms |
| DPP | 0.8 | 0.1506 | 0.1258 | 0.733 | 0.35ms |
| DPP | 1.0 | 0.0784 | 0.0782 | 0.947 | 0.35ms |
| MMR | 0.0 | 0.1582 | 0.1315 | 0.426 | 0.28ms |
| MMR | 0.2 | 0.1591 | 0.1319 | 0.429 | 0.24ms |
| MMR | 0.4 | 0.1597 | 0.1326 | 0.433 | 0.23ms |
| MMR | 0.6 | 0.1605 | 0.1334 | 0.443 | 0.23ms |
| MMR | 0.8 | 0.1590 | 0.1332 | 0.485 | 0.23ms |
| MMR | 1.0 | 0.0780 | 0.0781 | 0.942 | 0.24ms |
| MSD | 0.0 | 0.1582 | 0.1315 | 0.426 | 0.25ms |
| MSD | 0.2 | 0.1606 | 0.1330 | 0.506 | 0.24ms |
| MSD | 0.4 | 0.1583 | 0.1317 | 0.638 | 0.24ms |
| MSD | 0.6 | 0.1537 | 0.1261 | 0.837 | 0.24ms |
| MSD | 0.8 | 0.1270 | 0.1118 | 0.940 | 0.25ms |
| MSD | 1.0 | 0.0779 | 0.0777 | 0.977 | 0.24ms |
| SSD | 0.0 | 0.1582 | 0.1315 | 0.426 | 0.02ms |
| SSD | 0.2 | 0.1590 | 0.1324 | 0.429 | 3.25ms |
| SSD | 0.4 | 0.1600 | 0.1329 | 0.435 | 3.20ms |
| SSD | 0.6 | 0.1608 | 0.1336 | 0.449 | 3.12ms |
| SSD | 0.8 | 0.1585 | 0.1328 | 0.523 | 3.12ms |
| SSD | 1.0 | 0.0775 | 0.0776 | 0.958 | 3.09ms |

### ml-32m

| Strategy | λ | nDCG@10 | MRR | ILAD | Latency |
|----------|--:|--------:|----:|-----:|--------:|
| DPP | 0.0 | 0.0547 | 0.0422 | 0.331 | 0.04ms |
| DPP | 0.2 | 0.0614 | 0.0465 | 0.394 | 0.42ms |
| DPP | 0.4 | 0.0620 | 0.0471 | 0.418 | 0.37ms |
| DPP | 0.6 | 0.0629 | 0.0477 | 0.473 | 0.33ms |
| DPP | 0.8 | 0.0616 | 0.0462 | 0.582 | 0.32ms |
| DPP | 1.0 | 0.0151 | 0.0167 | 0.734 | 0.35ms |
| MMR | 0.0 | 0.0547 | 0.0422 | 0.331 | 0.27ms |
| MMR | 0.2 | 0.0553 | 0.0427 | 0.333 | 0.26ms |
| MMR | 0.4 | 0.0566 | 0.0435 | 0.338 | 0.25ms |
| MMR | 0.6 | 0.0575 | 0.0445 | 0.348 | 0.30ms |
| MMR | 0.8 | 0.0606 | 0.0445 | 0.389 | 0.23ms |
| MMR | 1.0 | 0.0156 | 0.0162 | 0.721 | 0.26ms |
| MSD | 0.0 | 0.0547 | 0.0422 | 0.331 | 0.23ms |
| MSD | 0.2 | 0.0567 | 0.0434 | 0.416 | 0.23ms |
| MSD | 0.4 | 0.0575 | 0.0426 | 0.536 | 0.24ms |
| MSD | 0.6 | 0.0512 | 0.0392 | 0.660 | 0.24ms |
| MSD | 0.8 | 0.0411 | 0.0324 | 0.752 | 0.24ms |
| MSD | 1.0 | 0.0132 | 0.0132 | 0.801 | 0.23ms |
| SSD | 0.0 | 0.0547 | 0.0422 | 0.331 | 0.01ms |
| SSD | 0.2 | 0.0565 | 0.0431 | 0.335 | 3.45ms |
| SSD | 0.4 | 0.0579 | 0.0445 | 0.341 | 3.38ms |
| SSD | 0.6 | 0.0612 | 0.0456 | 0.354 | 3.25ms |
| SSD | 0.8 | 0.0634 | 0.0472 | 0.415 | 3.25ms |
| SSD | 1.0 | 0.0149 | 0.0166 | 0.736 | 3.21ms |

</details>

## Key Insights

1. **SSD excels at light diversification** - Best nDCG when ILAD is 0.3-0.5
2. **MSD dominates high diversity regions** - Best choice when ILAD > 0.7
3. **DPP is a good middle-ground** - Competitive across multiple regions
4. **MMR is fastest but least effective** - Rarely wins any diversity region

### Choosing the Right Strategy

| Your Goal | Best Strategy | λ | Why |
|-----------|---------------|---|-----|
| Light diversification | **SSD** | 0.6-0.8 | Highest nDCG at ILAD 0.3-0.5 |
| Moderate diversification | **DPP** or **MSD** | 0.4-0.6 | Good balance at ILAD 0.5-0.7 |
| Heavy diversification | **MSD** | 0.4-0.6 | Best at ILAD 0.7-0.9 |
| Maximum diversity | **MSD** | 0.6-0.8 | Dominates at ILAD > 0.9 |
| Speed-critical | **MMR** | 0.6-0.8 | O(k·n) complexity |

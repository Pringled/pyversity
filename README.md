# Pyversity — Diversified Re‑Ranking for Retrieval

Pyversity is a small, fast library for diversifying retrieval results.
Retrieval systems often return highly similar items. Pyversity efficiently re-ranks these results to encourage diversity, surfacing items that remain relevant but less redundant.

It implements several popular strategies such as MMR, MSD, DPP, and Cover with a clear, unified API. More information about the supported strategies can be found in the [supported strategies section](#supported-strategies).


## Quickstart

Install `pyversity` with:

```bash
pip install pyversity
```

Diversify retrieval results:
```python
import numpy as np
from pyversity import diversify, Strategy

# Define embeddings and scores
embeddings  = np.random.randn(100, 256).astype(np.float32)
scores  = np.random.rand(100).astype(np.float32)

# Diversify with with a chosen strategy (in this case MMR)
diversified_result = diversify(
    embeddings=embeddings,
    scores=scores,
    k=10,
    strategy=Strategy.MMR,
)
# Get the indicices of the diversified result
diversified_indices = diversified_result.indices
```



## Supported Strategies

The following table describes the supported strategies, how they work, their time complexity, and when to use them.

| Strategy                              | What It Does                                                                                   | Time Complexity           | When to Use                                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| **MMR** (Maximum Marginal Relevance)  | Keeps the most relevant items while down-weighting those too similar to what’s already picked. | **O(k · n · d)**          | Best **default**. Fast, simple, and works well when you just want to avoid near-duplicates.    |
| **MSD** (Max Sum of Distances)        | Prefers items that are both relevant and far from *all* previous selections.                   | **O(k · n · d)**          | Use when you want stronger spread, i.e. results that cover a wider range of topics or styles.      |
| **DPP** (Determinantal Point Process) | Samples diverse yet relevant items using probabilistic “repulsion.”                            | **O(k · n · d + n · k²)** | Ideal when you want to eliminate redundancy or ensure diversity is built-in to selection.  |
| **COVER** (Facility-Location)         | Ensures selected items collectively represent the full dataset’s structure.                    | **O(k · n²)**             | Great for topic coverage or clustering scenarios, but slower for large `n`. |

## References

The implementations in this package are based on the following research papers:

- **MMR**: Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. [Link](https://dl.acm.org/doi/pdf/10.1145/290941.291025)

- **MSD**: Borodin, A., Lee, H. C., & Ye, Y. (2012). Max-sum diversification, monotone submodular functions and dynamic updates. [Link](https://arxiv.org/pdf/1203.6397)

- **COVER**: Puthiya Parambath, S. A., Usunier, N., & Grandvalet, Y. (2016). A coverage-based approach to recommendation diversity on similarity graph. [Link](https://dl.acm.org/doi/10.1145/2959100.2959149)

- **DPP**: Kulesza, A., & Taskar, B. (2012). Determinantal Point Processes for Machine Learning. [Link](https://arxiv.org/pdf/1207.6083)

- **DPP (efficient greedy implementation)**: Chen, L., Zhang, G., & Zhou, H. (2018). Fast greedy MAP inference for determinantal point process to improve recommendation diversity.
[Link](https://arxiv.org/pdf/1709.05135)

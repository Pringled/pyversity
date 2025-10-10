# Pyversity — Diversified Re‑Ranking for Retrieval

**Pyversity** is a small, fast library for *diversifying* retrieval results.
Given a set of vector embeddings and relevance scores, Pyversity selects a subset that balances **relevance** and **diversity**.

It implements several standard strategies with clear, well‑documented APIs:

- **MMR** — Maximum Marginal Relevance
- **MSD** — Max Sum of (pairwise) Distances
- **DPP** — Determinantal Point Processes
- **COVER** — Facility‑location/coverage with concave gains

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
diversified_indices, diversified_scores = diversify(
    embeddings=embeddings,
    scores=scores,
    k=10,
    strategy=Strategy.MMR,
)
```


## Supported Strategies
| Strategy                              | What It Does                                                                                   | Time Complexity           | When to Use                                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| **MMR** (Maximum Marginal Relevance)  | Keeps the most relevant items while down-weighting those too similar to what’s already picked. | **O(k · n · d)**          | Best **default**. Fast, simple, and works well when you just want to avoid near-duplicates.    |
| **MSD** (Max Sum of Distances)        | Prefers items that are both relevant and far from *all* previous selections.                   | **O(k · n · d)**          | Use when you want stronger spread — results that cover a wider range of topics or styles.      |
| **COVER** (Facility-Location)         | Ensures selected items collectively represent the full dataset’s structure.                    | **O(k · n²)**             | Great for **topic coverage** or clustering scenarios. Higher quality, but slower on large `n`. |
| **DPP** (Determinantal Point Process) | Samples diverse yet relevant items using probabilistic “repulsion.”                            | **O(k · n · d + n · k²)** | Ideal when you want to **eliminate redundancy** or ensure diversity is built-in to selection.  |

## References

The implementations in this package are based on the following research papers:

- **MMR**: Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’98), pp. 335–336. Association for Computing Machinery, Melbourne, Australia. https://doi.org/10.1145/290941.291025

- **MSD**: Borodin, A., Lee, H. C., & Ye, Y. (2012). Max-sum diversification, monotone submodular functions and dynamic updates. In Proceedings of the 31st ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems (PODS ’12), pp. 155–166. Association for Computing Machinery, Scottsdale, Arizona, USA. https://doi.org/10.1145/2213556.2213580

- **COVER**: Puthiya Parambath, S. A., Usunier, N., & Grandvalet, Y. (2016). A coverage-based approach to recommendation diversity on similarity graph. In Proceedings of the 10th ACM Conference on Recommender Systems (RecSys ’16), pp. 15–22. Association for Computing Machinery, Boston, Massachusetts, USA. https://doi.org/10.1145/2959100.2959149

- **DPP**: Kulesza, A., & Taskar, B. (2012). Determinantal Point Processes for Machine Learning. Foundations and Trends in Machine Learning, 5(2–3), 123–286. https://api.semanticscholar.org/CorpusID:51975610

- **DPP (efficient greedy implementation)**: Chen, L., Zhang, G., & Zhou, H. (2018). Fast greedy MAP inference for determinantal point process to improve recommendation diversity. In Proceedings of the 32nd International Conference on Neural Information Processing Systems (NIPS ’18), pp. 5627–5638. Curran Associates Inc., Montréal, Canada.

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

# Random embeddings and scores
embeddings  = np.random.randn(100, 256).astype(np.float32)
scores  = np.random.rand(100).astype(np.float32)

# Rerank with with a chosen strategy (in this case MMR)
diversified_indices, scores = diversify(
    Strategy.MMR,
    embeddings,
    scores,
    k=10,
)
```


## Supported Strategies
| Strategy                              | What It Does                                                                                   | Time Complexity           | When to Use                                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| **MMR** (Maximum Marginal Relevance)  | Keeps the most relevant items while down-weighting those too similar to what’s already picked. | **O(k · n · d)**          | Best **default**. Fast, simple, and works well when you just want to avoid near-duplicates.    |
| **MSD** (Max Sum of Distances)        | Prefers items that are both relevant and far from *all* previous selections.                   | **O(k · n · d)**          | Use when you want stronger spread — results that cover a wider range of topics or styles.      |
| **COVER** (Facility-Location)         | Ensures selected items collectively represent the full dataset’s structure.                    | **O(k · n²)**             | Great for **topic coverage** or clustering scenarios. Higher quality, but slower on large `n`. |
| **DPP** (Determinantal Point Process) | Samples diverse yet relevant items using probabilistic “repulsion.”                            | **O(k · n · d + n · k²)** | Ideal when you want to **eliminate redundancy** or ensure diversity is built-in to selection.  |

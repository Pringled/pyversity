
<h2 align="center">
  <img width="35%" alt="Pyversity logo" src="assets/images/pyversity_logo.png"><br/>
  Fast Diversification for Retrieval
</h2>

<div align="center">

[Quickstart](#quickstart) •
[Supported Strategies](#supported-strategies) •
[Motivation](#motivation)

</div>

Pyversity is a fast, lightweight library for diversifying retrieval results.
Retrieval systems often return highly similar items. Pyversity efficiently re-ranks these results to encourage diversity, surfacing items that remain relevant but less redundant.

It implements several popular diversification strategies such as MMR, MSD, DPP, and Cover with a clear, unified API. More information about the supported strategies can be found in the [supported strategies section](#supported-strategies). The only dependency is NumPy, making the package very lightweight.


## Quickstart

Install `pyversity` with:

```bash
pip install pyversity
```

Diversify retrieval results:
```python
import numpy as np
from pyversity import diversify, Strategy

# Define embeddings and scores (e.g. cosine similarities of a query result)
embeddings = np.random.randn(100, 256)
scores = np.random.rand(100)

# Diversify the result
diversified_result = diversify(
    embeddings=embeddings,
    scores=scores,
    k=10, # Number of items to select
    strategy=Strategy.MMR, # Diversification strategy to use
    diversity=0.5 # Diversity parameter (higher values prioritize diversity)
)

# Get the indices of the diversified result
diversified_indices = diversified_result.indices
```

The returned `DiversificationResult` can be used to access the diversified `indices`, as well as the `selection_scores` of the selected strategy and other useful info. The strategies are extremely fast and scalable: this example runs in milliseconds.

The `diversity` parameter tunes the trade-off between relevance and diversity: 0.0 focuses purely on relevance (no diversification), while 1.0 maximizes diversity, potentially at the cost of relevance.

## Supported Strategies

The following table describes the supported strategies, how they work, their time complexity, and when to use them. The papers linked in the [references](#references) section provide more in-depth information on the strengths/weaknesses of the supported strategies.

| Strategy                              | What It Does                                                                                   | Time Complexity           | When to Use                                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| **MMR** (Maximal Marginal Relevance)  | Keeps the most relevant items while down-weighting those too similar to what’s already picked. | **O(k · n · d)**          | Good default. Fast, simple, and works well when you just want to avoid near-duplicates.    |
| **MSD** (Max Sum of Distances)        | Prefers items that are both relevant and far from *all* previous selections.                   | **O(k · n · d)**          | Use when you want stronger spread, i.e. results that cover a wider range of topics or styles.      |
| **DPP** (Determinantal Point Process) | Samples diverse yet relevant items using probabilistic “repulsion.”                            | **O(k · n · d + n · k²)** | Ideal when you want to eliminate redundancy or ensure diversity is built-in to selection.  |
| **COVER** (Facility-Location)         | Ensures selected items collectively represent the full dataset’s structure.                    | **O(k · n²)**             | Great for topic coverage or clustering scenarios, but slower for large `n`. |


## Motivation

Traditional retrieval systems rank results purely by relevance (how closely each item matches the query). While effective, this can lead to redundancy: top results often look nearly identical, which can create a poor user experience.

Diversification techniques like MMR, MSD, COVER, and DPP help balance relevance and variety.
Each new item is chosen not only because it’s relevant, but also because it adds new information that wasn’t already covered by earlier results.

This improves exploration, user satisfaction, and coverage across many domains, for example:

- E-commerce: Show different product styles, not multiple copies of the same black pants.
- News search: Highlight articles from different outlets or viewpoints.
- Academic retrieval: Surface papers from different subfields or methods.
- RAG / LLM contexts: Avoid feeding the model near-duplicate passages.

## References

The implementations in this package are based on the following research papers:

- **MMR**: Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. [Link](https://dl.acm.org/doi/pdf/10.1145/290941.291025)

- **MSD**: Borodin, A., Lee, H. C., & Ye, Y. (2012). Max-sum diversification, monotone submodular functions and dynamic updates. [Link](https://arxiv.org/pdf/1203.6397)

- **COVER**: Puthiya Parambath, S. A., Usunier, N., & Grandvalet, Y. (2016). A coverage-based approach to recommendation diversity on similarity graph. [Link](https://dl.acm.org/doi/10.1145/2959100.2959149)

- **DPP**: Kulesza, A., & Taskar, B. (2012). Determinantal Point Processes for Machine Learning. [Link](https://arxiv.org/pdf/1207.6083)

- **DPP (efficient greedy implementation)**: Chen, L., Zhang, G., & Zhou, H. (2018). Fast greedy MAP inference for determinantal point process to improve recommendation diversity.
[Link](https://arxiv.org/pdf/1709.05135)

## Author

Thomas van Dongen

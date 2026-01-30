# Golden-value regression tests. Regenerate with: python tests/test_regression.py
import numpy as np
import pytest
from pyversity import Metric, Strategy, diversify

# Dataset 1: 20 items, 8 dims, k=5
_RNG1 = np.random.default_rng(42)
_RAW1 = _RNG1.standard_normal((20, 8)).astype(np.float32)
EMBEDDINGS_1 = _RAW1 / np.maximum(np.linalg.norm(_RAW1, axis=1, keepdims=True), 1e-7)
SCORES_1 = _RNG1.uniform(0.1, 1.0, size=20).astype(np.float32)
K_1 = 5

_RAW1_RECENT = _RNG1.standard_normal((3, 8)).astype(np.float32)
RECENT_1 = _RAW1_RECENT / np.maximum(np.linalg.norm(_RAW1_RECENT, axis=1, keepdims=True), 1e-7)

# Dataset 2: 50 items (45 + 5 near-duplicates), 32 dims, k=10
_RNG2 = np.random.default_rng(123)
_RAW2_BASE = _RNG2.standard_normal((45, 32)).astype(np.float32)
_RAW2_DUPES = _RAW2_BASE[:5] + _RNG2.normal(0, 0.01, (5, 32)).astype(np.float32)
_RAW2 = np.vstack([_RAW2_BASE, _RAW2_DUPES])
EMBEDDINGS_2 = _RAW2 / np.maximum(np.linalg.norm(_RAW2, axis=1, keepdims=True), 1e-7)
SCORES_2 = _RNG2.uniform(0.1, 1.0, size=50).astype(np.float32)
K_2 = 10

_RAW2_RECENT = _RNG2.standard_normal((4, 32)).astype(np.float32)
RECENT_2 = _RAW2_RECENT / np.maximum(np.linalg.norm(_RAW2_RECENT, axis=1, keepdims=True), 1e-7)

_DATASETS = {
    1: (EMBEDDINGS_1, SCORES_1, K_1),
    2: (EMBEDDINGS_2, SCORES_2, K_2),
}

# Golden test cases
CASES: list[dict] = [
    # Dataset 1, default parameters
    dict(
        id="d1-mmr-0.0",
        strategy=Strategy.MMR,
        diversity=0.0,
        dataset=1,
        indices=[14, 6, 10, 18, 15],
        scores=[0.993138, 0.9576095, 0.9424392, 0.904102, 0.9025095],
    ),
    dict(
        id="d1-mmr-0.5",
        strategy=Strategy.MMR,
        diversity=0.5,
        dataset=1,
        indices=[14, 6, 16, 4, 15],
        scores=[0.496569, 0.4788047, 0.3501282, 0.3043389, 0.2549736],
    ),
    dict(
        id="d1-mmr-1.0",
        strategy=Strategy.MMR,
        diversity=1.0,
        dataset=1,
        indices=[14, 4, 5, 8, 10],
        scores=[0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    dict(
        id="d1-msd-0.0",
        strategy=Strategy.MSD,
        diversity=0.0,
        dataset=1,
        indices=[14, 6, 10, 18, 15],
        scores=[0.993138, 0.9576095, 0.9424392, 0.904102, 0.9025095],
    ),
    dict(
        id="d1-msd-0.5",
        strategy=Strategy.MSD,
        diversity=0.5,
        dataset=1,
        indices=[14, 17, 10, 18, 5],
        scores=[0.496569, 1.2563956, 1.6436498, 2.0642159, 2.5259635],
    ),
    dict(
        id="d1-msd-1.0",
        strategy=Strategy.MSD,
        diversity=1.0,
        dataset=1,
        indices=[14, 17, 8, 7, 12],
        scores=[0.0, 1.6110779, 2.372129, 3.538434, 4.6836829],
    ),
    dict(
        id="d1-dpp-0.0",
        strategy=Strategy.DPP,
        diversity=0.0,
        dataset=1,
        indices=[14, 6, 10, 18, 15],
        scores=[0.993138, 0.9576095, 0.9424392, 0.904102, 0.9025095],
    ),
    dict(
        id="d1-dpp-0.5",
        strategy=Strategy.DPP,
        diversity=0.5,
        dataset=1,
        indices=[14, 10, 18, 4, 6],
        scores=[3.8320513, 3.1946325, 2.278928, 2.0403609, 1.7162278],
    ),
    dict(
        id="d1-dpp-1.0",
        strategy=Strategy.DPP,
        diversity=1.0,
        dataset=1,
        indices=[0, 3, 13, 19, 12],
        scores=[1.0000001, 0.9999955, 0.9922758, 0.9399173, 0.8916375],
    ),
    dict(
        id="d1-cover-0.0",
        strategy=Strategy.COVER,
        diversity=0.0,
        dataset=1,
        indices=[14, 6, 10, 18, 15],
        scores=[0.993138, 0.9576095, 0.9424392, 0.904102, 0.9025095],
    ),
    dict(
        id="d1-cover-0.5",
        strategy=Strategy.COVER,
        diversity=0.5,
        dataset=1,
        indices=[1, 6, 18, 3, 15],
        scores=[4.2265797, 2.5809641, 2.2656803, 1.8373549, 1.4752011],
    ),
    dict(
        id="d1-cover-1.0",
        strategy=Strategy.COVER,
        diversity=1.0,
        dataset=1,
        indices=[1, 6, 18, 3, 0],
        scores=[7.8537917, 4.2043185, 3.6272588, 2.827899, 2.24547],
    ),
    dict(
        id="d1-ssd-0.0",
        strategy=Strategy.SSD,
        diversity=0.0,
        dataset=1,
        indices=[14, 6, 10, 18, 15],
        scores=[0.993138, 0.9576095, 0.9424392, 0.904102, 0.9025095],
    ),
    dict(
        id="d1-ssd-0.5",
        strategy=Strategy.SSD,
        diversity=0.5,
        dataset=1,
        indices=[14, 6, 18, 10, 17],
        scores=[1.3788071, 1.281781, 1.0446954, 1.0005223, 0.9349319],
    ),
    dict(
        id="d1-ssd-1.0",
        strategy=Strategy.SSD,
        diversity=1.0,
        dataset=1,
        indices=[14, 17, 10, 5, 8],
        scores=[1.4142135, 1.3872166, 1.1854348, 1.1495396, 1.0392091],
    ),
    dict(
        id="d1-ssd-recent-0.5",
        strategy=Strategy.SSD,
        diversity=0.5,
        dataset=1,
        kwargs={"recent_embeddings": RECENT_1},
        indices=[14, 6, 17, 18, 15],
        scores=[1.2624965, 1.1179161, 0.9618138, 0.8816395, 0.6654232],
    ),
    # Dataset 1, non-default parameters
    dict(
        id="d1-mmr-dot-0.5",
        strategy=Strategy.MMR,
        diversity=0.5,
        dataset=1,
        kwargs={"metric": Metric.DOT, "normalize": False},
        indices=[14, 6, 16, 4, 15],
        scores=[0.496569, 0.4788047, 0.3501282, 0.3043389, 0.2549736],
    ),
    dict(
        id="d1-msd-dot-0.5",
        strategy=Strategy.MSD,
        diversity=0.5,
        dataset=1,
        kwargs={"metric": Metric.DOT, "normalize": False},
        indices=[14, 17, 10, 18, 5],
        scores=[0.496569, 0.7563956, 0.6436498, 0.564216, 0.5259635],
    ),
    dict(
        id="d1-cover-gamma0.8-0.5",
        strategy=Strategy.COVER,
        diversity=0.5,
        dataset=1,
        kwargs={"gamma": 0.8},
        indices=[15, 6, 14, 3, 18],
        scores=[3.339906, 2.8881586, 2.5098512, 2.31229, 2.2262282],
    ),
    dict(
        id="d1-dpp-scale2-0.5",
        strategy=Strategy.DPP,
        diversity=0.5,
        dataset=1,
        kwargs={"scale": 2.0},
        indices=[14, 10, 6, 18, 4],
        scores=[14.684617, 10.2230501, 6.7176108, 5.8371487, 4.0808001],
    ),
    dict(
        id="d1-ssd-raw-0.5",
        strategy=Strategy.SSD,
        diversity=0.5,
        dataset=1,
        kwargs={"normalize": False, "append_bias": False, "normalize_scores": False},
        indices=[14, 10, 4, 18, 6],
        scores=[0.996569, 0.9707946, 0.9087705, 0.89896, 0.835229],
    ),
    dict(
        id="d1-ssd-window3-0.5",
        strategy=Strategy.SSD,
        diversity=0.5,
        dataset=1,
        kwargs={"window": 3},
        indices=[14, 6, 18, 10, 3],
        scores=[1.3788071, 1.281781, 1.0446954, 1.0005223, 1.0854802],
    ),
    dict(
        id="d2-mmr-0.5",
        strategy=Strategy.MMR,
        diversity=0.5,
        dataset=2,
        indices=[34, 44, 0, 8, 11, 2, 33, 21, 35, 49],
        scores=[
            0.4982942,
            0.4752265,
            0.4423259,
            0.4213459,
            0.4192513,
            0.4143822,
            0.3673249,
            0.3629451,
            0.3548062,
            0.3130703,
        ],
    ),
    dict(
        id="d2-msd-0.5",
        strategy=Strategy.MSD,
        diversity=0.5,
        dataset=2,
        indices=[34, 44, 8, 26, 0, 35, 24, 33, 48, 2],
        scores=[
            0.4982942,
            1.0318062,
            1.5670512,
            2.1662273,
            2.6055198,
            3.1134424,
            3.6113653,
            4.1183109,
            4.5576282,
            5.0157132,
        ],
    ),
    dict(
        id="d2-dpp-0.5",
        strategy=Strategy.DPP,
        diversity=0.5,
        dataset=2,
        indices=[34, 11, 44, 41, 0, 14, 21, 2, 35, 8],
        scores=[
            5.1772084,
            4.6469154,
            4.3583627,
            4.0794702,
            3.3010585,
            3.0233638,
            2.8123801,
            2.6220398,
            2.5165148,
            2.2487965,
        ],
    ),
    dict(
        id="d2-cover-0.5",
        strategy=Strategy.COVER,
        diversity=0.5,
        dataset=2,
        indices=[21, 7, 28, 1, 23, 48, 2, 41, 15, 49],
        scores=[
            6.4557028,
            4.119772,
            3.1374972,
            2.4369617,
            2.2468584,
            2.0424407,
            1.9487213,
            1.7973837,
            1.756186,
            1.6591004,
        ],
    ),
    dict(
        id="d2-ssd-0.5",
        strategy=Strategy.SSD,
        diversity=0.5,
        dataset=2,
        indices=[34, 44, 11, 41, 0, 2, 14, 21, 8, 35],
        scores=[
            1.52924,
            1.376298,
            1.3285697,
            1.2422681,
            1.2112706,
            1.1682143,
            1.1040297,
            1.0666059,
            1.0444114,
            1.0130265,
        ],
    ),
    dict(
        id="d2-ssd-recent-0.5",
        strategy=Strategy.SSD,
        diversity=0.5,
        dataset=2,
        kwargs={"recent_embeddings": RECENT_2},
        indices=[34, 11, 44, 41, 2, 0, 21, 14, 15, 8],
        scores=[
            1.3352783,
            1.2505096,
            1.2227125,
            1.1950345,
            1.1083739,
            1.0967314,
            1.052088,
            1.0184829,
            0.9867558,
            0.975732,
        ],
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_golden_values(case: dict) -> None:
    """Assert that strategy outputs match recorded golden values."""
    embeddings, scores, k = _DATASETS[case["dataset"]]
    kwargs = case.get("kwargs", {})

    result = diversify(embeddings, scores, k, strategy=case["strategy"], diversity=case["diversity"], **kwargs)

    assert np.array_equal(
        result.indices,
        np.array(case["indices"]),
    ), f"Indices mismatch for {case['id']}"

    assert np.allclose(
        result.selection_scores,
        np.array(case["scores"], dtype=np.float32),
        atol=1e-6,
    ), f"Scores mismatch for {case['id']}"


if __name__ == "__main__":
    for case in CASES:
        embeddings, scores, k = _DATASETS[case["dataset"]]
        kwargs = case.get("kwargs", {})
        r = diversify(embeddings, scores, k, strategy=case["strategy"], diversity=case["diversity"], **kwargs)
        print(  # noqa: T201
            f"{case['id']}:\n"
            f"    indices={r.indices.tolist()},\n"
            f"    scores={[round(float(s), 7) for s in r.selection_scores]},"
        )

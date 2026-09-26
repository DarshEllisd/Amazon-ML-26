import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.assign import matches, tune_tau  # noqa: E402


def scored(rows):
    return pl.DataFrame(rows, schema={"s1_id": pl.Int64, "m_src": pl.Int8, "m_id": pl.Int64, "p": pl.Float64},
                        orient="row")


def test_each_record_goes_to_its_best_entity_only():
    s = scored([(1, 2, 10, 0.9), (2, 2, 10, 0.8), (2, 3, 11, 0.7), (1, 3, 12, 0.3)])
    got = sorted(matches(s, 0.5).rows())
    assert got == [(1, 2, 10), (2, 3, 11)]


def test_tune_tau_prefers_precision():
    s = scored([(1, 2, 10, 0.9), (1, 2, 11, 0.4)])
    gt = pl.DataFrame({"s1_id": [1], "m_src": [2], "m_id": [10]}, schema={"s1_id": pl.Int64, "m_src": pl.Int8, "m_id": pl.Int64})
    tau, curve = tune_tau(s, gt, pl.DataFrame({"s1_id": [1]}))
    assert 0.4 < tau <= 0.9 and max(f for _, f in curve) == 1.0

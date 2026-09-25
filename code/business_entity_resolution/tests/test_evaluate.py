import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.evaluate import blocking_report, f05, macro_f05, per_entity_scores  # noqa: E402


def pairs(rows):
    return pl.DataFrame(rows, schema={"s1_id": pl.Int64, "m_src": pl.Int8, "m_id": pl.Int64}, orient="row")


def test_readme_example():
    # predict [S2-47, S2-193, S3-812], truth [S2-47, S3-812] -> 0.714
    assert f05(2, 3, 2) == pytest.approx(0.7142857, abs=1e-6)
    pred = pairs([(1, 2, 47), (1, 2, 193), (1, 3, 812)])
    gt = pairs([(1, 2, 47), (1, 3, 812)])
    ents = pl.DataFrame({"s1_id": [1]})
    assert macro_f05(pred, gt, ents) == pytest.approx(0.7142857, abs=1e-6)


def test_singletons_and_misses():
    gt = pairs([(1, 2, 10), (2, 3, 20)])
    ents = pl.DataFrame({"s1_id": [1, 2, 3, 4]})  # 3 and 4 are singletons
    pred = pairs([
        (1, 2, 10),            # perfect -> 1
        (2, 2, 20),            # wrong source, same numeric id -> 0
        (4, 2, 99),            # singleton with a prediction -> 0
    ])                         # entity 3: singleton, empty -> 1
    s = per_entity_scores(pred, gt, ents).sort("s1_id")
    assert s["f"].to_list() == [1.0, 0.0, 1.0, 0.0]
    assert macro_f05(pred, gt, ents) == 0.5


def test_duplicate_predictions_ignored_and_outside_entities_dropped():
    gt = pairs([(1, 2, 10)])
    ents = pl.DataFrame({"s1_id": [1]})
    pred = pairs([(1, 2, 10), (1, 2, 10), (7, 2, 10)])
    assert macro_f05(pred, gt, ents) == 1.0


def test_blocking_report():
    gt = pairs([(1, 2, 10), (1, 3, 11), (2, 2, 12)])
    ents = pl.DataFrame({"s1_id": [1, 2, 3]})
    cand = pairs([(1, 2, 10), (1, 2, 50), (2, 2, 60), (3, 3, 70)])
    r = blocking_report(cand, gt, ents)
    assert r["pair_recall"] == pytest.approx(1 / 3)
    assert r["candidate_pairs"] == 4
    # entity1 oracle picks {10}: p=1, r=.5 -> .8333; entity2 -> 0; entity3 singleton -> 1
    assert r["oracle_f05"] == pytest.approx((1.25 * 0.5 / (0.25 + 0.5) + 0 + 1) / 3)

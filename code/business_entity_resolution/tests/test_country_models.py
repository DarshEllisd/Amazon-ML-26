import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_pipeline as rp  # noqa: E402
from ber import rerank  # noqa: E402


def test_country_models_parsing():
    assert rp.country_models("France=final_e5:e5-small") == {"France": ("final_e5", "e5-small")}
    assert rp.country_models(" France = m1 , US=m2:bge-m3 ") == {"France": ("m1", None), "US": ("m2", "bge-m3")}
    assert rp.country_models("") == {}


def test_rules_and_bands_per_reranker_kind():
    assert rp.FINAL_RULES["bge-m3"]["extra"] == 0.5 and rp.FINAL_RULES["e5-small"]["extra"] == 1.0
    os.environ.pop("BER_RERANK_ALL", None)
    assert rerank.rerank_all("bge-m3") and not rerank.rerank_all("e5-small")
    assert rerank.model_dir("e5-small").name.startswith("reranker") and "bge" not in rerank.model_dir("e5-small").name
    assert rerank.model_dir("bge-m3").name.endswith("_bge-m3")

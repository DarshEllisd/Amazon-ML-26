import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.topk import rowwise_dot, sparse_topk  # noqa: E402


def test_topk_matches_brute_force():
    rng = np.random.default_rng(0)
    Q = normalize(sp.random(300, 50, density=0.1, format="csr", random_state=1, dtype=np.float32))
    X = normalize(sp.random(200, 50, density=0.1, format="csr", random_state=2, dtype=np.float32))
    q, x, v = sparse_topk(Q, X, k=5, chunk=64, workers=4)
    dense = (Q @ X.T).toarray()
    for i in rng.choice(300, 40, replace=False):
        got = sorted(v[q == i], reverse=True)
        nz = np.sort(dense[i][dense[i] > 0])[::-1][:5]
        assert np.allclose(got, nz, atol=1e-6)
        assert np.allclose(dense[i, x[q == i]], v[q == i], atol=1e-6)


def test_rowwise_dot():
    A = normalize(sp.random(20, 10, density=0.5, format="csr", random_state=3))
    B = normalize(sp.random(20, 10, density=0.5, format="csr", random_state=4))
    assert np.allclose(rowwise_dot(A, B), (A.multiply(B)).sum(1).A.ravel(), atol=1e-6)


def test_gpu_topk_agrees_with_cpu():
    import pytest
    from ber import topk as T
    if not T._CUDA:
        pytest.skip("no CUDA")
    Q = normalize(sp.random(500, 80, density=0.1, format="csr", random_state=5, dtype=np.float32))
    X = normalize(sp.random(300, 80, density=0.1, format="csr", random_state=6, dtype=np.float32))
    a = T.sparse_topk(Q, X, 4)
    b = T.gpu_topk(Q, X, 4)
    sa = {(int(q), int(x)) for q, x in zip(a[0], a[1])}
    sb = {(int(q), int(x)) for q, x in zip(b[0], b[1])}
    assert len(sa ^ sb) <= 0.01 * len(sa)       # ties may break differently
    assert np.isclose(np.sort(a[2]).sum(), np.sort(b[2]).sum(), rtol=1e-4)

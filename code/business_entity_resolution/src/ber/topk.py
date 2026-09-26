"""Top-k cosine neighbours between two L2-normalised sparse matrices.

`topk` runs on the GPU (torch sparse CSR x dense) when CUDA is available and
falls back to the chunked scipy implementation `sparse_topk` otherwise; both
return the same neighbours up to ties.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import torch
    _CUDA = torch.cuda.is_available()
except ImportError:  # CPU-only environment
    torch, _CUDA = None, False

GPU_CELLS = 150_000_000  # budget for each dense block on the GPU (index x chunk, vocab x chunk)

_EMPTY = (np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.float32))


def _rows_topk(prod, k):
    """Per-row top-k of a CSR product -> (row, col, score) arrays."""
    indptr, indices, data = prod.indptr, prod.indices, prod.data
    counts = np.diff(indptr)
    rows, cols, vals = [], [], []
    small = np.flatnonzero((counts > 0) & (counts <= k))
    if small.size:
        # rows with <= k nonzeros keep all of them
        sel = np.concatenate([np.arange(indptr[r], indptr[r + 1]) for r in small])
        rows.append(np.repeat(small, counts[small]))
        cols.append(indices[sel])
        vals.append(data[sel])
    for r in np.flatnonzero(counts > k):
        a, b = indptr[r], indptr[r + 1]
        top = np.argpartition(-data[a:b], k - 1)[:k] + a
        rows.append(np.full(k, r, dtype=np.int64))
        cols.append(indices[top])
        vals.append(data[top])
    if not rows:
        return _EMPTY
    return (np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int32),
            np.concatenate(vals).astype(np.float32))


def sparse_topk(Q, X, k, chunk=1000, min_score=0.0, workers=8):
    """For every row of Q, the k rows of X with the highest dot product (> min_score).

    Q, X: CSR matrices with the same column space. Returns (q_idx, x_idx, score).
    """
    Xt = X.T.tocsr()
    starts = range(0, Q.shape[0], chunk)

    def run(s):
        prod = Q[s:s + chunk] @ Xt
        if min_score > 0:
            prod.data[prod.data <= min_score] = 0
            prod.eliminate_zeros()
        r, c, v = _rows_topk(prod.tocsr(), k)
        return r + s, c, v

    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(run, starts))
    if not parts:
        return _EMPTY
    return tuple(np.concatenate(x) for x in zip(*parts))


def _to_gpu_csr(m, dev):
    return torch.sparse_csr_tensor(
        torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
        torch.from_numpy(m.data.astype(np.float32)), size=m.shape, device=dev)


def gpu_topk(Q, X, k, min_score=0.0):
    """Same contract as sparse_topk, computed on the GPU."""
    if Q.shape[0] == 0 or X.shape[0] == 0 or X.nnz == 0:
        return _EMPTY
    X = X.tocsr()
    # keep only the columns the index uses: other query columns contribute nothing
    used = np.unique(X.indices)
    Xc = X[:, used].tocsr()
    dev = torch.device("cuda")
    Xt = _to_gpu_csr(Xc, dev)
    n, v = Xc.shape
    kk = min(k, n)
    chunk = int(max(64, min(4096, GPU_CELLS // max(n, v))))
    QT = Q.tocsr()[:, used].T.tocsr()                      # v x n_query
    rows, cols, vals = [], [], []
    for s in range(0, QT.shape[1], chunk):
        q = QT[:, s:s + chunk].tocsr()
        dense = _to_gpu_csr(q, dev).to_dense()              # v x c, densified on the GPU
        scores = (Xt @ dense).T                             # c x n
        val, idx = torch.topk(scores, kk, dim=1)
        val, idx = val.cpu().numpy().ravel(), idx.cpu().numpy().ravel()
        r = np.repeat(np.arange(s, s + q.shape[1]), kk)
        keep = val > max(min_score, 0.0)
        rows.append(r[keep])
        cols.append(idx[keep].astype(np.int32))
        vals.append(val[keep])
        del dense, scores
    return (np.concatenate(rows).astype(np.int64), np.concatenate(cols),
            np.concatenate(vals).astype(np.float32))


def topk(Q, X, k, min_score=0.0):
    if _CUDA:
        return gpu_topk(Q, X, k, min_score=min_score)
    return sparse_topk(Q, X, k, min_score=min_score)


def rowwise_dot(A, B):
    """dot(A[i], B[i]) for aligned CSR matrices of the same shape."""
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _knn_dist(xt, xq, k: int, metric: str, leave_one_out: bool) -> np.ndarray:
    n_train = len(xt)
    if n_train == 0:
        return np.full(len(xq), np.nan)
    extra = 1 if leave_one_out else 0
    k_use = min(k + extra, n_train)
    nn = NearestNeighbors(n_neighbors=k_use, metric=metric, algorithm="auto")
    nn.fit(xt)
    dist, _ = nn.kneighbors(xq)
    if leave_one_out and dist.shape[1] > 1:
        # Meyer–Pebesma / CAST: training DI excludes the point itself.
        dist = dist[:, 1:]
    take = min(k, dist.shape[1])
    return dist[:, :take].mean(axis=1)


def dissimilarity_index(X_train, X_query, k: int = 10, leave_one_out: bool = False) -> np.ndarray:
    """Meyer & Pebesma (2021) DI in standardized Euclidean space.

    Scaling is fit on training only. When ``leave_one_out`` is True (used for the
    training-set threshold), the zero self-distance is dropped.
    """
    scaler = StandardScaler()
    xt = scaler.fit_transform(np.asarray(X_train, dtype=float))
    xq = scaler.transform(np.asarray(X_query, dtype=float))
    return _knn_dist(xt, xq, k=k, metric="euclidean", leave_one_out=leave_one_out)


def aoa_threshold(X_train, k: int = 10) -> float:
    di = dissimilarity_index(X_train, X_train, k=k, leave_one_out=True)
    return float(np.mean(di) + 1.645 * np.std(di))


def cosine_dissimilarity(X_train, X_query, k: int = 10, leave_one_out: bool = False) -> np.ndarray:
    """Mean cosine distance to kNN (natural for unit-length AlphaEarth vectors)."""
    xt = np.asarray(X_train, dtype=float)
    xq = np.asarray(X_query, dtype=float)
    xt = xt / np.clip(np.linalg.norm(xt, axis=1, keepdims=True), 1e-9, None)
    xq = xq / np.clip(np.linalg.norm(xq, axis=1, keepdims=True), 1e-9, None)
    return _knn_dist(xt, xq, k=k, metric="cosine", leave_one_out=leave_one_out)


def cosine_threshold(X_train, k: int = 10) -> float:
    di = cosine_dissimilarity(X_train, X_train, k=k, leave_one_out=True)
    return float(np.mean(di) + 1.645 * np.std(di))


def area_of_applicability(X_train, X_query, k: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    di = dissimilarity_index(X_train, X_query, k=k, leave_one_out=False)
    thr = aoa_threshold(X_train, k=k)
    inside = di <= thr
    return di, inside, thr


def vote_entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1)


def error_by_distance(di: np.ndarray, error: np.ndarray, n_bins: int = 6) -> list[dict]:
    di = np.asarray(di, dtype=float)
    err = np.asarray(error, dtype=float)
    m = np.isfinite(di) & np.isfinite(err)
    di, err = di[m], err[m]
    if len(di) < n_bins:
        return []
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(di, qs))
    if len(edges) < 3:
        return []
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (di >= lo) & (di <= hi) if i == len(edges) - 2 else (di >= lo) & (di < hi)
        if sel.sum() == 0:
            continue
        rows.append({
            "bin": i,
            "di_lo": float(lo),
            "di_hi": float(hi),
            "n": int(sel.sum()),
            "error_rate": float(err[sel].mean()),
        })
    return rows

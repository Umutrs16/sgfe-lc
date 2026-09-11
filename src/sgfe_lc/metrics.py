from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_recall_fscore_support,
)
from scipy.stats import chi2, wilcoxon


def _as_arrays(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask].astype(int), yp[mask].astype(int)


def classification_report_dict(y_true, y_pred, labels=None) -> dict:
    yt, yp = _as_arrays(y_true, y_pred)
    if labels is None:
        labels = sorted(set(yt) | set(yp))
    if len(yt) == 0:
        return {"n": 0, "oa": np.nan, "balanced_acc": np.nan, "macro_f1": np.nan, "miou": np.nan}
    oa = accuracy_score(yt, yp)
    bal = balanced_accuracy_score(yt, yp)
    macro_f1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    miou = jaccard_score(yt, yp, labels=labels, average="macro", zero_division=0)
    p, r, f1, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    per_class = {}
    for lab, ua, pa, f, n in zip(labels, p, r, f1, sup):
        per_class[int(lab)] = {
            "ua_precision": float(ua),
            "pa_recall": float(pa),
            "f1": float(f),
            "support": int(n),
        }
    cm = confusion_matrix(yt, yp, labels=labels)
    return {
        "n": int(len(yt)),
        "oa": float(oa),
        "balanced_acc": float(bal),
        "macro_f1": float(macro_f1),
        "miou": float(miou),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "labels": [int(x) for x in labels],
    }


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if proba.ndim != 2 or len(y_true) == 0:
        return float("nan")
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc = (pred[m] == y_true[m]).mean()
        ece += (m.mean()) * abs(acc - conf[m].mean())
    return float(ece)


def brier_multiclass(y_true, proba, labels) -> float:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    if proba.ndim != 2:
        return float("nan")
    idx = {c: i for i, c in enumerate(labels)}
    onehot = np.zeros_like(proba)
    for i, y in enumerate(y_true):
        if y in idx and idx[y] < proba.shape[1]:
            onehot[i, idx[y]] = 1
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    yt, pa = _as_arrays(y_true, pred_a)
    _, pb = _as_arrays(y_true, pred_b)
    n01 = int(((pa == yt) & (pb != yt)).sum())
    n10 = int(((pa != yt) & (pb == yt)).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    p = float(1 - chi2.cdf(stat, 1))
    return {"n01": n01, "n10": n10, "stat": float(stat), "p_value": p}


def paired_wilcoxon(scores_a, scores_b) -> dict:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) < 3:
        return {"stat": np.nan, "p_value": np.nan}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(stat), "p_value": float(p)}


def block_bootstrap_delta(y_true, pred_a, pred_b, blocks, labels, n_boot: int = 1000, seed: int = 20260902):
    """Paired block bootstrap of ΔMacro-F1 (A − B). Blocks are the resampling unit."""
    yt = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)
    blk = np.asarray(blocks)
    rng = np.random.default_rng(seed)
    uniq = np.unique(blk)
    deltas = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(blk == b)[0] for b in drawn]) if len(drawn) else np.array([], dtype=int)
        if len(idx) < 8:
            continue
        fa = f1_score(yt[idx], pa[idx], labels=labels, average="macro", zero_division=0)
        fb = f1_score(yt[idx], pb[idx], labels=labels, average="macro", zero_division=0)
        deltas.append(fa - fb)
    if not deltas:
        return {"delta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "delta_boot_mean": np.nan}
    arr = np.asarray(deltas)
    obs = float(
        f1_score(yt, pa, labels=labels, average="macro", zero_division=0)
        - f1_score(yt, pb, labels=labels, average="macro", zero_division=0)
    )
    return {
        "delta": obs,
        "delta_boot_mean": float(arr.mean()),
        "ci_lo": float(np.quantile(arr, 0.025)),
        "ci_hi": float(np.quantile(arr, 0.975)),
    }


def block_bootstrap_f1(y_true, pred, blocks, labels, n_boot: int = 1000, seed: int = 20260902):
    """Block-bootstrap Macro-F1 for a single prediction vector."""
    yt = np.asarray(y_true)
    yp = np.asarray(pred)
    blk = np.asarray(blocks)
    rng = np.random.default_rng(seed)
    uniq = np.unique(blk)
    scores = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(blk == b)[0] for b in drawn]) if len(drawn) else np.array([], dtype=int)
        if len(idx) < 8:
            continue
        scores.append(f1_score(yt[idx], yp[idx], labels=labels, average="macro", zero_division=0))
    point = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    if not scores:
        return {"macro_f1": point, "ci_lo": np.nan, "ci_hi": np.nan}
    arr = np.asarray(scores)
    return {"macro_f1": point, "ci_lo": float(np.quantile(arr, 0.025)), "ci_hi": float(np.quantile(arr, 0.975))}


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 20260902, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def summarize_folds(fold_reports: list[dict]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(fold_reports):
        rows.append(
            {
                "fold": i,
                "n": r.get("n"),
                "oa": r.get("oa"),
                "balanced_acc": r.get("balanced_acc"),
                "macro_f1": r.get("macro_f1"),
                "miou": r.get("miou"),
            }
        )
    return pd.DataFrame(rows)


def reports_to_table(named_reports: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, r in named_reports.items():
        row = {"model": name, "n": r.get("n"), "oa": r.get("oa"),
               "balanced_acc": r.get("balanced_acc"), "macro_f1": r.get("macro_f1"),
               "miou": r.get("miou")}
        for cid, stats in (r.get("per_class") or {}).items():
            row[f"f1_{cid}"] = stats["f1"]
            row[f"pa_{cid}"] = stats["pa_recall"]
            row[f"ua_{cid}"] = stats["ua_precision"]
        rows.append(row)
    return pd.DataFrame(rows)

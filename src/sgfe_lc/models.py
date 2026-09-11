from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURE_GROUPS = {
    "M1_L8": ("l8_",),
    "M2_S2": ("s2_",),
    "M3_S1S2": ("s1_", "s2_"),
    "M4_AEF": ("A",),
    "M4_AEF_TERRAIN": ("A", "elev", "slope"),
    "M5_AEF_ENV": ("A", "elev", "slope", "aspect_", "precip", "t2m"),
    "M6_AEF_PHENO": ("A", "l8_ndvi", "s2_ndvi", "l8_evi", "s2_evi"),
}


def select_feature_columns(columns, group: str) -> list[str]:
    prefixes = FEATURE_GROUPS[group]
    cols = []
    for c in columns:
        if c in {"class_id", "site_id", "country", "year", "split", "fold_id", "block_id",
                 "tier", "longitude", "latitude", "geometry"}:
            continue
        if group == "M4_AEF" and c.startswith("A") and c[1:].isdigit():
            cols.append(c)
            continue
        if group == "M4_AEF":
            continue
        if any(c.startswith(p) or c == p or p in c for p in prefixes):
            if group.startswith("M5") or group.startswith("M6") or not c.startswith("A") or (
                c.startswith("A") and c[1:].isdigit() and any(p == "A" for p in prefixes)
            ):
                cols.append(c)
    if group == "M4_AEF":
        cols = [c for c in columns if c.startswith("A") and c[1:].isdigit()]
    if group == "M4_AEF_TERRAIN":
        aef = [c for c in columns if c.startswith("A") and c[1:].isdigit()]
        terrain = [c for c in columns if c in {"elev", "slope"}]
        cols = aef + terrain
    if group == "M5_AEF_ENV":
        aef = [c for c in columns if c.startswith("A") and c[1:].isdigit()]
        env = [c for c in columns if c in {"elev", "slope", "aspect_sin", "aspect_cos", "precip_annual", "t2m_mean"}
               or c.startswith("elev") or c.startswith("slope") or c.startswith("precip") or c.startswith("t2m")]
        cols = aef + env
    if group == "M6_AEF_PHENO":
        aef = [c for c in columns if c.startswith("A") and c[1:].isdigit()]
        pheno = [c for c in columns if any(k in c.lower() for k in
                 ["ndvi_p90", "ndvi_max", "ndvi_amp", "ndvi_std", "evi_p90", "evi_max"])]
        cols = aef + pheno
    return sorted(set(cols))


@dataclass
class FittedModel:
    name: str
    pipeline: Pipeline | None
    classes_: np.ndarray
    feature_names: list[str]


def make_estimator(name: str, seed: int = 20260902, n_trees: int = 300, max_depth: int = 6):
    if name == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    if name == "xgb":
        clf = XGBClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
    raise ValueError(name)


def fit_predict(name, X_train, y_train, X_test, seed=20260902, n_trees=300, sample_weight=None) -> tuple[np.ndarray, np.ndarray | None, FittedModel]:
    y_arr = np.asarray(y_train, dtype=float).ravel()
    labels = np.unique(y_arr[np.isfinite(y_arr)])
    if len(X_train) == 0 or len(labels) < 2:
        pred = np.full(len(X_test), labels[0] if len(labels) else np.nan)
        model = FittedModel(
            name=name, pipeline=None, classes_=labels,
            feature_names=list(getattr(X_train, "columns", [])),
        )
        return pred, None, model
    remap = {int(c): i for i, c in enumerate(labels)}
    inv = {i: int(c) for c, i in remap.items()}
    y_enc = np.array([remap[int(v)] for v in y_arr], dtype=int)
    est = make_estimator(name, seed=seed, n_trees=n_trees)
    fit_kw = {}
    if sample_weight is not None:
        fit_kw["clf__sample_weight"] = np.asarray(sample_weight, dtype=float)
    est.fit(X_train, y_enc, **fit_kw)
    pred_enc = est.predict(X_test)
    pred = np.vectorize(inv.get)(pred_enc)
    proba = None
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X_test)
    model = FittedModel(name=name, pipeline=est, classes_=labels, feature_names=list(getattr(X_train, "columns", [])))
    return pred, proba, model

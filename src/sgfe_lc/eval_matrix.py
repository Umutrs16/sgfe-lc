"""Redesigned experiment matrix after JAG-style review (exact-year, few-shot, reliability weights)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .aoa import (
    area_of_applicability,
    cosine_dissimilarity,
    cosine_threshold,
    error_by_distance,
)
from .config import load_config, resolve_path
from .legend import BENCHMARK_CLASSES, CORE_CLASSES, LEVEL1, RANGELAND_ID

# Largest k such that every benchmark class has ≥k unique train/dev labels.
# Bare train/dev n=28 and Shrubland train/dev n=34, so 40-shot was silently capped.
HEADLINE_SHOTS = (5, 10, 20, 25)
TRACK_B_SUPPORTED = [1, 3, 5]  # Cropland, Grassland, Bare; Shrubland n=2 is descriptive only
from .metrics import (
    block_bootstrap_delta,
    block_bootstrap_f1,
    bootstrap_ci,
    classification_report_dict,
    expected_calibration_error,
    mcnemar_test,
)
from .models import fit_predict, select_feature_columns


def _xy(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["class_id"], errors="coerce")
    mask = y.notna() & X.notna().any(axis=1)
    return X.loc[mask], y.loc[mask].astype(int), mask


def exact_year(df):
    if "role" in df.columns:
        return df[df["role"] == "exact_year"].copy()
    year = pd.to_numeric(df.get("year"), errors="coerce")
    return df[year.isin([2017, 2018])].copy()


def historical(df):
    if "role" in df.columns:
        return df[df["role"] == "historical"].copy()
    year = pd.to_numeric(df.get("year"), errors="coerce")
    return df[year.isin([2013, 2014])].copy()


def _to_rangeland(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.replace({4: RANGELAND_ID})


def e1_exact_year_products(df: pd.DataFrame) -> dict:
    ey = exact_year(df)
    labels_fine = list(CORE_CLASSES)
    # Common headline: crop / rangeland / bare. Built (n=2) is descriptive only.
    labels_common = [1, 3, 5]
    headline = pd.to_numeric(ey["class_id"], errors="coerce").isin(BENCHMARK_CLASSES)
    out = {
        "n_exact_year": int(len(ey)),
        "n_headline": int(headline.sum()),
        "headline_classes_fine": labels_fine,
        "headline_classes_common": labels_common,
        "excluded_from_headline": [7, 2, 6],
        "fine_ontology": {},
        "common_ontology": {},
    }
    for col, name in (("dw_l1", "Dynamic World"), ("esri_l1", "Esri"), ("glc_l1", "GLC_FCS30D")):
        if col not in ey.columns:
            continue
        yt = pd.to_numeric(ey["class_id"], errors="coerce")
        yp = pd.to_numeric(ey[col], errors="coerce")
        m = yt.notna() & yp.notna() & (yp > 0) & headline
        if name == "Esri":
            # Fine ontology: Esri cannot represent shrubland; exclude class 4 from labels.
            fine_labels = [c for c in labels_fine if c != 4]
            out["fine_ontology"][name] = classification_report_dict(yt[m], yp[m], labels=fine_labels)
            out["fine_ontology"][name]["note"] = "Esri rangeland has no shrub class; shrubland excluded from fine-ontology labels."
        else:
            out["fine_ontology"][name] = classification_report_dict(yt[m], yp[m], labels=labels_fine)
        ytr = _to_rangeland(yt[m])
        ypr = _to_rangeland(yp[m])
        out["common_ontology"][name] = classification_report_dict(ytr, ypr, labels=labels_common)
        if "block_id" in ey.columns:
            bb = block_bootstrap_f1(ytr, ypr, ey.loc[m, "block_id"].astype(str), labels_common)
            out["common_ontology"][name]["macro_f1_ci"] = [bb["ci_lo"], bb["ci_hi"]]
            out["common_ontology"][name]["n_blocks"] = int(ey.loc[m, "block_id"].nunique())
        # Fine-ontology overall Macro-F1 is diagnostic only; Esri lacks shrub.
        out["fine_ontology"][name]["ranking"] = "not_comparable" if name == "Esri" else "diagnostic"
    # GLC-only historical exact-year (survey year = product year)
    hist = historical(df)
    if "glc_l1" in hist.columns and len(hist):
        yt = pd.to_numeric(hist["class_id"], errors="coerce")
        yp = pd.to_numeric(hist["glc_l1"], errors="coerce")
        m = yt.notna() & yp.notna() & (yp > 0)
        out["glc_historical_exact_year"] = classification_report_dict(yt[m], yp[m], labels=labels_fine)
        out["glc_historical_exact_year"]["n_sites"] = int(len(hist))
    out["class_support"] = _class_support(df)
    common = out.get("common_ontology") or {}
    if {"Esri", "Dynamic World"}.issubset(common) and "block_id" in ey.columns:
        m = headline & pd.to_numeric(ey["class_id"], errors="coerce").notna()
        m = m & pd.to_numeric(ey.get("esri_l1"), errors="coerce").notna() & pd.to_numeric(ey.get("dw_l1"), errors="coerce").notna()
        yt = _to_rangeland(pd.to_numeric(ey.loc[m, "class_id"], errors="coerce"))
        ye = _to_rangeland(pd.to_numeric(ey.loc[m, "esri_l1"], errors="coerce"))
        yd = _to_rangeland(pd.to_numeric(ey.loc[m, "dw_l1"], errors="coerce"))
        out["esri_minus_dw"] = block_bootstrap_delta(yt, ye, yd, ey.loc[m, "block_id"].astype(str), labels_common)
    return out


def _class_support(df: pd.DataFrame) -> list[dict]:
    ey = exact_year(df)
    hist = historical(df)
    rows = []
    for cid, name in LEVEL1.items():
        n_ey = int((pd.to_numeric(ey.get("class_id"), errors="coerce") == cid).sum())
        n_hi = int((pd.to_numeric(hist.get("class_id"), errors="coerce") == cid).sum())
        rows.append({
            "class_id": int(cid),
            "class": "Snow/ice" if cid == 8 else name,
            "exact_year_n": n_ey, "historical_n": n_hi,
            "headline": cid in BENCHMARK_CLASSES,
            "map_class": cid != 8,
        })
    return rows


def _protocol_complexity(label: str) -> int:
    """None < ungated < soft < hard (any threshold)."""
    if label == "none":
        return 0
    if label == "ungated":
        return 1
    if label == "soft":
        return 2
    if str(label).startswith("hard"):
        return 3
    return 9


def _one_se_recommend(stats: dict) -> tuple[str | None, float, list[str]]:
    """Repeated-CV 1-SE: simplest protocol whose seed-level mean is within 1 SE of the best mean."""
    if not stats:
        return None, float("nan"), []
    best = max(stats, key=lambda k: stats[k]["mean"])
    floor = stats[best]["mean"] - stats[best]["se"]
    in_set = [k for k, rec in stats.items() if rec["mean"] == rec["mean"] and rec["mean"] >= floor]
    recommend = min(in_set, key=lambda k: (_protocol_complexity(k), -stats[k]["mean"])) if in_set else best
    return recommend, float(floor), in_set


def _spatial_cv(df, cols, estimator="xgb", seed=20260902, labels=None, sample_weight=None):
    labels = labels or list(CORE_CLASSES)
    X, y, mask = _xy(df, cols)
    w = None
    if sample_weight is not None:
        w = pd.Series(sample_weight, index=df.index).loc[X.index]
    groups = df.loc[mask, "block_id"].astype(str)
    folds = min(5, max(2, groups.nunique()))
    gkf = GroupKFold(n_splits=folds)
    reports = []
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        sw = None if w is None else w.iloc[tr]
        pred, _, _ = fit_predict(estimator, X.iloc[tr], y.iloc[tr], X.iloc[te], seed=seed + i, sample_weight=sw)
        reports.append(classification_report_dict(y.iloc[te], pred, labels=labels))
    f1s = [r["macro_f1"] for r in reports]
    mean, lo, hi = bootstrap_ci(f1s, seed=seed)
    return {"folds": reports, "macro_f1_mean": mean, "macro_f1_ci": [lo, hi], "n_features": len(cols)}


def e2_features(df: pd.DataFrame, estimator="xgb", seed=20260902) -> dict:
    ey = exact_year(df)
    train = ey[ey["split"] != "test"]
    out = {}
    for group in ("M1_L8", "M2_S2", "M3_S1S2", "M4_AEF", "M5_AEF_ENV"):
        cols = select_feature_columns(train.columns, group)
        if len(cols) < 3:
            out[group] = {"error": f"insufficient columns: {len(cols)}"}
            continue
        out[group] = _spatial_cv(train, cols, estimator=estimator, seed=seed)
        out[group]["n_train"] = int(len(train))
    return out


def e3_fewshot(df: pd.DataFrame, shots=HEADLINE_SHOTS, repeats=30, estimator="xgb", seed=20260902) -> dict:
    """Paired few-shot: the same stratified draw trains L8, S1/S2 and AEF."""
    ey = exact_year(df)
    test = ey[ey["split"] == "test"]
    train = ey[ey["split"] != "test"]
    groups = [g for g in ("M1_L8", "M3_S1S2", "M4_AEF") if len(select_feature_columns(df.columns, g)) >= 3]
    if test.empty or train.empty or not groups:
        return {"rows": [], "paired_delta": [], "le_auc": []}
    ytr_all = pd.to_numeric(train["class_id"], errors="coerce")
    yt_all = pd.to_numeric(test["class_id"], errors="coerce")
    shot_labels = [lab for lab in BENCHMARK_CLASSES if int((ytr_all == lab).sum()) >= 1 and int((yt_all == lab).sum()) >= 1]
    if len(shot_labels) < 2:
        return {"rows": [], "paired_delta": [], "le_auc": []}
    train_support = {int(lab): int((ytr_all == lab).sum()) for lab in shot_labels}
    test_ids = set(test.index)
    rng = np.random.default_rng(seed)
    raw = []
    realized_max = {int(lab): 0 for lab in shot_labels}
    for k in shots:
        for r in range(repeats):
            idx = []
            for lab in shot_labels:
                lab_idx = ytr_all.index[ytr_all == lab]
                take = min(int(k), int(len(lab_idx)))
                realized_max[int(lab)] = max(realized_max[int(lab)], take)
                if take < 1:
                    continue
                chosen = rng.choice(lab_idx.to_numpy(), size=take, replace=False).tolist()
                if test_ids.intersection(chosen):
                    raise RuntimeError("few-shot sampled a locked-test site")
                idx.extend(chosen)
            if len(idx) < 8:
                continue
            scores = {}
            for group in groups:
                cols = select_feature_columns(df.columns, group)
                Xt, yt, _ = _xy(test, cols)
                Xtr, ytr, _ = _xy(train.loc[idx], cols)
                if len(Xtr) < 8 or len(Xt) < 8:
                    continue
                pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed + r)
                scores[group] = classification_report_dict(yt, pred, labels=shot_labels)["macro_f1"]
            if len(scores) < 2:
                continue
            raw.append({"shots": k, "rep": r, **scores})
    long = pd.DataFrame(raw)
    rows = []
    if long.empty:
        return {"rows": [], "paired_delta": [], "le_auc": []}
    for group in groups:
        if group not in long.columns:
            continue
        for k, sub in long.groupby("shots"):
            mean, lo, hi = bootstrap_ci(sub[group].dropna(), seed=seed)
            rows.append({
                "group": group, "shots": int(k), "n_repeats": int(sub[group].notna().sum()),
                "macro_f1": mean, "ci_lo": lo, "ci_hi": hi,
                "shot_classes": shot_labels,
                "excluded_core": [c for c in CORE_CLASSES if c not in shot_labels],
            })
    paired = []
    if {"M4_AEF", "M3_S1S2"}.issubset(long.columns):
        for k, sub in long.groupby("shots"):
            d = (sub["M4_AEF"] - sub["M3_S1S2"]).dropna()
            mean, lo, hi = bootstrap_ci(d, seed=seed)
            paired.append({"shots": int(k), "delta": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(d))})
    le_auc = []
    shot_x = np.array(shots, float)
    for group in groups:
        if group not in long.columns:
            continue
        aucs = []
        for _, sub in long.groupby("rep"):
            s = sub.set_index("shots")[group].reindex(shots)
            if s.notna().sum() < 2:
                continue
            y = s.to_numpy(float)
            aucs.append(float(np.trapz(y, shot_x) / (shot_x[-1] - shot_x[0])))
        mean, lo, hi = bootstrap_ci(aucs, seed=seed) if aucs else (np.nan, np.nan, np.nan)
        le_auc.append({"group": group, "le_auc": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(aucs))})
        for row in rows:
            if row["group"] == group:
                row["le_auc"] = mean
                row["le_auc_ci_lo"] = lo
                row["le_auc_ci_hi"] = hi
    return {
        "rows": rows, "paired_delta": paired, "le_auc": le_auc,
        "shots": list(shots),
        "train_support": train_support,
        "realized_max": realized_max,
        "sampling": "unique train/dev labels, without replacement; test sites excluded",
        "capped_classes": {str(k): v for k, v in realized_max.items() if v < max(shots)},
    }


def e4_percent(df: pd.DataFrame, fractions=(0.05, 0.10, 0.20, 0.40, 0.80, 1.0), repeats=6, estimator="xgb", seed=20260902):
    ey = exact_year(df)
    test = ey[ey["split"] == "test"]
    train = ey[ey["split"] != "test"]
    rows = []
    rng = np.random.default_rng(seed)
    for group in ("M1_L8", "M3_S1S2", "M4_AEF"):
        cols = select_feature_columns(df.columns, group)
        if len(cols) < 3:
            continue
        Xt, yt, _ = _xy(test, cols)
        Xtr, ytr, _ = _xy(train, cols)
        for frac in fractions:
            scores = []
            for r in range(repeats):
                idx = []
                for lab in ytr.unique():
                    lab_idx = ytr.index[ytr == lab]
                    take = max(1, int(round(len(lab_idx) * frac)))
                    take = min(take, len(lab_idx))
                    idx.extend(rng.choice(lab_idx.to_numpy(), size=take, replace=False).tolist())
                pred, _, _ = fit_predict(estimator, Xtr.loc[idx], ytr.loc[idx], Xt, seed=seed + r)
                scores.append(classification_report_dict(yt, pred, labels=list(CORE_CLASSES))["macro_f1"])
            mean, lo, hi = bootstrap_ci(scores, seed=seed)
            rows.append({"group": group, "frac": frac, "macro_f1": mean, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def _hist_subset(hist, mode, high=0.80):
    if hist.empty:
        return hist
    if mode == "none":
        return hist.iloc[0:0]
    if mode == "ungated":
        out = hist.copy()
        out["w_rel"] = 1.0
        return out
    if mode == "hard":
        w = pd.to_numeric(hist.get("w_rel"), errors="coerce")
        out = hist.loc[w >= high].copy()
        out["w_rel"] = 1.0
        return out
    # soft
    return hist.copy()


def e6_temporal(df: pd.DataFrame, estimator="xgb", seed=20260902, high=0.80) -> dict:
    ey = exact_year(df)
    test = ey[ey["split"] == "test"]
    train_ey = ey[ey["split"] != "test"]
    hist = historical(df)
    hist = hist[hist["split"] != "test"]  # no leakage from test blocks
    cols = select_feature_columns(df.columns, "M4_AEF") or select_feature_columns(df.columns, "M3_S1S2")
    if len(cols) < 3 or test.empty:
        return {"error": "features or test missing"}
    Xt, yt, _ = _xy(test, cols)
    reports, preds, weights_used = {}, {}, {}
    for mode in ("none", "ungated", "hard", "soft"):
        extra = _hist_subset(hist, mode, high=high)
        tr = pd.concat([train_ey, extra], ignore_index=False)
        Xtr, ytr, _ = _xy(tr, cols)
        w = pd.to_numeric(tr.loc[Xtr.index].get("w_rel"), errors="coerce").fillna(1.0)
        pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed, sample_weight=w)
        reports[mode] = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
        reports[mode]["n_train_exact"] = int(len(train_ey))
        reports[mode]["n_train_hist"] = int(len(extra))
        bb = block_bootstrap_f1(yt, pred, blocks=test.loc[Xt.index, "block_id"].astype(str), labels=list(CORE_CLASSES), seed=seed)
        reports[mode]["macro_f1_ci"] = [bb["ci_lo"], bb["ci_hi"]]
        preds[mode] = pred
        weights_used[mode] = float(w.mean()) if len(w) else np.nan
    tests = {}
    blocks = test.loc[Xt.index, "block_id"].astype(str)
    if "soft" in preds and "none" in preds:
        tests["soft_vs_none"] = block_bootstrap_delta(yt, preds["soft"], preds["none"], blocks, CORE_CLASSES, seed=seed)
        tests["soft_vs_none_mcnemar_oa"] = mcnemar_test(yt, preds["soft"], preds["none"])
    if "soft" in preds and "ungated" in preds:
        tests["soft_vs_ungated"] = block_bootstrap_delta(yt, preds["soft"], preds["ungated"], blocks, CORE_CLASSES, seed=seed)
        tests["soft_vs_ungated_mcnemar_oa"] = mcnemar_test(yt, preds["soft"], preds["ungated"])
    if "ungated" in preds and "none" in preds:
        tests["ungated_vs_none"] = block_bootstrap_delta(yt, preds["ungated"], preds["none"], blocks, CORE_CLASSES, seed=seed)
    if "hard" in preds and "none" in preds:
        tests["hard_vs_none"] = block_bootstrap_delta(yt, preds["hard"], preds["none"], blocks, CORE_CLASSES, seed=seed)
        tests["hard_vs_none_mcnemar_oa"] = mcnemar_test(yt, preds["hard"], preds["none"])
    if "hard" in preds and "ungated" in preds:
        tests["hard_vs_ungated"] = block_bootstrap_delta(yt, preds["hard"], preds["ungated"], blocks, CORE_CLASSES, seed=seed)
    if "hard" in preds and "soft" in preds:
        tests["hard_vs_soft"] = block_bootstrap_delta(yt, preds["hard"], preds["soft"], blocks, CORE_CLASSES, seed=seed)
    return {"reports": reports, "deltas": tests, "mean_weight": weights_used}


def e6_threshold_sensitivity(df: pd.DataFrame, thresholds=(0.60, 0.70, 0.80, 0.90), estimator="xgb", seed=20260902) -> pd.DataFrame:
    ey = exact_year(df)
    test = ey[ey["split"] == "test"]
    train_ey = ey[ey["split"] != "test"]
    hist = historical(df)
    hist = hist[hist["split"] != "test"]
    cols = select_feature_columns(df.columns, "M4_AEF")
    if len(cols) < 3 or test.empty:
        return pd.DataFrame()
    Xt, yt, _ = _xy(test, cols)
    rows = []
    for thr in thresholds:
        extra = _hist_subset(hist, "hard", high=thr)
        tr = pd.concat([train_ey, extra], ignore_index=False)
        Xtr, ytr, _ = _xy(tr, cols)
        pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
        rep = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
        rows.append({"threshold": thr, "n_hist": int(len(extra)), "macro_f1": rep["macro_f1"], "oa": rep["oa"]})
    return pd.DataFrame(rows)


def e6_nested_select(df: pd.DataFrame, estimator="xgb", seed=20260902, target_year=None,
                     feature_group=None) -> dict:
    """Choose historical protocol + hard threshold on train/dev spatial CV only.

    If target_year is set (Track B: 2018), validation folds keep only that survey year.
    2017 exact-year sites and historical extras remain training augmentation.
    Track B RF uses the same predictor set as the GEE map (AEF + elev + slope).
    """
    ey = exact_year(df)
    train_ey = ey[ey["split"] != "test"].copy()
    hist = historical(df)
    hist = hist[hist["split"] != "test"]
    if feature_group is None:
        feature_group = "M4_AEF_TERRAIN" if estimator == "rf" and target_year == 2018 else "M4_AEF"
    cols = select_feature_columns(df.columns, feature_group)
    if len(cols) < 3 or train_ey.empty:
        return {"error": "insufficient train/dev"}
    candidates = [("none", None), ("ungated", None)]
    if estimator == "xgb":
        candidates.append(("soft", None))
    candidates.extend([("hard", 0.60), ("hard", 0.70), ("hard", 0.80), ("hard", 0.90)])
    groups = train_ey["block_id"].astype(str)
    folds = min(5, max(2, groups.nunique()))
    gkf = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(train_ey))
    rows = []
    for mode, thr in candidates:
        scores = []
        for i, (tr_idx, te_idx) in enumerate(gkf.split(dummy, groups=groups)):
            tr = train_ey.iloc[tr_idx].copy()
            te = train_ey.iloc[te_idx].copy()
            val_blocks = set(te["block_id"].astype(str))
            # Validation and training blocks are spatially disjoint for every year and label source.
            tr = tr[~tr["block_id"].astype(str).isin(val_blocks)]
            if target_year is not None:
                te = te[pd.to_numeric(te.get("year"), errors="coerce").eq(int(target_year))]
            extra = _hist_subset(hist, mode, high=thr if thr is not None else 0.80)
            if not extra.empty and "block_id" in extra.columns:
                extra = extra[~extra["block_id"].astype(str).isin(val_blocks)]
            Xtr, ytr, _ = _xy(pd.concat([tr, extra]), cols)
            Xt, yt, _ = _xy(te, cols)
            if len(Xtr) < 8 or len(Xt) < 5 or ytr.nunique() < 2:
                continue
            w = pd.to_numeric(pd.concat([tr, extra]).loc[Xtr.index].get("w_rel"), errors="coerce").fillna(1.0)
            pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed + i, sample_weight=w)
            scores.append(classification_report_dict(yt, pred, labels=list(CORE_CLASSES))["macro_f1"])
        mean, lo, hi = bootstrap_ci(scores, seed=seed) if scores else (np.nan, np.nan, np.nan)
        rows.append({"mode": mode, "threshold": thr, "cv_macro_f1": mean, "ci_lo": lo, "ci_hi": hi, "n_folds": len(scores)})
    tab = pd.DataFrame(rows)
    if tab.empty or not np.isfinite(tab["cv_macro_f1"]).any():
        return {"selected": {"mode": "hard", "threshold": 0.80}, "table": []}
    best = tab.loc[tab["cv_macro_f1"].idxmax()]
    return {
        "selected": {"mode": best["mode"], "threshold": None if pd.isna(best["threshold"]) else float(best["threshold"]),
                     "cv_macro_f1": float(best["cv_macro_f1"])},
        "table": tab.to_dict(orient="records"),
        "target_year": target_year,
        "feature_group": feature_group,
        "n_predictors": int(len(cols)),
        "note": (
            "Selection used train/dev spatial CV only; locked test was not used. "
            + (
                "When a block was assigned to a 2018 validation fold, all 2017 exact-year "
                "and historical samples in that block were removed from the corresponding "
                "training fold."
                if target_year == 2018
                else "Validation folds used pooled 2017–2018 train/dev sites."
            )
        ),
    }


def e7_repeated_holdout(df: pd.DataFrame, n_rep=20, estimator="xgb", seed=20260902) -> dict:
    """Same spatial splits for M3 and M4; paired ΔF1 = AEF − S1/S2."""
    ey = exact_year(df)
    rows = []
    rng = np.random.default_rng(seed)
    blocks = ey["block_id"].dropna().unique()
    cols = {g: select_feature_columns(ey.columns, g) for g in ("M3_S1S2", "M4_AEF")}
    for r in range(n_rep):
        drawn = rng.choice(blocks, size=max(1, int(round(0.2 * len(blocks)))), replace=False)
        te = ey[ey["block_id"].isin(drawn)]
        tr = ey[~ey["block_id"].isin(drawn)]
        if len(tr) < 20 or len(te) < 8:
            continue
        f1 = {}
        for group in ("M3_S1S2", "M4_AEF"):
            if len(cols[group]) < 3:
                continue
            Xtr, ytr, _ = _xy(tr, cols[group])
            Xt, yt, _ = _xy(te, cols[group])
            if len(Xtr) < 8 or len(Xt) < 8:
                continue
            pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed + r)
            rep = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
            f1[group] = rep["macro_f1"]
            rows.append({"group": group, "rep": r, "macro_f1": rep["macro_f1"], "oa": rep["oa"]})
        if "M3_S1S2" in f1 and "M4_AEF" in f1:
            rows.append({"group": "delta_AEF_minus_S1S2", "rep": r, "macro_f1": f1["M4_AEF"] - f1["M3_S1S2"], "oa": np.nan})
    tab = pd.DataFrame(rows)
    summary = {}
    if tab.empty:
        return {"repeats": [], "summary": {}}
    for group, sub in tab.groupby("group"):
        mean, lo, hi = bootstrap_ci(sub["macro_f1"], seed=seed)
        summary[group] = {"holdout_macro_f1": mean, "holdout_ci": [lo, hi]}
    return {"repeats": tab.to_dict(orient="records"), "summary": summary}


def e8_loco_matched(df: pd.DataFrame, repeats=20, estimator="xgb", seed=20260902) -> pd.DataFrame:
    ey = exact_year(df)
    countries = [c for c in ("KAZ", "UZB") if (ey["country"] == c).any()]
    train_sizes = []
    for hold in countries:
        n = ((ey["country"] != hold) & (ey["split"] != "test")).sum()
        train_sizes.append(int(n))
    usable = [n for n in train_sizes if n >= 8]
    n_match = min(usable) if usable else max(8, min(train_sizes) if train_sizes else 8)
    rng = np.random.default_rng(seed)
    rows = []
    for group in ("M3_S1S2", "M4_AEF"):
        cols = select_feature_columns(ey.columns, group)
        if len(cols) < 3:
            continue
        for hold in countries:
            pool = ey[(ey["country"] != hold) & (ey["split"] != "test")]
            te = ey[ey["country"] == hold]
            if len(pool) < 8 or len(te) < 8:
                continue
            Xte, yte, _ = _xy(te, cols)
            Xpool, ypool, _ = _xy(pool, cols)
            if len(Xte) < 8 or len(Xpool) < 8 or ypool.nunique() < 2:
                continue
            scores = []
            last_pred = None
            if len(te) >= 20:
                within_f1 = _spatial_cv(te, cols, estimator=estimator, seed=seed)["macro_f1_mean"]
            else:
                within_f1 = np.nan
            blocks_te = te.loc[Xte.index, "block_id"].astype(str)
            for r in range(repeats):
                take = min(n_match, len(pool), len(Xpool))
                samp = pool.sample(n=take, random_state=int(rng.integers(0, 1_000_000)))
                Xtr, ytr, _ = _xy(samp, cols)
                if len(Xtr) < 8 or ytr.nunique() < 2:
                    continue
                try:
                    pred, _, _ = fit_predict(estimator, Xtr, ytr, Xte, seed=seed + r)
                except Exception:
                    continue
                last_pred = pred
                scores.append(classification_report_dict(yte, pred, labels=list(CORE_CLASSES))["macro_f1"])
            if not scores:
                continue
            mean, lo, hi = bootstrap_ci(scores, seed=seed)
            train_ci = [lo, hi]
            if last_pred is not None:
                bb = block_bootstrap_f1(yte, last_pred, blocks_te, CORE_CLASSES, seed=seed)
                target_ci = [bb["ci_lo"], bb["ci_hi"]]
                if float(np.std(scores)) < 1e-4:
                    mean = bb["macro_f1"]
            else:
                target_ci = [np.nan, np.nan]
            lo = float(np.nanmin([train_ci[0], target_ci[0]]))
            hi = float(np.nanmax([train_ci[1], target_ci[1]]))
            rows.append({
                "group": group, "test_country": hold, "n_test": int(len(te)),
                "n_match": n_match, "macro_f1": mean, "ci_lo": lo, "ci_hi": hi,
                "ci_train_subsample": train_ci, "ci_target_block": target_ci,
                "within_macro_f1": within_f1,
                "score_sd": float(np.std(scores)),
            })
    return pd.DataFrame(rows)


def e8_historical_countries(df: pd.DataFrame, estimator="xgb", seed=20260902) -> pd.DataFrame:
    """KGZ/TJK have no 2017–2018 field points; test 2013/2014 labels on 2018 features."""
    ey = exact_year(df)
    hist = historical(df)
    rows = []
    for group in ("M3_S1S2", "M4_AEF"):
        cols = select_feature_columns(df.columns, group)
        if len(cols) < 3:
            continue
        tr = ey[ey["split"] != "test"]
        Xtr, ytr, _ = _xy(tr, cols)
        for hold in ("KGZ", "TJK"):
            te = hist[(hist["country"] == hold) & (hist["split"] != "test")]
            if len(te) < 8:
                te = hist[hist["country"] == hold]
            if len(te) < 8:
                continue
            Xt, yt, _ = _xy(te, cols)
            pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
            rep = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
            rows.append({"group": group, "test_country": hold, "label_years": "2013-2014",
                         **{k: rep[k] for k in ("n", "oa", "macro_f1")}})
    return pd.DataFrame(rows)


def e9_env(df: pd.DataFrame, estimator="xgb", seed=20260902) -> pd.DataFrame:
    ey = exact_year(df)
    colname = "env_stratum" if "env_stratum" in ey.columns else ("ecoregion" if "ecoregion" in ey.columns else None)
    if not colname:
        return pd.DataFrame()
    cols = select_feature_columns(ey.columns, "M4_AEF") or select_feature_columns(ey.columns, "M3_S1S2")
    rows = []
    for zone in ey[colname].dropna().unique():
        if "nan" in str(zone).lower():
            continue
        tr = ey[(ey[colname] != zone) & (ey["split"] != "test")]
        te = ey[ey[colname] == zone]
        if len(tr) < 20 or len(te) < 8:
            continue
        Xtr, ytr, _ = _xy(tr, cols)
        Xt, yt, _ = _xy(te, cols)
        pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
        rep = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
        rows.append({"holdout_stratum": zone, **{k: rep[k] for k in ("n", "oa", "macro_f1", "miou")}})
    return pd.DataFrame(rows)


def e10_pseudo(df: pd.DataFrame, estimator="xgb", seed=20260902) -> dict:
    if "tier" not in df.columns or not (df.get("tier") == "C").any():
        return {"status": "not_run", "note": "No Tier C consensus pixels were harvested; module reserved for map-stage expansion."}
    ey = exact_year(df)
    test = ey[ey["split"] == "test"]
    train = ey[ey["split"] != "test"]
    pseudo = df[(df["tier"] == "C") & (df.get("split", "traindev") != "test")]
    cols = select_feature_columns(df.columns, "M4_AEF")
    Xt, yt, _ = _xy(test, cols)
    out = {}
    for name, tr in (("field_only", train), ("field_plus_consensus", pd.concat([train, pseudo]))):
        Xtr, ytr, _ = _xy(tr, cols)
        pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
        out[name] = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
    out["status"] = "run"
    return out


def e11_uncertainty(df: pd.DataFrame, estimator="xgb", seed=20260902) -> dict:
    ey = exact_year(df)
    cols = select_feature_columns(ey.columns, "M4_AEF")
    if len(cols) < 3:
        return {"error": "AEF columns missing"}
    train = ey[ey["split"] != "test"]
    test = ey[ey["split"] == "test"]
    Xtr, ytr, _ = _xy(train, cols)
    Xt, yt, _ = _xy(test, cols)
    pred, proba, model = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
    di, inside, thr = area_of_applicability(Xtr, Xt)
    cos = cosine_dissimilarity(Xtr, Xt, leave_one_out=False)
    cos_thr = cosine_threshold(Xtr)
    err = (np.asarray(yt) != np.asarray(pred)).astype(float)
    out = {
        "aoa_threshold": thr,
        "cosine_threshold": cos_thr,
        "n_test": int(len(yt)),
        "threshold_method": "training_LOO_mean_plus_1.645sd",
        "scaling": "StandardScaler fit on training only; query points transformed",
        "self_inclusion": False,
    }
    tes = test.loc[Xt.index]
    for name, mask in (
        ("euclid_inside", inside),
        ("euclid_outside", ~inside),
        ("cosine_inside", cos <= cos_thr),
        ("cosine_outside", cos > cos_thr),
    ):
        if mask.sum() < 5:
            out[name] = {"n": int(mask.sum())}
            continue
        rep = classification_report_dict(yt[mask], pred[mask], labels=list(CORE_CLASSES))
        sub = tes.iloc[np.flatnonzero(np.asarray(mask))]
        ysub = pd.to_numeric(sub["class_id"], errors="coerce")
        rep["class_share"] = {int(k): float(v) for k, v in ysub.value_counts(normalize=True).to_dict().items() if pd.notna(k)}
        rep["country_share"] = {str(k): float(v) for k, v in sub["country"].value_counts(normalize=True).to_dict().items()}
        out[name] = rep
    out["euclid_error_vs_di"] = error_by_distance(di, err)
    out["cosine_error_vs_di"] = error_by_distance(cos, err)
    if len(err) >= 8:
        from scipy.stats import spearmanr
        out["spearman_error_euclid"] = {k: float(v) for k, v in zip(("rho", "p"), spearmanr(di, err)[:2])}
        out["spearman_error_cosine"] = {k: float(v) for k, v in zip(("rho", "p"), spearmanr(cos, err)[:2])}
    class_cond = {}
    for cid in CORE_CLASSES:
        m = np.asarray(yt) == cid
        if m.sum() < 5:
            continue
        class_cond[int(cid)] = {
            "n": int(m.sum()),
            "oa_inside": float((pred[m & inside] == yt[m & inside]).mean()) if (m & inside).sum() else np.nan,
            "oa_outside": float((pred[m & ~inside] == yt[m & ~inside]).mean()) if (m & ~inside).sum() else np.nan,
            "n_inside": int((m & inside).sum()),
            "n_outside": int((m & ~inside).sum()),
        }
    out["class_conditioned_euclid"] = class_cond
    if proba is not None:
        out["maxprob_mean"] = float(proba.max(axis=1).mean())
        remap = {int(c): i for i, c in enumerate(model.classes_)}
        y_enc = np.array([remap.get(int(v), -1) for v in yt])
        m = y_enc >= 0
        out["ece"] = expected_calibration_error(y_enc[m], proba[m])
        tes = tes.copy()
        tes["maxprob"] = proba.max(axis=1)
        tes["di_euclid"] = di
        tes["di_cosine"] = cos
        tes["error"] = err
        tes["pred"] = pred
        tdir = resolve_path(load_config(), "tables")
        tdir.mkdir(parents=True, exist_ok=True)
        tes.drop(columns="geometry", errors="ignore").to_csv(tdir / "aoa_test_points.csv", index=False)
    return out


def e_rf_replication(df: pd.DataFrame, seed=20260902) -> dict:
    """Sample-level RF vs XGBoost on the locked test (representation robustness)."""
    ey = exact_year(df)
    test = ey[ey["split"] == "test"]
    train = ey[ey["split"] != "test"]
    hist = historical(df)
    hist = hist[hist["split"] != "test"]
    out = {}
    for group in ("M3_S1S2", "M4_AEF"):
        cols = select_feature_columns(df.columns, group)
        if len(cols) < 3:
            continue
        Xt, yt, _ = _xy(test, cols)
        Xtr, ytr, _ = _xy(train, cols)
        for est in ("xgb", "rf"):
            pred, _, _ = fit_predict(est, Xtr, ytr, Xt, seed=seed)
            out[f"{group}_{est}_field"] = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
        for mode in ("ungated", "hard"):
            extra = _hist_subset(hist, mode, high=0.80)
            Xh, yh, _ = _xy(pd.concat([train, extra]), cols)
            pred, _, _ = fit_predict("rf", Xh, yh, Xt, seed=seed)
            key = f"{group}_rf_{mode}"
            out[key] = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
            out[key]["n_train_hist"] = int(len(extra))
            out[key]["map_protocol"] = mode == "ungated"
    return out


def e12_map_2018(df: pd.DataFrame, protocol="ungated", high=0.80, seed=20260902) -> dict:
    """Track B: 2018 wall-to-wall validation on 2018 locked-test sites only."""
    ey = exact_year(df)
    year = pd.to_numeric(ey.get("year"), errors="coerce")
    test = ey[(ey["split"] == "test") & year.eq(2018)]
    train = ey[ey["split"] != "test"]
    hist = historical(df)
    hist = hist[hist["split"] != "test"]
    extra = _hist_subset(hist, protocol, high=high)
    cols = select_feature_columns(df.columns, "M4_AEF_TERRAIN") or select_feature_columns(df.columns, "M4_AEF")
    if len(cols) < 3 or test.empty:
        return {"error": "missing AEF or 2018 test"}
    Xt, yt, _ = _xy(test, cols)
    out = {
        "track": "B_2018_mapping",
        "protocol": protocol,
        "threshold": None if protocol not in {"hard", "soft"} else float(high),
        "n_test_2018": int(len(test)),
        "n_test_scored": int(len(yt)),
        "n_train_exact": int(len(train)),
        "n_train_hist": int(len(extra)),
        "class_support_2018_test": {
            int(k): int(v) for k, v in pd.to_numeric(test["class_id"], errors="coerce").value_counts().items()
        },
        "training_class_support": {
            int(k): int(v) for k, v in pd.to_numeric(
                pd.concat([train, extra])["class_id"], errors="coerce"
            ).value_counts().items()
        },
        "snow_ice_training_n": 0,
        "feature_group": "M4_AEF_TERRAIN" if "elev" in cols or "slope" in cols else "M4_AEF",
        "n_predictors": int(len(cols)),
        "note": "2017 field sites are excluded from this map-year score. Snow/ice has no field labels.",
    }
    blocks = test.loc[Xt.index, "block_id"].astype(str) if "block_id" in test.columns else None
    for est in ("xgb", "rf"):
        Xtr, ytr, _ = _xy(pd.concat([train, extra]), cols)
        pred, _, _ = fit_predict(est, Xtr, ytr, Xt, seed=seed)
        rep = classification_report_dict(yt, pred, labels=list(CORE_CLASSES))
        sup = classification_report_dict(yt, pred, labels=list(TRACK_B_SUPPORTED))
        if blocks is not None:
            bb = block_bootstrap_f1(yt, pred, blocks, TRACK_B_SUPPORTED, seed=seed)
            sup["macro_f1_ci"] = [bb["ci_lo"], bb["ci_hi"]]
            bb4 = block_bootstrap_f1(yt, pred, blocks, CORE_CLASSES, seed=seed)
            rep["macro_f1_ci"] = [bb4["ci_lo"], bb4["ci_hi"]]
        shrub = (rep.get("per_class") or {}).get(4) or {}
        rec = {
            **rep,
            "supported_labels": list(TRACK_B_SUPPORTED),
            "supported_macro_f1": sup.get("macro_f1"),
            "supported_macro_f1_ci": sup.get("macro_f1_ci"),
            "supported_oa": sup.get("oa"),
            "shrub_f1": shrub.get("f1"),
            "shrub_support": shrub.get("support"),
            "headline": "OA on all 2018 test sites; Macro-F1 on Cropland/Grassland/Bare only",
        }
        out[f"M4_AEF_{est}_{protocol}"] = rec
    out["headline_estimator"] = "rf"
    return out


def e12_products_2018(df: pd.DataFrame, seed=20260902, protocol="hard", high=0.90) -> dict:
    """Same 2018 locked-test sites, common ontology, supported classes as Track B."""
    ey = exact_year(df)
    year = pd.to_numeric(ey.get("year"), errors="coerce")
    test = ey[(ey["split"] == "test") & year.eq(2018)].copy()
    yt = _to_rangeland(pd.to_numeric(test.get("class_id"), errors="coerce"))
    labels = list(TRACK_B_SUPPORTED)
    out = {"n": int(len(test)), "labels": labels, "ontology": "common: crop / rangeland / bare"}
    blocks = test["block_id"].astype(str) if "block_id" in test.columns else None
    for col, name in (("dw_l1", "Dynamic World"), ("esri_l1", "Esri"), ("glc_l1", "GLC_FCS30D")):
        if col not in test.columns:
            continue
        yp = _to_rangeland(pd.to_numeric(test[col], errors="coerce"))
        m = yt.notna() & yp.notna() & (yp > 0)
        if int(m.sum()) < 8:
            continue
        rep = classification_report_dict(yt[m], yp[m], labels=labels)
        if blocks is not None:
            bb = block_bootstrap_f1(yt[m], yp[m], test.loc[yt[m].index, "block_id"].astype(str), labels, seed=seed)
            rep["macro_f1_ci"] = [bb["ci_lo"], bb["ci_hi"]]
        rep["n"] = int(m.sum())
        out[name] = rep
    cols = select_feature_columns(df.columns, "M4_AEF")
    if len(cols) >= 3 and not test.empty:
        train = ey[ey["split"] != "test"]
        hist = historical(df)
        hist = hist[hist["split"] != "test"]
        extra = _hist_subset(hist, protocol, high=high)
        Xt, yt_raw, _ = _xy(test, cols)
        Xtr, ytr, _ = _xy(pd.concat([train, extra]), cols)
        pred, _, _ = fit_predict("rf", Xtr, ytr, Xt, seed=seed)
        yt_r = _to_rangeland(yt_raw)
        yp_r = _to_rangeland(pd.Series(pred, index=yt_raw.index))
        m = yt_r.notna() & yp_r.notna()
        rep = classification_report_dict(yt_r[m], yp_r[m], labels=labels)
        if blocks is not None:
            bb = block_bootstrap_f1(
                yt_r[m], yp_r[m], test.loc[yt_r[m].index, "block_id"].astype(str), labels, seed=seed
            )
            rep["macro_f1_ci"] = [bb["ci_lo"], bb["ci_hi"]]
        rep["n"] = int(m.sum())
        rep["protocol"] = protocol
        rep["threshold"] = None if protocol not in {"hard", "soft"} else float(high)
        out["AEF + RF (this study)"] = rep
    return out


def e12_paired_product_deltas(
    df: pd.DataFrame,
    gee_pred: pd.Series | None = None,
    seed: int = 20260902,
) -> dict:
    """Paired block-bootstrap ΔMacro-F1 of AEF+GEE versus DW / GLC / Esri."""
    ey = exact_year(df)
    year = pd.to_numeric(ey.get("year"), errors="coerce")
    test = ey[(ey["split"] == "test") & year.eq(2018)].copy()
    if test.empty:
        return {"error": "no 2018 locked-test sites"}
    if "site_id" in test.columns:
        test = test.set_index(test["site_id"].astype(str), drop=False)
    yt = _to_rangeland(pd.to_numeric(test.get("class_id"), errors="coerce"))
    labels = list(TRACK_B_SUPPORTED)
    blocks = test["block_id"].astype(str)
    if gee_pred is None:
        return {"error": "GEE mosaic predictions missing", "n": int(len(test))}
    gee = pd.Series(gee_pred)
    gee.index = gee.index.astype(str)
    gee = _to_rangeland(pd.to_numeric(gee.reindex(yt.index.astype(str)), errors="coerce"))
    out = {"n": int(len(test)), "labels": labels, "ontology": "common: crop / rangeland / bare"}
    for col, name in (("dw_l1", "Dynamic World"), ("glc_l1", "GLC_FCS30D"), ("esri_l1", "Esri")):
        if col not in test.columns:
            continue
        yp = _to_rangeland(pd.to_numeric(test[col], errors="coerce"))
        m = yt.notna() & yp.notna() & gee.notna() & (yp > 0)
        if int(m.sum()) < 8:
            continue
        rec = block_bootstrap_delta(yt[m], gee[m], yp[m], blocks[m], labels, seed=seed)
        rec["n"] = int(m.sum())
        rec["contrast"] = f"AEF+GEE − {name}"
        out[name] = rec
    return out


def e12_lobo_product_deltas(df: pd.DataFrame, gee_pred: pd.Series) -> dict:
    """Leave-one-locked-block-out ΔMacro-F1 (AEF+GEE − product). Direction check only."""
    ey = exact_year(df)
    year = pd.to_numeric(ey.get("year"), errors="coerce")
    test = ey[(ey["split"] == "test") & year.eq(2018)].copy()
    test["site_id"] = test["site_id"].astype(str)
    test = test.set_index("site_id", drop=False)
    yt = _to_rangeland(pd.to_numeric(test.get("class_id"), errors="coerce"))
    labels = list(TRACK_B_SUPPORTED)
    gee = _to_rangeland(pd.to_numeric(pd.Series(gee_pred), errors="coerce"))
    gee.index = gee.index.astype(str)
    gee = gee.reindex(yt.index)
    blocks = test["block_id"].astype(str)
    products = (("dw_l1", "Dynamic World"), ("glc_l1", "GLC_FCS30D"), ("esri_l1", "Esri"))
    rows = []
    for blk in sorted(blocks.unique()):
        keep = blocks != blk
        rec = {"omitted_block": str(blk), "n_kept": int(keep.sum())}
        for col, name in products:
            if col not in test.columns:
                continue
            yp = _to_rangeland(pd.to_numeric(test[col], errors="coerce"))
            m = keep & yt.notna() & yp.notna() & gee.notna() & (yp > 0)
            if int(m.sum()) < 8:
                rec[name] = None
                continue
            fa = classification_report_dict(yt[m], gee[m], labels=labels)["macro_f1"]
            fb = classification_report_dict(yt[m], yp[m], labels=labels)["macro_f1"]
            rec[name] = float(fa - fb)
        rows.append(rec)
    esri = [r.get("Esri") for r in rows if r.get("Esri") is not None]
    return {
        "rows": rows,
        "n_blocks": int(blocks.nunique()),
        "esri_all_negative": bool(esri) and all(v < 0 for v in esri),
        "note": "Leave-one-block-out direction check; not a significance test.",
    }


def e3_classifier_robustness(df: pd.DataFrame, shots=HEADLINE_SHOTS, repeats=30, seed=20260902) -> dict:
    """Few-shot × {logreg, rf, xgb} × {L8, S1/S2, AEF} on the same headline draws as Fig. 4."""
    ey = exact_year(df)
    test = ey[ey["split"] == "test"]
    train = ey[ey["split"] != "test"]
    groups = [g for g in ("M1_L8", "M3_S1S2", "M4_AEF") if len(select_feature_columns(df.columns, g)) >= 3]
    clfs = ("logreg", "rf", "xgb")
    ytr_all = pd.to_numeric(train["class_id"], errors="coerce")
    yt_all = pd.to_numeric(test["class_id"], errors="coerce")
    shot_labels = [lab for lab in BENCHMARK_CLASSES if int((ytr_all == lab).sum()) >= 1 and int((yt_all == lab).sum()) >= 1]
    rng = np.random.default_rng(seed)
    raw = []
    for k in shots:
        for r in range(repeats):
            idx = []
            for lab in shot_labels:
                lab_idx = ytr_all.index[ytr_all == lab]
                take = min(k, len(lab_idx))
                if take:
                    idx.extend(rng.choice(lab_idx.to_numpy(), size=take, replace=False).tolist())
            if len(idx) < 8:
                continue
            for group in groups:
                cols = select_feature_columns(df.columns, group)
                Xt, yt, _ = _xy(test, cols)
                Xtr, ytr, _ = _xy(train.loc[idx], cols)
                if len(Xtr) < 8 or ytr.nunique() < 2:
                    continue
                for clf in clfs:
                    pred, _, _ = fit_predict(clf, Xtr, ytr, Xt, seed=seed + r)
                    raw.append({
                        "shots": k, "rep": r, "group": group, "classifier": clf,
                        "macro_f1": classification_report_dict(yt, pred, labels=shot_labels)["macro_f1"],
                    })
    tab = pd.DataFrame(raw)
    summary = []
    if not tab.empty:
        for keys, sub in tab.groupby(["shots", "group", "classifier"]):
            mean, lo, hi = bootstrap_ci(sub["macro_f1"], seed=seed)
            summary.append({
                "shots": int(keys[0]), "group": keys[1], "classifier": keys[2],
                "macro_f1": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(sub)),
            })
    return {"rows": summary, "n_repeats": repeats, "classifiers": list(clfs)}


def e6_nested_stability(df: pd.DataFrame, n_seeds=15, base=20260902, estimator="xgb",
                        target_year=None, feature_group=None) -> dict:
    """Repeat nested spatial CV across seeds; report selection frequency and repeated-CV 1-SE."""
    if feature_group is None:
        feature_group = "M4_AEF_TERRAIN" if estimator == "rf" and target_year == 2018 else "M4_AEF"
    picks = []
    means = {}
    for i in range(n_seeds):
        rec = e6_nested_select(
            df, estimator=estimator, seed=base + i * 19, target_year=target_year,
            feature_group=feature_group,
        )
        sel = rec.get("selected") or {}
        key = sel.get("mode")
        if key == "hard" and sel.get("threshold") is not None:
            key = f"hard_{sel.get('threshold')}"
        picks.append(key)
        for row in rec.get("table") or []:
            label = row["mode"] if row["mode"] != "hard" else f"hard_{row.get('threshold')}"
            means.setdefault(label, []).append(float(row["cv_macro_f1"]))
    freq = pd.Series(picks).value_counts(normalize=True).to_dict() if picks else {}
    stats = {}
    for lab, vals in means.items():
        arr = np.asarray(vals, float)
        stats[lab] = {
            "mean": float(np.nanmean(arr)),
            "sd": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "se": float(np.nanstd(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        }
    if stats:
        best = max(stats, key=lambda k: stats[k]["mean"])
        recommend, one_se, in_set = _one_se_recommend(stats)
        for lab, rec in stats.items():
            rec["within_one_se"] = lab in in_set
    else:
        best, recommend, one_se, in_set = None, None, np.nan, []
    return {
        "n_seeds": n_seeds,
        "selection_frequency": freq,
        "protocol_means": stats,
        "best_mean": best,
        "one_se_threshold": one_se,
        "one_se_set": in_set,
        "one_se_recommendation": recommend,
        "picks": picks,
        "estimator": estimator,
        "target_year": target_year,
        "feature_group": feature_group,
        "note": (
            "Repeated-CV 1-SE: among protocols whose mean CV Macro-F1 across 15 nested-CV seeds "
            "is within 1 SE of the best seed-level mean, prefer the simplest "
            "(None < Ungated < Soft < Hard). SE is the standard error of those seed-level means."
        ),
    }


def spatial_degradation(e2: dict, e4: pd.DataFrame) -> dict:
    """Compare development spatial-CV Macro-F1 vs locked-test 100% labels."""
    out = {}
    full = e4[e4["frac"] == 1.0] if isinstance(e4, pd.DataFrame) and "frac" in e4.columns else pd.DataFrame()
    for group in ("M3_S1S2", "M4_AEF"):
        cv = (e2.get(group) or {}).get("macro_f1_mean")
        locked = None
        if not full.empty:
            row = full[full["group"] == group]
            if len(row):
                locked = float(row.iloc[0]["macro_f1"])
        out[group] = {"dev_spatial_cv": cv, "locked_test_100": locked,
                      "drop": (None if cv is None or locked is None else float(cv) - locked)}
    return out


def _json_ready(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_redesign(df: pd.DataFrame, out_dir: Path | None = None) -> dict:
    cfg = load_config()
    out_dir = Path(out_dir) if out_dir else resolve_path(cfg, "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    tables.mkdir(exist_ok=True)

    e1 = e1_exact_year_products(df)
    e2 = e2_features(df)
    e3 = e3_fewshot(df)
    e4 = e4_percent(df)
    e6 = e6_temporal(df, high=cfg["migration"]["high_confidence"])
    e6s = e6_threshold_sensitivity(df)
    e6n = e6_nested_select(df)
    e6["nested_selection"] = e6n
    e7 = e7_repeated_holdout(df)
    e8 = e8_loco_matched(df)
    e8h = e8_historical_countries(df)
    e9 = e9_env(df)
    e10 = e10_pseudo(df)
    e11 = e11_uncertainty(df)
    erf = e_rf_replication(df)

    results = {
        "design": {
            "exact_year_years": [2017, 2018],
            "n_exact_year": int(len(exact_year(df))),
            "n_historical": int(len(historical(df))),
            "core_classes": CORE_CLASSES,
            "benchmark_classes": BENCHMARK_CLASSES,
            "block_crs": "EPSG:6933",
            "pseudo_labels": e10.get("status", "unknown"),
        },
        "E1_products": e1,
        "E2_features": e2,
        "E3_fewshot": e3,
        "E4_percent": e4,
        "E6_temporal": e6,
        "E6_threshold_sensitivity": e6s,
        "E7_repeated_holdout": e7,
        "E8_loco_matched": e8,
        "E8_historical_countries": e8h,
        "E9_env": e9,
        "E10_pseudo": e10,
        "E11_uncertainty": e11,
        "E_rf_replication": erf,
    }
    if isinstance(e3, dict) and e3.get("rows"):
        pd.DataFrame(e3["rows"]).to_csv(tables / "table_e3_fewshot.csv", index=False)
        pd.DataFrame(e3.get("paired_delta") or []).to_csv(tables / "table_e3_paired_delta.csv", index=False)
    elif isinstance(e3, pd.DataFrame):
        e3.to_csv(tables / "table_e3_fewshot.csv", index=False)
    if isinstance(e4, pd.DataFrame):
        e4.to_csv(tables / "table_e4_percent.csv", index=False)
    if isinstance(e8, pd.DataFrame):
        e8.to_csv(tables / "table_e8_cross_country.csv", index=False)
    if isinstance(e8h, pd.DataFrame) and not e8h.empty:
        e8h.to_csv(tables / "table_e8_historical_countries.csv", index=False)
    if isinstance(e6s, pd.DataFrame) and not e6s.empty:
        e6s.to_csv(tables / "table_e6_threshold.csv", index=False)
    if isinstance(e9, pd.DataFrame) and not e9.empty:
        e9.to_csv(tables / "table_e9_env.csv", index=False)
    (out_dir / "experiment_results_v2.json").write_text(
        json.dumps(_json_ready(results), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results

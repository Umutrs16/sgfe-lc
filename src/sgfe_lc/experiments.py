from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .aoa import area_of_applicability
from .config import load_config, resolve_path
from .legend import LEVEL1, LEVEL1_EVAL
from .metrics import (
    bootstrap_ci,
    classification_report_dict,
    expected_calibration_error,
    mcnemar_test,
    paired_wilcoxon,
    reports_to_table,
)
from .models import fit_predict, select_feature_columns


def _xy(df: pd.DataFrame, cols: list[str]):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["class_id"], errors="coerce")
    mask = y.notna() & X.notna().any(axis=1)
    return X.loc[mask], y.loc[mask].astype(int), mask


def e1_product_benchmark(df: pd.DataFrame, labels=None) -> dict:
    labels = labels or list(LEVEL1_EVAL)
    reports = {}
    product_cols = [c for c in df.columns if c.endswith("_l1_2018") or c in {
        "glc_l1_2018", "dw_l1_2018", "esri_l1_2018", "nab_l1_2018"
    }]
    for col in product_cols:
        yt = pd.to_numeric(df["class_id"], errors="coerce")
        yp = pd.to_numeric(df[col], errors="coerce")
        m = yt.notna() & yp.notna() & (yp > 0)
        reports[col] = classification_report_dict(yt[m], yp[m], labels=labels)
    return reports


def _spatial_cv(df, feature_cols, estimator="xgb", n_folds=5, seed=20260902):
    X, y, mask = _xy(df, feature_cols)
    groups = df.loc[mask, "block_id"].astype(str)
    folds = min(n_folds, max(2, groups.nunique()))
    gkf = GroupKFold(n_splits=folds)
    fold_reports = []
    preds = np.full(len(df), np.nan)
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        pred, proba, _ = fit_predict(estimator, X.iloc[tr], y.iloc[tr], X.iloc[te], seed=seed + i)
        idx = X.index[te]
        preds[df.index.get_indexer(idx)] = pred
        fold_reports.append(classification_report_dict(y.iloc[te], pred, labels=list(LEVEL1_EVAL)))
    return fold_reports, preds


def e2_feature_ablation(df: pd.DataFrame, estimator="xgb", seed=20260902) -> dict:
    train = df[df["split"] != "test"].copy()
    out = {}
    for group in ("M1_L8", "M2_S2", "M3_S1S2", "M4_AEF", "M5_AEF_ENV", "M6_AEF_PHENO"):
        cols = select_feature_columns(train.columns, group)
        if len(cols) < 3:
            out[group] = {"error": f"insufficient columns: {cols}"}
            continue
        folds, _ = _spatial_cv(train, cols, estimator=estimator, seed=seed)
        f1s = [r["macro_f1"] for r in folds]
        mean, lo, hi = bootstrap_ci(f1s, seed=seed)
        out[group] = {
            "n_features": len(cols),
            "folds": folds,
            "macro_f1_mean": mean,
            "macro_f1_ci": [lo, hi],
        }
    return out


def e3_label_efficiency(df: pd.DataFrame, fractions=None, repeats=10, estimator="xgb", seed=20260902) -> pd.DataFrame:
    cfg = load_config()
    fractions = fractions or cfg["label_efficiency"]["fractions"]
    repeats = repeats or cfg["label_efficiency"]["repeats"]
    test = df[df["split"] == "test"]
    if "tier" in df.columns:
        train = df[(df["split"] != "test") & (df["tier"].isin(["A", "B"]))]
    else:
        train = df[df["split"] != "test"]
    rows = []
    rng = np.random.default_rng(seed)
    for group in ("M1_L8", "M3_S1S2", "M4_AEF", "M5_AEF_ENV"):
        cols = select_feature_columns(df.columns, group)
        if len(cols) < 3 or test.empty or train.empty:
            continue
        Xt, yt, _ = _xy(test, cols)
        Xtr_all, ytr_all, _ = _xy(train, cols)
        for frac in fractions:
            scores = []
            for r in range(repeats):
                n = max(len(LEVEL1), int(round(len(Xtr_all) * frac)))
                n = min(n, len(Xtr_all))
                # stratified-ish sample
                idx = []
                for lab in ytr_all.unique():
                    lab_idx = ytr_all.index[ytr_all == lab]
                    take = max(1, int(round(len(lab_idx) * frac)))
                    take = min(take, len(lab_idx))
                    chosen = rng.choice(lab_idx.to_numpy(), size=take, replace=False)
                    idx.extend(chosen.tolist())
                if len(idx) < 8:
                    continue
                pred, _, _ = fit_predict(estimator, Xtr_all.loc[idx], ytr_all.loc[idx], Xt, seed=seed + r)
                rep = classification_report_dict(yt, pred, labels=list(LEVEL1_EVAL))
                scores.append(rep["macro_f1"])
            mean, lo, hi = bootstrap_ci(scores, seed=seed)
            rows.append({
                "group": group, "frac": frac, "n_repeats": len(scores),
                "macro_f1": mean, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def e4_migration_ablation(df: pd.DataFrame, estimator="xgb", seed=20260902) -> dict:
    test = df[df["split"] == "test"]
    if "tier" not in df.columns:
        return {"error": "tier column missing; run migration scoring first"}
    cols = select_feature_columns(df.columns, "M4_AEF") or select_feature_columns(df.columns, "M3_S1S2")
    if len(cols) < 3 or test.empty:
        return {"error": "features or test set missing"}
    Xt, yt, _ = _xy(test, cols)
    scenarios = {
        "no_migration": df[(df["split"] != "test") & (df["tier"] == "A")],
        "direct_migration": df[(df["split"] != "test") & (df["tier"].isin(["A", "B_ungated", "B"]))],
        "gated_migration": df[(df["split"] != "test") & (df["tier"].isin(["A", "B"]))],
    }
    reports = {}
    preds = {}
    for name, tr in scenarios.items():
        if tr.empty:
            reports[name] = {"n": 0}
            continue
        Xtr, ytr, _ = _xy(tr, cols)
        pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
        reports[name] = classification_report_dict(yt, pred, labels=list(LEVEL1_EVAL))
        preds[name] = pred
    tests = {}
    if "no_migration" in preds and "gated_migration" in preds:
        tests["gated_vs_none"] = mcnemar_test(yt, preds["gated_migration"], preds["no_migration"])
    if "direct_migration" in preds and "gated_migration" in preds:
        tests["gated_vs_direct"] = mcnemar_test(yt, preds["gated_migration"], preds["direct_migration"])
    return {"reports": reports, "mcnemar": tests}


def e5_pseudo_ablation(df: pd.DataFrame, estimator="xgb", seed=20260902) -> dict:
    test = df[df["split"] == "test"]
    cols = select_feature_columns(df.columns, "M4_AEF") or select_feature_columns(df.columns, "M3_S1S2")
    if "tier" not in df.columns or test.empty or len(cols) < 3:
        return {"error": "need tier + features"}
    Xt, yt, _ = _xy(test, cols)
    scenarios = {
        "field_only": df[(df["split"] != "test") & (df["tier"].isin(["A", "B"]))],
        "field_plus_consensus": df[(df["split"] != "test") & (df["tier"].isin(["A", "B", "C"]))],
        "field_plus_consensus_aoa": df[(df["split"] != "test") & (df["tier"].isin(["A", "B", "C"])) & (df.get("aoa_inside", True) != False)],
    }
    reports = {}
    for name, tr in scenarios.items():
        if tr.empty:
            reports[name] = {"n": 0}
            continue
        Xtr, ytr, _ = _xy(tr, cols)
        pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
        reports[name] = classification_report_dict(yt, pred, labels=list(LEVEL1_EVAL))
    return reports


def e6_loco(df: pd.DataFrame, estimator="xgb", seed=20260902) -> pd.DataFrame:
    countries = [c for c in ("KAZ", "KGZ", "TJK", "UZB") if (df["country"] == c).any()]
    rows = []
    for group in ("M3_S1S2", "M4_AEF"):
        cols = select_feature_columns(df.columns, group)
        if len(cols) < 3:
            continue
        for hold in countries:
            tr = df[(df["country"] != hold) & (df["country"].isin(countries)) & (df["split"] != "test")]
            te = df[df["country"] == hold]
            if len(tr) < 20 or len(te) < 8:
                continue
            Xtr, ytr, _ = _xy(tr, cols)
            Xt, yt, _ = _xy(te, cols)
            pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
            rep = classification_report_dict(yt, pred, labels=list(LEVEL1_EVAL))
            rows.append({"group": group, "test_country": hold, **{k: rep[k] for k in ("n", "oa", "macro_f1", "miou", "balanced_acc")}})
    return pd.DataFrame(rows)


def e7_ecoregion(df: pd.DataFrame, estimator="xgb", seed=20260902) -> pd.DataFrame:
    if "ecoregion" not in df.columns:
        return pd.DataFrame()
    cols = select_feature_columns(df.columns, "M4_AEF") or select_feature_columns(df.columns, "M3_S1S2")
    rows = []
    for zone in df["ecoregion"].dropna().unique():
        if "nan" in str(zone).lower():
            continue
        tr = df[(df["ecoregion"] != zone) & (df["split"] != "test")]
        te = df[df["ecoregion"] == zone]
        if len(tr) < 20 or len(te) < 8:
            continue
        Xtr, ytr, _ = _xy(tr, cols)
        Xt, yt, _ = _xy(te, cols)
        pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
        rep = classification_report_dict(yt, pred, labels=list(LEVEL1_EVAL))
        rows.append({"holdout_ecoregion": zone, **{k: rep[k] for k in ("n", "oa", "macro_f1", "miou")}})
    return pd.DataFrame(rows)


def e_aoa_contrast(df: pd.DataFrame, estimator="xgb", seed=20260902) -> dict:
    if "aoa_inside" not in df.columns:
        return {"error": "aoa_inside missing"}
    cols = select_feature_columns(df.columns, "M4_AEF") or select_feature_columns(df.columns, "M3_S1S2")
    test = df[df["split"] == "test"]
    train = df[df["split"] != "test"]
    if len(cols) < 3 or test.empty or train.empty:
        return {"error": "features or splits missing"}
    Xtr, ytr, _ = _xy(train, cols)
    Xt, yt, tmask = _xy(test, cols)
    pred, _, _ = fit_predict(estimator, Xtr, ytr, Xt, seed=seed)
    inside = test.loc[Xt.index, "aoa_inside"].astype(bool)
    out = {}
    for name, mask in (("inside", inside), ("outside", ~inside)):
        if mask.sum() < 5:
            out[name] = {"n": int(mask.sum())}
            continue
        out[name] = classification_report_dict(yt[mask], pred[mask.to_numpy()], labels=list(LEVEL1_EVAL))
    return out


def attach_aoa(df: pd.DataFrame, group="M4_AEF") -> pd.DataFrame:
    cols = select_feature_columns(df.columns, group)
    if len(cols) < 3:
        return df
    train = df[df["split"] != "test"]
    Xtr, _, _ = _xy(train, cols)
    Xall, _, mask = _xy(df, cols)
    di, inside, thr = area_of_applicability(Xtr, Xall)
    out = df.copy()
    out.loc[Xall.index, "aoa_di"] = di
    out.loc[Xall.index, "aoa_inside"] = inside
    out.attrs["aoa_threshold"] = thr
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


def run_all_experiments(df: pd.DataFrame, out_dir: Path | None = None) -> dict:
    cfg = load_config()
    out_dir = Path(out_dir) if out_dir else resolve_path(cfg, "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    tables.mkdir(exist_ok=True)

    results = {
        "E1_products": e1_product_benchmark(df),
        "E2_ablation": e2_feature_ablation(df),
        "E3_label_efficiency": e3_label_efficiency(df),
        "E4_migration": e4_migration_ablation(df),
        "E5_pseudo": e5_pseudo_ablation(df),
        "E6_loco": e6_loco(df),
        "E7_ecoregion": e7_ecoregion(df),
        "AOA_error": e_aoa_contrast(df),
    }
    if isinstance(results["E1_products"], dict) and results["E1_products"]:
        reports_to_table(results["E1_products"]).to_csv(tables / "table_e1_products.csv", index=False)
    if isinstance(results["E3_label_efficiency"], pd.DataFrame):
        results["E3_label_efficiency"].to_csv(tables / "table_e3_label_efficiency.csv", index=False)
    if isinstance(results["E6_loco"], pd.DataFrame):
        results["E6_loco"].to_csv(tables / "table_e6_loco.csv", index=False)
    if isinstance(results["E7_ecoregion"], pd.DataFrame) and not results["E7_ecoregion"].empty:
        results["E7_ecoregion"].to_csv(tables / "table_e7_ecoregion.csv", index=False)

    payload = _json_ready(results)
    (out_dir / "experiment_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results

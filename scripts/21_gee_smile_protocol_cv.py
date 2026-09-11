"""Single-seed 2018-targeted nested-CV protocol check with GEE Smile RF.

The Track B selector in e6_nested_select(..., estimator='rf') is sklearn
RandomForestClassifier. This script repeats that fold logic once (seed 20260902)
with ee.Classifier.smileRandomForest (400 trees) on year-matched A00–A63 + elev +
slope. It does not retune the locked protocol unless Smile RF no longer ranks
hard 0.60 first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.eval_matrix import (
    CORE_CLASSES,
    _hist_subset,
    _json_ready,
    classification_report_dict,
    exact_year,
    historical,
)
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import sample_year_matched_training
from sgfe_lc.gee.mapping import train_gee_rf


PREDICTORS = [f"A{i:02d}" for i in range(64)] + ["elev", "slope"]
CANDIDATES = [
    ("none", None),
    ("ungated", None),
    ("hard", 0.60),
    ("hard", 0.70),
    ("hard", 0.80),
    ("hard", 0.90),
]
SEED = 20260902


def _proto_key(mode, thr):
    if mode == "hard" and thr is not None:
        return f"hard_{float(thr):g}"
    return str(mode)


def _fc_from_rows(rows):
    feats = []
    for r in rows:
        props = {k: r[k] for k in PREDICTORS if k in r and r[k] is not None}
        props["site_id"] = str(r["site_id"])
        props["class_id"] = int(r["class_id"])
        feats.append(ee.Feature(None, props))
    return ee.FeatureCollection(feats)


def _score_fold(train_rows, val_rows, n_trees):
    if len(train_rows) < 8 or len(val_rows) < 5:
        return None
    ytr = {int(r["class_id"]) for r in train_rows}
    if len(ytr) < 2:
        return None
    clf = train_gee_rf(_fc_from_rows(train_rows), PREDICTORS, n_trees=n_trees)
    pred_fc = _fc_from_rows(val_rows).classify(clf)
    feats = pred_fc.getInfo().get("features") or []
    yt, yp = [], []
    for feat in feats:
        p = feat.get("properties") or {}
        if p.get("class_id") is None or p.get("classification") is None:
            continue
        yt.append(int(p["class_id"]))
        yp.append(int(p["classification"]))
    if len(yt) < 5:
        return None
    return float(classification_report_dict(yt, yp, labels=list(CORE_CLASSES))["macro_f1"])


def main():
    cfg = load_config()
    ee_mod = init_ee()
    del ee_mod
    gdf = gpd.read_file(resolve_path(cfg, "processed") / "sites_exact_year.gpkg")
    ey = exact_year(gdf)
    train_ey = ey[ey["split"] != "test"].copy()
    hist = historical(gdf)
    hist = hist[hist["split"] != "test"].copy()
    pool = pd.concat([train_ey, hist], ignore_index=True)
    out_path = resolve_path(cfg, "results") / "gee_smile_protocol_cv.json"
    store_path = resolve_path(cfg, "results") / "experiment_results_v2.json"

    print("sampling year-matched AEF + terrain for", len(pool), "train/dev rows...")
    sampled = sample_year_matched_training(
        pool,
        properties=["site_id", "class_id", "block_id", "year", "role"],
        scale=30,
        tile_scale=8,
    )
    raw = sampled.getInfo().get("features") or []
    feat_rows = []
    for feat in raw:
        p = feat.get("properties") or {}
        if p.get("site_id") is None or p.get("class_id") is None:
            continue
        if any(p.get(b) is None for b in PREDICTORS):
            continue
        p["site_id"] = str(p["site_id"])
        feat_rows.append(p)
    feat_df = pd.DataFrame(feat_rows)
    print("sampled usable rows", len(feat_df))
    if feat_df.empty:
        raise SystemExit("Year-matched GEE sampling returned no usable rows.")

    groups = train_ey["block_id"].astype(str)
    folds = min(5, max(2, groups.nunique()))
    gkf = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(train_ey))
    n_trees = int(cfg["models"]["gee_rf_trees"])

    payload = {
        "classifier": "ee.Classifier.smileRandomForest",
        "n_trees": n_trees,
        "seed": SEED,
        "feature_group": "M4_AEF_TERRAIN",
        "n_predictors": 66,
        "n_sampled": int(len(feat_df)),
        "note": (
            "Single nested-CV seed matching e6_nested_select(estimator='rf', seed=20260902, "
            "target_year=2018). Validation folds keep 2018 train/dev sites only; training "
            "blocks are spatially disjoint for every year and label source. Macro-F1 uses "
            "CORE_CLASSES, matching the sklearn RF selector."
        ),
        "table": [],
        "fold_scores": {},
    }
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    done = {r.get("key") for r in payload.get("table") or [] if r.get("cv_macro_f1") is not None}
    for mode, thr in CANDIDATES:
        key = _proto_key(mode, thr)
        if key in done:
            print("skip completed", key)
            continue
        scores = []
        for i, (tr_idx, te_idx) in enumerate(gkf.split(dummy, groups=groups)):
            tr = train_ey.iloc[tr_idx].copy()
            te = train_ey.iloc[te_idx].copy()
            val_blocks = set(te["block_id"].astype(str))
            tr = tr[~tr["block_id"].astype(str).isin(val_blocks)]
            te = te[pd.to_numeric(te.get("year"), errors="coerce").eq(2018)]
            extra = _hist_subset(hist, mode, high=thr if thr is not None else 0.80)
            if not extra.empty and "block_id" in extra.columns:
                extra = extra[~extra["block_id"].astype(str).isin(val_blocks)]
            train_ids = set(pd.concat([tr, extra])["site_id"].astype(str))
            val_ids = set(te["site_id"].astype(str))
            train_rows = feat_df[feat_df["site_id"].isin(train_ids)].to_dict("records")
            val_rows = feat_df[feat_df["site_id"].isin(val_ids)].to_dict("records")
            print(f"  {key} fold {i + 1}/{folds} n_train={len(train_rows)} n_val={len(val_rows)}")
            try:
                f1 = _score_fold(train_rows, val_rows, n_trees)
            except Exception as exc:
                print("    failed", exc)
                f1 = None
            print("    macro_f1", f1)
            if f1 is not None:
                scores.append(f1)
        mean = float(np.mean(scores)) if scores else float("nan")
        rec = {
            "key": key,
            "mode": mode,
            "threshold": thr,
            "cv_macro_f1": None if not np.isfinite(mean) else mean,
            "n_folds": int(len(scores)),
            "fold_scores": scores,
        }
        payload["fold_scores"][key] = scores
        payload["table"] = [r for r in (payload.get("table") or []) if r.get("key") != key] + [rec]
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", key, "mean", mean)

    tab = pd.DataFrame(payload.get("table") or [])
    tab = tab[tab["cv_macro_f1"].notna()]
    selected = None
    still_hard060 = False
    if not tab.empty:
        best = tab.loc[tab["cv_macro_f1"].idxmax()]
        selected = {
            "mode": best["mode"],
            "threshold": None if pd.isna(best.get("threshold")) else float(best["threshold"]),
            "cv_macro_f1": float(best["cv_macro_f1"]),
            "key": best["key"],
        }
        still_hard060 = str(best["key"]) in {"hard_0.6", "hard_0.60"}
    payload["selected"] = selected
    payload["still_selects_hard_0.60"] = still_hard060
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    store = json.loads(store_path.read_text(encoding="utf-8"))
    store["E6_gee_smile_protocol"] = payload
    store_path.write_text(json.dumps(_json_ready(store), indent=2, ensure_ascii=False), encoding="utf-8")
    print("selected", selected, "still_hard_0.60", still_hard060)
    return 0 if still_hard060 or selected is None else 2


if __name__ == "__main__":
    raise SystemExit(main())

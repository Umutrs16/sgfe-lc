"""Score the locked-protocol GEE RF at 2018 locked-test field sites."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.eval_matrix import (
    TRACK_B_SUPPORTED,
    CORE_CLASSES,
    _json_ready,
    _to_rangeland,
    block_bootstrap_f1,
    classification_report_dict,
    e12_paired_product_deltas,
    exact_year,
)
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import sample_year_matched_training, sites_to_fc
from sgfe_lc.gee.features import alphaearth_features, terrain_features
from sgfe_lc.gee.mapping import (
    mosaic_tile_folder,
    protocol_asset_tag,
    remask_unofficial_series,
    train_gee_rf,
)
from sgfe_lc.gee.roi import site_hull_roi


COUNTRIES = ["Kazakhstan", "Kyrgyzstan", "Tajikistan", "Uzbekistan", "Turkmenistan"]
PREDICTORS = [f"A{i:02d}" for i in range(64)] + ["elev", "slope"]


def _asset_or_none(ee, asset_id: str):
    try:
        info = ee.data.getAsset(asset_id)
    except Exception:
        return None
    return ee.Image(asset_id) if info else None


def load_official_mosaic(ee, cfg, suffix="hard060"):
    root = cfg["project"]["asset_root"]
    tag = f"_{suffix}" if suffix else ""
    imgs = []
    sources = []
    for name in COUNTRIES:
        key = name.replace(" ", "")
        aid = f"{root}/map_2018_{key}_30m{tag}"
        img = _asset_or_none(ee, aid)
        if img is None:
            folder = f"{root}/{key.lower()}_tiles_{suffix}"
            img = mosaic_tile_folder(folder)
            if img is None:
                print("missing", aid, "and", folder)
                continue
            sources.append(folder)
            print("tiles", folder)
        else:
            sources.append(aid)
            print("asset", aid)
        imgs.append(img)
    if not imgs:
        return None, []
    return ee.ImageCollection(imgs).mosaic().rename("class_id"), sources


def _locked_training(gdf, mode, thr):
    mask = gdf["split"] != "test" if "split" in gdf.columns else True
    w = pd.to_numeric(gdf.get("w_rel"), errors="coerce").fillna(0)
    hist_ok = gdf["role"] == "historical"
    if mode == "none":
        hist_ok = hist_ok & False
    elif mode == "hard":
        hist_ok = hist_ok & (w >= float(thr))
    return gdf[mask & ((gdf["role"] == "exact_year") | hist_ok)].copy()


def live_classify(ee, cfg, train, test):
    lon = pd.concat([train["longitude"], test["longitude"]], ignore_index=True)
    lat = pd.concat([train["latitude"], test["latitude"]], ignore_index=True)
    roi = site_hull_roi(lon, lat, buffer_m=8000)
    feat_2018 = alphaearth_features(2018, roi).addBands(terrain_features())
    sampled = sample_year_matched_training(
        train, properties=["class_id"], scale=30, tile_scale=8,
    )
    clf = train_gee_rf(sampled, PREDICTORS, n_trees=cfg["models"]["gee_rf_trees"])
    classified = feat_2018.select(PREDICTORS).classify(clf).rename("map_class")
    fc_te = sites_to_fc(test, properties=["site_id", "class_id", "block_id"])
    return classified.sampleRegions(
        collection=fc_te,
        properties=["site_id", "class_id", "block_id"],
        scale=30,
        geometries=False,
        tileScale=4,
    )


def main():
    cfg = load_config()
    ee = init_ee()
    gdf = gpd.read_file(resolve_path(cfg, "processed") / "sites_exact_year.gpkg")
    ey = exact_year(gdf)
    year = pd.to_numeric(ey.get("year"), errors="coerce")
    test = ey[(ey["split"] == "test") & year.eq(2018)].copy()
    if test.empty:
        raise SystemExit("No 2018 locked-test sites.")
    store_path = resolve_path(cfg, "results") / "experiment_results_v2.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    e12 = store.get("E12_map_2018") or {}
    mode = e12.get("protocol") or "ungated"
    thr = e12.get("threshold")
    if isinstance(mode, str) and mode.startswith("hard"):
        if thr in (None, "null") and "_" in mode:
            thr = float(mode.split("_", 1)[1])
        mode = "hard"
    suffix = protocol_asset_tag(mode, thr)
    force_live = "--live" in sys.argv
    mosaic, sources = (None, []) if force_live else load_official_mosaic(ee, cfg, suffix=suffix)
    source = "exported_mosaic"
    if mosaic is not None:
        print("scoring exported mosaic", suffix)
        sampled = mosaic.rename("map_class").sampleRegions(
            collection=sites_to_fc(test, properties=["site_id", "class_id", "block_id"]),
            properties=["site_id", "class_id", "block_id"],
            scale=30,
            geometries=False,
            tileScale=4,
        )
    else:
        train = _locked_training(gdf, mode, thr if thr not in (None, "null") else 0.6)
        print("year-matched live-classify", mode, thr, "n_train", len(train))
        sampled = live_classify(ee, cfg, train, test)
        source = "year_matched_live"
        sources = ["year_matched_smileRandomForest"]
    feats = sampled.getInfo().get("features") or []
    rows = []
    for feat in feats:
        p = feat.get("properties") or {}
        if p.get("site_id") is None or p.get("map_class") is None:
            continue
        rows.append({
            "site_id": p["site_id"],
            "pred": int(p["map_class"]),
            "block_id": str(p.get("block_id", "")),
        })
    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise SystemExit("GEE sampleRegions returned no points.")
    pred_df["pred"] = remask_unofficial_series(pred_df["pred"])
    # If class_id was overwritten by the image, truth is lost. Recover from site table.
    test = test.copy()
    test["site_id"] = test["site_id"].astype(str)
    pred_df["site_id"] = pred_df["site_id"].astype(str)
    merged = test.merge(pred_df[["site_id", "pred"]], on="site_id", how="left")
    # Recover truth from the local table, never from a possibly overwritten GEE property.
    yt = pd.to_numeric(merged["class_id"], errors="coerce")
    yp = pd.to_numeric(merged["pred"], errors="coerce")
    ok = yt.notna() & yp.notna()
    print("scored", int(ok.sum()), "of", len(test))
    if int(ok.sum()) < 8:
        raise SystemExit("Too few scored GEE map points.")
    blocks = merged.loc[ok, "block_id"].astype(str)
    yt_ok = yt[ok].astype(int)
    yp_ok = yp[ok].astype(int)
    four = classification_report_dict(yt_ok, yp_ok, labels=list(CORE_CLASSES))
    sup = classification_report_dict(yt_ok, yp_ok, labels=list(TRACK_B_SUPPORTED))
    bb = block_bootstrap_f1(yt_ok, yp_ok, blocks, TRACK_B_SUPPORTED, seed=20260902)
    bb4 = block_bootstrap_f1(yt_ok, yp_ok, blocks, CORE_CLASSES, seed=20260902)
    shrub = (four.get("per_class") or {}).get(4) or {}
    gee = {
        "source": source,
        "classifier": "ee.Classifier.smileRandomForest",
        "n_trees": int(cfg["models"]["gee_rf_trees"]),
        "predictors": PREDICTORS,
        "predictor_note": "64 AlphaEarth bands + SRTM elev + slope; 2017 exact-year labels use 2017 AEF, all other training rows use 2018 AEF; prediction on the 2018 mosaic",
        "protocol": mode,
        "threshold": None if mode not in {"hard", "soft"} else (None if thr in (None, "null") else float(thr)),
        "scale_m": 30,
        "assets": sources,
        "n_test_2018": int(len(test)),
        "n_scored": int(ok.sum()),
        "oa": four.get("oa"),
        "macro_f1": four.get("macro_f1"),
        "macro_f1_ci": [bb4["ci_lo"], bb4["ci_hi"]],
        "supported_labels": list(TRACK_B_SUPPORTED),
        "supported_macro_f1": sup.get("macro_f1"),
        "supported_macro_f1_ci": [bb["ci_lo"], bb["ci_hi"]],
        "supported_oa": sup.get("oa"),
        "shrub_f1": shrub.get("f1"),
        "shrub_support": shrub.get("support"),
        "per_class": four.get("per_class"),
        "headline": "OA and supported Macro-F1 from the locked-protocol GEE RF at 2018 locked-test sites",
        "official_legend": ["Cropland", "Grassland", "Shrubland", "Bare", "Built"],
        "unofficial_remasked": ["Forest", "Water", "Snow/ice"],
    }
    yt_r = _to_rangeland(yt_ok)
    yp_r = _to_rangeland(yp_ok)
    common = classification_report_dict(yt_r, yp_r, labels=list(TRACK_B_SUPPORTED))
    bbc = block_bootstrap_f1(yt_r, yp_r, blocks, TRACK_B_SUPPORTED, seed=20260902)
    common["macro_f1_ci"] = [bbc["ci_lo"], bbc["ci_hi"]]
    common["n"] = int(ok.sum())
    common["protocol"] = mode
    common["threshold"] = None if mode not in {"hard", "soft"} else (None if thr in (None, "null") else float(thr))
    common["source"] = source
    store["E12_gee_map_2018"] = gee
    products = store.get("E12_products_2018") or {}
    products["AEF + GEE RF (this study)"] = common
    products.pop("AEF + RF (this study)", None)
    store["E12_products_2018"] = products
    pred_path = resolve_path(cfg, "tables") / "gee_map_2018_preds.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_tab = merged.loc[ok, ["site_id", "block_id", "class_id"]].copy()
    pred_tab["gee_pred"] = yp_ok.to_numpy()
    pred_tab.to_csv(pred_path, index=False)
    gee_series = pd.Series(yp_ok.to_numpy(), index=merged.loc[ok, "site_id"].astype(str))
    store["E12_paired_product_deltas"] = e12_paired_product_deltas(gdf, gee_pred=gee_series)
    store_path.write_text(json.dumps(_json_ready(store), indent=2, ensure_ascii=False), encoding="utf-8")
    print("GEE OA", gee["oa"], "supported F1", gee["supported_macro_f1"], "common F1", common.get("macro_f1"))
    print("paired", json.dumps(_json_ready(store["E12_paired_product_deltas"]), ensure_ascii=True))
    print("wrote", store_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

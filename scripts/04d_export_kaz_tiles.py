"""Cancel the whole-country Kazakhstan export and replace it with 5-degree tiles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from sgfe_lc.config import PROJECT_ROOT, load_config, resolve_path
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import sample_year_matched_training
from sgfe_lc.gee.features import alphaearth_features, stack_observational_features, terrain_features
from sgfe_lc.gee.mapping import (
    classify_region,
    ensure_asset_folder,
    protocol_asset_tag,
    train_gee_rf,
)
from sgfe_lc.gee.roi import country_fc, site_hull_roi


def _training_frame(cfg):
    path = resolve_path(cfg, "processed") / "sites_exact_year.gpkg"
    gdf = gpd.read_file(path)
    mask = gdf["split"] != "test"
    store = json.loads((resolve_path(cfg, "results") / "experiment_results_v2.json").read_text(encoding="utf-8"))
    e12 = store.get("E12_map_2018") or {}
    mode = e12.get("protocol") or "ungated"
    if isinstance(mode, str) and mode.startswith("hard"):
        mode = "hard"
    thr = e12.get("threshold")
    if mode == "hard" and "_" in str(e12.get("protocol") or "") and thr in (None, "null"):
        thr = float(str(e12.get("protocol")).split("_", 1)[1])
    w = pd.to_numeric(gdf.get("w_rel"), errors="coerce").fillna(0)
    hist_ok = gdf["role"] == "historical"
    if mode == "none":
        hist_ok = hist_ok & False
    elif mode == "hard":
        thr = float(thr if thr not in (None, "null") else 0.9)
        hist_ok = hist_ok & (w >= thr)
    train = gdf[mask & ((gdf["role"] == "exact_year") | hist_ok)]
    n_hist = int((hist_ok & mask).sum())
    print("KAZ tiles protocol", mode, "threshold", thr if mode == "hard" else None,
          "n_hist", n_hist, "n_train", int(len(train)))
    return train, mode, thr


def _local_tiles(country="Kazakhstan", step=5):
    gaul = gpd.read_file(PROJECT_ROOT / "data" / "lookups" / "ca_gaul2015.geojson")
    sub = gaul[gaul["ADM0_NAME"].astype(str).str.contains(country, case=False, na=False)]
    if sub.empty:
        raise SystemExit(f"No GAUL polygon for {country}")
    geom = sub.union_all() if hasattr(sub, "union_all") else sub.unary_union
    minx, miny, maxx, maxy = geom.bounds
    tiles = []
    lon = int(minx // step * step)
    while lon < maxx:
        lat = int(miny // step * step)
        while lat < maxy:
            cell = box(lon, lat, lon + step, lat + step)
            if cell.intersects(geom) and cell.intersection(geom).area > 0.02:
                tiles.append((lon, lat, lon + step, lat + step))
            lat += step
        lon += step
    return tiles


def _cancel_monolith(ee, suffix):
    for op in ee.data.listOperations():
        meta = op.get("metadata") or {}
        desc = str(meta.get("description") or "")
        state = meta.get("state")
        if desc == f"SGFE_LC_2018_Kazakhstan_30m_{suffix}" and state in {"PENDING", "RUNNING"}:
            ee.data.cancelOperation(op["name"])
            print("cancelled monolith", desc, state, op.get("name"))


def main():
    cfg = load_config()
    ee = init_ee()
    train, mode, thr = _training_frame(cfg)
    suffix = protocol_asset_tag(mode, thr)
    _cancel_monolith(ee, suffix)

    predictors = [f"A{i:02d}" for i in range(64)] + ["elev", "slope"]
    sampled = sample_year_matched_training(train, properties=["class_id"], scale=30, tile_scale=8)
    clf = train_gee_rf(sampled, predictors, n_trees=cfg["models"]["gee_rf_trees"])

    tiles = _local_tiles("Kazakhstan", step=5)
    folder = f"{cfg['project']['asset_root']}/kazakhstan_tiles_{suffix}"
    ensure_asset_folder(folder)
    geom = country_fc(["Kazakhstan"]).geometry()
    print("protocol", mode, thr, "suffix", suffix, "tiles", len(tiles), "folder", folder)

    rec = {}
    for west, south, east, north in tiles:
        name = f"e{west:03d}n{south:03d}"
        box_geom = ee.Geometry.Rectangle([west, south, east, north], None, False)
        roi = geom.intersection(box_geom, 100)
        feats = stack_observational_features(2018, roi, groups=["aef", "terrain"])
        pred = classify_region(feats, clf, predictors).clip(roi)
        task = ee.batch.Export.image.toAsset(
            image=pred,
            description=f"SGFE_LC_2018_Kazakhstan_30m_{suffix}_{name}",
            assetId=f"{folder}/{name}",
            region=roi,
            scale=30,
            maxPixels=1e13,
            pyramidingPolicy={"class_id": "mode"},
        )
        task.start()
        st = task.status()
        rec[name] = {"id": task.id, "state": st.get("state"), "bbox": [west, south, east, north]}
        print(" ", name, task.id, st.get("state"), [west, south, east, north])

    out = resolve_path(cfg, "results") / f"gee_export_tasks_{suffix}_kaz_tiles.json"
    rec["_folder"] = folder
    rec["_n_train"] = int(len(train))
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

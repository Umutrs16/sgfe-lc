"""Train a GEE Random Forest on locked training sites and export 30 m country tiles + 10 m typical zones."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import sample_year_matched_training
from sgfe_lc.gee.mapping import (
    classify_region,
    export_country_map,
    protocol_asset_tag,
    typical_zone_exports,
    train_gee_rf,
)


def main():
    cfg = load_config()
    ee = init_ee()
    path = resolve_path(cfg, "processed") / "sites_exact_year.gpkg"
    if not path.exists():
        path = resolve_path(cfg, "processed") / "sites_with_features.gpkg"
    if not path.exists():
        path = resolve_path(cfg, "processed") / "sites_harmonized.gpkg"
    if not path.exists():
        print("Need processed sites first.")
        return 2
    gdf = gpd.read_file(path)
    mask = (gdf["split"] != "test") if "split" in gdf.columns else True
    mode, thr = "hard", 0.90
    if "role" in gdf.columns:
        # Mapping extras follow the RF 1-SE protocol (Track B), not the XGBoost science selector.
        store_path = resolve_path(cfg, "results") / "experiment_results_v2.json"
        mode = "hard"
        thr = 0.90
        if store_path.exists():
            rec = json.loads(store_path.read_text(encoding="utf-8"))
            e12 = rec.get("E12_map_2018") or {}
            stab = rec.get("E6_nested_stability_rf") or {}
            rec_mode = e12.get("protocol") or stab.get("one_se_recommendation") or mode
            if isinstance(rec_mode, str) and rec_mode.startswith("hard"):
                mode = "hard"
                if e12.get("threshold") not in (None, "null"):
                    thr = float(e12["threshold"])
                elif "_" in rec_mode:
                    thr = float(rec_mode.split("_", 1)[1])
            elif rec_mode in {"none", "ungated", "soft"}:
                mode = rec_mode
        w = pd.to_numeric(gdf.get("w_rel"), errors="coerce").fillna(0)
        hist_ok = gdf["role"] == "historical"
        if mode == "none":
            hist_ok = hist_ok & False
        elif mode == "hard":
            hist_ok = hist_ok & (w >= thr)
        train = gdf[mask & ((gdf["role"] == "exact_year") | hist_ok)]
        n_exact = int(((gdf["role"] == "exact_year") & mask).sum())
        n_hist = int((hist_ok & mask).sum())
        print("map protocol", mode, "threshold", thr if mode == "hard" else None,
              "n_exact", n_exact, "n_hist", n_hist)
        if mode == "hard" and abs(float(thr) - 0.9) < 1e-9:
            print("hard 0.90 historical extras", n_hist)
    elif "tier" in gdf.columns:
        train = gdf[mask & gdf["tier"].isin(["A", "B"])]
    else:
        year = pd.to_numeric(gdf["year"], errors="coerce")
        train = gdf[mask & year.isin([2017, 2018])]
    if "class_id" not in train.columns or train.empty:
        print("No training rows.")
        return 2

    aef_cols = [f"A{i:02d}" for i in range(64)]
    predictors = aef_cols + ["elev", "slope"]
    print("training sites", len(train), "tiers", train["tier"].value_counts().to_dict() if "tier" in train.columns else {})
    sampled = sample_year_matched_training(
        train, properties=["class_id"], scale=30, tile_scale=8,
    )
    clf = train_gee_rf(sampled, predictors, n_trees=cfg["models"]["gee_rf_trees"])
    suffix = protocol_asset_tag(mode, thr)

    tasks = {}
    # Kazakhstan is too large for a single asset; tiles are started by scripts/04d.
    for name in ("Kyrgyzstan", "Tajikistan", "Uzbekistan", "Turkmenistan"):
        try:
            tasks[name] = export_country_map(
                clf, predictors, year=2018, country=name, scale=30, groups=["aef", "terrain"], suffix=suffix
            )
        except Exception as e:
            print("FAILED country", name, e)
    try:
        zone_tasks = typical_zone_exports(
            clf, predictors, year=2018, scale=10, groups=["aef", "terrain"], suffix=suffix
        )
    except Exception as e:
        print("FAILED zones", e)
        zone_tasks = {}
    rec = {}
    print("Country export tasks:")
    for k, t in tasks.items():
        st = t.status()
        rec[k] = {"id": t.id, "state": st.get("state")}
        print(" ", k, t.id, st.get("state"))
    print("Typical-zone 10 m tasks:")
    for k, t in zone_tasks.items():
        st = t.status()
        rec[f"zone_{k}"] = {"id": t.id, "state": st.get("state")}
        print(" ", k, t.id, st.get("state"))
    out = resolve_path(cfg, "results") / f"gee_export_tasks_{suffix}.json"
    out.write_text(__import__("json").dumps(rec, indent=2), encoding="utf-8")
    print("Wrote", out)
    print("Asset folder:", cfg["project"]["asset_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

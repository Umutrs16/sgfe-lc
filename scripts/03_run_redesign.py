"""Assemble year-matched features, relock equal-area blocks, run redesigned experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.eval_matrix import run_redesign
from sgfe_lc.merge import assemble_year_matched, assign_roles_and_weights, write_gpkg
from sgfe_lc.spatial import environmental_stratum, lock_spatial_splits_constrained


def main():
    cfg = load_config()
    processed = resolve_path(cfg, "processed")
    sites = gpd.read_file(processed / "sites_harmonized.gpkg")
    year = pd.to_numeric(sites["year"], errors="coerce")
    eligible = year.isin([2017, 2018])
    sites = lock_spatial_splits_constrained(
        sites,
        block_km=cfg["splits"]["block_km"],
        test_fraction_blocks=cfg["splits"]["test_fraction_blocks"],
        n_folds=cfg["splits"]["n_folds"],
        seed=cfg["project"]["seed"],
        crs="EPSG:6933",
        eligible_mask=eligible,
        stratify_col="country",
        headline_classes=tuple(cfg["splits"].get("headline_classes", [1, 3, 4, 5])),
        min_train_per_class=int(cfg["splits"].get("min_train_per_class", 20)),
        min_test_per_class=int(cfg["splits"].get("min_test_per_class", 10)),
        max_test_site_fraction=float(cfg["splits"].get("max_test_site_fraction", 0.33)),
    )
    gdf = assemble_year_matched(sites, processed / "gee_exports")
    gdf = assign_roles_and_weights(gdf, high=cfg["migration"]["high_confidence"])
    if {"elev", "precip_annual"}.issubset(gdf.columns):
        gdf["env_stratum"] = environmental_stratum(gdf["elev"], gdf["precip_annual"])
    # Tier C stays supplementary and is loaded only inside E10 if the CSV exists.
    pseudo_path = processed / "pseudo_tier_c.csv"
    if pseudo_path.exists():
        extra = pd.read_csv(pseudo_path)
        extra["tier"] = "C"
        extra["role"] = "pseudo"
        extra["split"] = "traindev"
        extra["w_rel"] = extra.get("w_rel", 0.5)
        if "geometry" not in extra.columns and {"longitude", "latitude"}.issubset(extra.columns):
            extra = gpd.GeoDataFrame(
                extra,
                geometry=gpd.points_from_xy(extra["longitude"], extra["latitude"]),
                crs="EPSG:4326",
            )
        gdf = pd.concat([gdf, extra], ignore_index=True)
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        print("attached supplementary Tier C", int((gdf.get("tier") == "C").sum()))
    write_gpkg(gdf, processed / "sites_exact_year.gpkg")
    summary = {
        "n": int(len(gdf)),
        "exact_year": int((gdf["role"] == "exact_year").sum()),
        "historical": int((gdf["role"] == "historical").sum()),
        "test_exact": int(((gdf["role"] == "exact_year") & (gdf["split"] == "test")).sum()),
        "train_exact": int(((gdf["role"] == "exact_year") & (gdf["split"] != "test")).sum()),
        "aef_cols": int(sum(c.startswith("A") and c[1:].isdigit() for c in gdf.columns)),
        "has_dw_l1": "dw_l1" in gdf.columns,
        "has_esri_l1": "esri_l1" in gdf.columns,
        "has_glc_l1": "glc_l1" in gdf.columns,
    }
    (processed / "exact_year_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["has_dw_l1"] or summary["aef_cols"] < 32:
        print("Year-matched 2017 stacks look incomplete. Run scripts/02b_extract_year_2017.py first.")
        return 2
    if "--assemble-only" in sys.argv:
        print("Assemble-only done")
        return 0
    results = run_redesign(gdf)
    print("Wrote redesigned results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

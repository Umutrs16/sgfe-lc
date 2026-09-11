"""Merge GEE exports and run experiments E1–E7."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.experiments import attach_aoa, run_all_experiments
from sgfe_lc.figures import fig1_site_map, fig3_product_bars, fig4_label_efficiency, fig5_loco_heatmap, fig6_ablation_forest
from sgfe_lc.merge import assign_tiers, merge_feature_tables, write_gpkg
from sgfe_lc.spatial import ecoregion_bin


def main():
    cfg = load_config()
    processed = resolve_path(cfg, "processed")
    sites_path = processed / "sites_harmonized.gpkg"
    if not sites_path.exists():
        print("Missing processed sites. Run scripts/00_audit_sites.py")
        return 2
    gdf = gpd.read_file(sites_path)
    export_dir = processed / "gee_exports"
    files = []
    if export_dir.exists():
        files = [p for p in export_dir.glob("SGFE_*.csv") if "_part" not in p.name]
    if files:
        gdf = merge_feature_tables(gdf, files)
        gdf = assign_tiers(gdf, high=cfg["migration"]["high_confidence"], map_year=cfg["project"]["mapping_year"])
    else:
        print("WARNING: no GEE export CSVs in data/processed/gee_exports — running only on site attributes.")

    if {"elev", "precip_annual"}.issubset(gdf.columns):
        gdf["ecoregion"] = ecoregion_bin(gdf["elev"], gdf["precip_annual"])

    gdf = attach_aoa(gdf)
    write_gpkg(gdf, processed / "sites_with_features.gpkg")

    results = run_all_experiments(gdf)
    fig1_site_map(gdf)
    if results.get("E1_products"):
        fig3_product_bars(results["E1_products"])
    if hasattr(results.get("E3_label_efficiency"), "empty") and not results["E3_label_efficiency"].empty:
        fig4_label_efficiency(results["E3_label_efficiency"])
    if hasattr(results.get("E6_loco"), "empty") and not results["E6_loco"].empty:
        fig5_loco_heatmap(results["E6_loco"])
    e4 = results.get("E4_migration") or {}
    scores = {}
    for name, rep in (e4.get("reports") or {}).items():
        if isinstance(rep, dict) and "macro_f1" in rep:
            scores[name] = rep["macro_f1"]
    if scores:
        fig6_ablation_forest(scores)
    print("Wrote results to", resolve_path(cfg, "results"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

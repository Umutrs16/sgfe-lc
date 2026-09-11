"""Export site-level Landsat/S1/S2/AEF/product/migration tables from GEE to Drive."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import export_migration_scores, export_product_labels, export_site_features
import geopandas as gpd


def main():
    cfg = load_config()
    init_ee()
    path = resolve_path(cfg, "processed") / "sites_harmonized.gpkg"
    if not path.exists():
        print("Run scripts/00_audit_sites.py first.")
        return 2
    gdf = gpd.read_file(path)
    year = cfg["project"]["mapping_year"]
    t1 = export_site_features(gdf, year=year)
    t2 = export_product_labels(gdf, years=(2013, 2014, 2017, 2018))
    t3 = export_migration_scores(gdf[gdf["year"].isin([2013, 2014])], survey_years=(2013, 2014), map_year=year)
    print("Started Drive export tasks:")
    for bag in (t1, t2, t3):
        for k, task in bag.items():
            print(" ", k, task.id, task.status().get("state"))
    print("Download CSVs from Google Drive folder:", cfg["project"]["drive_folder"])
    print("Then place them in data/processed/gee_exports/ and run scripts/03_run_experiments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sample all site-level GEE features and download CSVs locally."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import extract_all_local


def load_sites():
    cfg = load_config()
    processed = resolve_path(cfg, "processed")
    gpkg = processed / "sites_harmonized.gpkg"
    csv = processed / "sites_harmonized.csv"
    if gpkg.exists():
        return gpd.read_file(gpkg)
    df = pd.read_csv(csv)
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )


def main():
    cfg = load_config()
    init_ee()
    gdf = load_sites()
    print("sites", len(gdf), "project", cfg["project"]["gee_project"])
    out_dir = resolve_path(cfg, "processed") / "gee_exports"
    paths = extract_all_local(gdf, out_dir, year=cfg["project"]["mapping_year"])
    for k, p in paths.items():
        print("DONE", k, p, p.stat().st_size if p.exists() else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sample GLC_FCS30D 2015 and 2016 at all field sites."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import extract_bundle_local
from sgfe_lc.gee.products import glc_fcs30d_year


def main():
    cfg = load_config()
    init_ee()
    processed = resolve_path(cfg, "processed")
    gdf = gpd.read_file(processed / "sites_harmonized.gpkg")
    out_dir = processed / "gee_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in (2015, 2016):
        dest = out_dir / f"SGFE_GLC{year}.csv"
        print("===", dest.name, "sites", len(gdf), "===")
        extract_bundle_local(
            gdf,
            lambda roi, y=year: glc_fcs30d_year(y, roi),
            dest,
            chunk=300,
            scale=30,
        )
        print("wrote", dest, dest.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

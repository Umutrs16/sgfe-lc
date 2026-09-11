"""Extract 2017 exact-year features/products for 2017 field sites."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import extract_year_local


def main():
    cfg = load_config()
    init_ee()
    sites = gpd.read_file(resolve_path(cfg, "processed") / "sites_harmonized.gpkg")
    year = pd.to_numeric(sites["year"], errors="coerce")
    sub = sites.loc[year.eq(2017)].copy()
    print("2017 sites", len(sub))
    out = resolve_path(cfg, "processed") / "gee_exports"
    paths = extract_year_local(sub, out, year=2017)
    for k, p in paths.items():
        print("DONE", k, p, p.stat().st_size if p.exists() else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

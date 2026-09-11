"""Attach GLC 2015/2016 to the locked site table and recompute field-anchored weights."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.merge import assign_roles_and_weights, write_gpkg


def _attach(gdf, path: Path, year: int):
    tab = pd.read_csv(path)
    col = f"glc_l1_{year}"
    if col not in tab.columns:
        raise SystemExit(f"{path} missing {col}: {list(tab.columns)}")
    tab = tab[["site_id", col]].drop_duplicates("site_id")
    tab["site_id"] = tab["site_id"].astype(str)
    out = gdf.copy()
    out["site_id"] = out["site_id"].astype(str)
    if col in out.columns:
        out = out.drop(columns=[col])
    return out.merge(tab, on="site_id", how="left")


def main():
    cfg = load_config()
    processed = resolve_path(cfg, "processed")
    path = processed / "sites_exact_year.gpkg"
    gdf = gpd.read_file(path)
    exports = processed / "gee_exports"
    for year in (2015, 2016):
        gdf = _attach(gdf, exports / f"SGFE_GLC{year}.csv", year)
        print("attached", year, "non-null", int(pd.to_numeric(gdf[f"glc_l1_{year}"], errors="coerce").notna().sum()))
    pseudo = gdf[gdf.get("role") == "pseudo"].copy() if "role" in gdf.columns else gdf.iloc[0:0]
    core = gdf[gdf.get("role") != "pseudo"].copy() if "role" in gdf.columns else gdf
    old_w = pd.to_numeric(core.get("w_rel"), errors="coerce")
    core = assign_roles_and_weights(core, high=cfg["migration"]["high_confidence"])
    new_w = pd.to_numeric(core.get("w_rel"), errors="coerce")
    hist = core[(core["role"] == "historical") & (core["split"] != "test")]
    print("hard090 extras", int((pd.to_numeric(hist["w_rel"], errors="coerce") >= 0.90).sum()))
    print("w_rel changed", int((old_w.reindex(core.index).fillna(-1) - new_w.fillna(-1)).abs().gt(1e-9).sum()))
    if len(pseudo):
        gdf = pd.concat([core, pseudo], ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=getattr(core, "crs", "EPSG:4326"))
    else:
        gdf = core
    write_gpkg(gdf, path)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

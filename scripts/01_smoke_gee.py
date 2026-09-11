"""Verify GEE project access and key catalogs used by the paper."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgfe_lc.config import gee_project, load_config
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.roi import central_asia_roi, country_fc
from sgfe_lc.gee.features import alphaearth_features
from sgfe_lc.gee.products import dynamic_world_annual, esri_annual, glc_fcs30d_year


PROBE_POINTS = {
    "Almaty": (76.8512, 43.2220),
    "Tashkent": (69.2401, 41.2995),
    "Bishkek": (74.5698, 42.8746),
    "Dushanbe": (68.7870, 38.5598),
    "Ashgabat_extrapolation": (58.3833, 37.9500),
}


def main():
    cfg = load_config()
    ee = init_ee()
    print("GEE_PROJECT", gee_project(cfg))
    roi = central_asia_roi()
    area_km2 = roi.area(maxError=1000).divide(1e6).getInfo()
    print(f"ROI_area_km2 {area_km2:.0f}")

    aef = alphaearth_features(2018, roi)
    dw = dynamic_world_annual(2018, roi)
    glc = glc_fcs30d_year(2018, roi)
    esri = esri_annual(2018, roi)
    stack = aef.select(["A00", "A01"]).addBands(dw.select(["dw_l1_2018", "dw_maxprob"])).addBands(glc).addBands(esri)

    rows = []
    for name, (lon, lat) in PROBE_POINTS.items():
        pt = ee.Geometry.Point([lon, lat])
        sample = stack.reduceRegion(ee.Reducer.first(), pt, 30, bestEffort=True, tileScale=4).getInfo()
        sample["site"] = name
        rows.append(sample)
        print(name, sample)
    print("SMOKE_OK", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

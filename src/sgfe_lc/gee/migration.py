from __future__ import annotations

import ee

from .features import _scale_l8
from .products import glc_fcs30d_year


def monthly_ndvi(year: int, roi: ee.Geometry) -> ee.Image:
    months = []
    for m in range(1, 13):
        start = ee.Date.fromYMD(year, m, 1)
        end = start.advance(1, "month")
        col = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(roi)
            .filterDate(start, end)
            .map(_scale_l8)
            .select("NDVI")
        )
        months.append(col.median().rename(f"ndvi_{year}_{m:02d}"))
    img = months[0]
    for p in months[1:]:
        img = img.addBands(p)
    return img


def stability_scores(survey_year: int, map_year: int, roi: ee.Geometry) -> ee.Image:
    """Export NDVI change / phenology scores plus GLC self-persistence.

    Official S_product used in training weights is recomputed in merge.py as
    field-anchored agreement with the survey class. The GEE S_product band is
    retained only as S_product_self.
    """
    glc_s = glc_fcs30d_year(survey_year, roi).rename("glc_s")
    glc_m = glc_fcs30d_year(map_year, roi).rename("glc_m")
    same = glc_s.eq(glc_m).And(glc_s.gt(0))
    # consecutive years 2015-2018 if mapping 2018
    years = list(range(min(survey_year, map_year), max(survey_year, map_year) + 1))
    stable = same
    prev = glc_s
    for y in years[1:]:
        cur = glc_fcs30d_year(y, roi)
        stable = stable.And(cur.eq(prev))
        prev = cur
    s_product = stable.rename("S_product")

    ndvi_s = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(f"{survey_year}-01-01", f"{survey_year + 1}-01-01")
        .map(_scale_l8)
        .select("NDVI")
        .median()
    )
    ndvi_m = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(f"{map_year}-01-01", f"{map_year + 1}-01-01")
        .map(_scale_l8)
        .select("NDVI")
        .median()
    )
    delta = ndvi_m.subtract(ndvi_s).abs()
    # 0.12 threshold from config default, mapped linearly to [0,1]
    s_change = ee.Image(1).subtract(delta.divide(0.25)).clamp(0, 1).rename("S_change")

    amp_s = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(f"{survey_year}-01-01", f"{survey_year + 1}-01-01")
        .map(_scale_l8)
        .select("NDVI")
        .reduce(ee.Reducer.percentile([10, 90]))
    )
    amp_m = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(f"{map_year}-01-01", f"{map_year + 1}-01-01")
        .map(_scale_l8)
        .select("NDVI")
        .reduce(ee.Reducer.percentile([10, 90]))
    )
    amp_s_v = amp_s.select("NDVI_p90").subtract(amp_s.select("NDVI_p10"))
    amp_m_v = amp_m.select("NDVI_p90").subtract(amp_m.select("NDVI_p10"))
    amp_diff = amp_s_v.subtract(amp_m_v).abs()
    s_pheno = ee.Image(1).subtract(amp_diff.divide(0.35)).clamp(0, 1).rename("S_pheno")

    c_mig = s_product.multiply(s_change).multiply(s_pheno).rename("C_mig")
    return s_product.addBands([s_change, s_pheno, c_mig, delta.rename("ndvi_abs_delta")])

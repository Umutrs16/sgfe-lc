from __future__ import annotations

import ee

from ..legend import DW_BANDS, DW_TO_L1, ESRI_TO_L1, glc_remap_lists


def glc_fcs30d_year(year: int, roi: ee.Geometry) -> ee.Image:
    """Annual GLC_FCS30D (2000–2022) as Level-1 classes.

    Community catalog tiles store 23 bands b1–b23 = 2000–2022.
    """
    if year < 2000 or year > 2022:
        raise ValueError(f"GLC_FCS30D annual band not defined for {year}")
    band = f"b{year - 1999}"
    raw = (
        ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual")
        .filterBounds(roi)
        .mosaic()
        .select(band)
        .rename("glc_raw")
    )
    from_codes, to_codes = glc_remap_lists()
    return raw.remap(from_codes, to_codes, 0).rename(f"glc_l1_{year}").clip(roi)


def dynamic_world_annual(year: int, roi: ee.Geometry) -> ee.Image:
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(DW_BANDS)
    )
    prob = dw.mean().rename([f"dw_{b}" for b in DW_BANDS])
    label = prob.toArray().arrayArgmax().arrayGet([0]).rename("dw_argmax")
    # remap DW argmax 0-8 to Level-1
    dw_ids = [DW_TO_L1[b] for b in DW_BANDS]
    l1 = label.remap(list(range(len(DW_BANDS))), dw_ids, 0).rename(f"dw_l1_{year}")
    maxp = prob.reduce(ee.Reducer.max()).rename("dw_maxprob")
    entropy = _prob_entropy(prob, prefix="dw")
    return l1.addBands([maxp, entropy]).addBands(prob).clip(roi)


def esri_annual(year: int, roi: ee.Geometry) -> ee.Image:
    col = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS").filterBounds(roi)
    # system:index like '38T_2018'
    filtered = col.filter(ee.Filter.stringContains("system:index", f"_{year}"))
    raw = filtered.mosaic().select("b1").rename("esri_raw")
    from_c = list(ESRI_TO_L1.keys())
    to_c = [ESRI_TO_L1[k] for k in from_c]
    return raw.remap(from_c, to_c, 0).rename(f"esri_l1_{year}").clip(roi)


def naboureh_year(year: int, asset_id: str | None, roi: ee.Geometry) -> ee.Image | None:
    if not asset_id:
        return None
    img = ee.Image(asset_id).rename(f"nab_l1_{year}")
    return img.clip(roi)


def product_stack(year: int, roi: ee.Geometry, naboureh_asset: str | None = None) -> ee.Image:
    parts = [
        glc_fcs30d_year(year, roi),
        dynamic_world_annual(year, roi),
        esri_annual(year, roi),
    ]
    nab = naboureh_year(year, naboureh_asset, roi)
    if nab is not None:
        parts.append(nab)
    img = parts[0]
    for p in parts[1:]:
        img = img.addBands(p)
    return img


def product_agreement(year: int, roi: ee.Geometry, naboureh_asset: str | None = None) -> ee.Image:
    """Mode and vote entropy of exact-year Level-1 products (training prior only)."""
    glc = glc_fcs30d_year(year, roi).rename("p1")
    dw = dynamic_world_annual(year, roi).select(f"dw_l1_{year}").rename("p2")
    esri = esri_annual(year, roi).rename("p3")
    stack = glc.addBands([dw, esri])
    if naboureh_asset:
        stack = stack.addBands(naboureh_year(year, naboureh_asset, roi).rename("p4"))
    mode = stack.reduce(ee.Reducer.mode()).rename("product_mode")
    count = ee.Image(0)
    for b in stack.bandNames().getInfo() if False else ["p1", "p2", "p3"] + (["p4"] if naboureh_asset else []):
        count = count.add(stack.select(b).eq(mode))
    count = count.rename("product_agree_n")
    # disagreement: 1 - agree_fraction
    nprod = 4 if naboureh_asset else 3
    agree_frac = count.divide(nprod).rename("product_agree_frac")
    return mode.addBands([count, agree_frac])


def _prob_entropy(prob: ee.Image, prefix: str) -> ee.Image:
    safe = prob.max(1e-6)
    ent = safe.multiply(safe.log()).reduce(ee.Reducer.sum()).multiply(-1)
    return ent.rename(f"{prefix}_entropy")

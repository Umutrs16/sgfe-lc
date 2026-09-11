from __future__ import annotations

import ee

SEASONS = {
    "djf": ("-12-01", "-03-01", -1, 0),  # handled separately
    "mam": ("-03-01", "-06-01", 0, 0),
    "jja": ("-06-01", "-09-01", 0, 0),
    "son": ("-09-01", "-12-01", 0, 0),
}


def _scale_l8(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0)).And(
        qa.bitwiseAnd(1 << 1).eq(0)
    )
    optical = img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).multiply(0.0000275).add(-0.2)
    optical = optical.updateMask(mask).updateMask(optical.gt(0).And(optical.lt(1)))
    blue = optical.select("SR_B2")
    green = optical.select("SR_B3")
    red = optical.select("SR_B4")
    nir = optical.select("SR_B5")
    swir1 = optical.select("SR_B6")
    swir2 = optical.select("SR_B7")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename("EVI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI")
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI")
    bsi = swir1.add(red).subtract(nir.add(blue)).divide(swir1.add(red).add(nir.add(blue))).rename("BSI")
    nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")
    savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename("SAVI")
    out = optical.addBands([ndvi, evi, ndwi, mndwi, ndbi, bsi, nbr, savi])
    return out.copyProperties(img, ["system:time_start"])


def _scale_s2(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(1)).And(scl.neq(0))
    optical = img.select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]).multiply(0.0001)
    optical = optical.updateMask(mask).updateMask(optical.gt(0).And(optical.lt(1)))
    blue = optical.select("B2")
    green = optical.select("B3")
    red = optical.select("B4")
    re1 = optical.select("B5")
    nir = optical.select("B8")
    swir1 = optical.select("B11")
    swir2 = optical.select("B12")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename("EVI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI")
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI")
    bsi = swir1.add(red).subtract(nir.add(blue)).divide(swir1.add(red).add(nir.add(blue))).rename("BSI")
    ndre = nir.subtract(re1).divide(nir.add(re1)).rename("NDRE")
    out = optical.addBands([ndvi, evi, ndwi, mndwi, ndbi, bsi, ndre])
    return out.copyProperties(img, ["system:time_start"])


def _season_filter(year: int, season: str) -> ee.Filter:
    if season == "djf":
        start = ee.Date.fromYMD(year - 1, 12, 1)
        end = ee.Date.fromYMD(year, 3, 1)
    elif season == "mam":
        start = ee.Date.fromYMD(year, 3, 1)
        end = ee.Date.fromYMD(year, 6, 1)
    elif season == "jja":
        start = ee.Date.fromYMD(year, 6, 1)
        end = ee.Date.fromYMD(year, 9, 1)
    elif season == "son":
        start = ee.Date.fromYMD(year, 9, 1)
        end = ee.Date.fromYMD(year, 12, 1)
    else:
        raise ValueError(season)
    return ee.Filter.date(start, end)


def _stats(imgcol: ee.ImageCollection, prefix: str, bands: list[str]) -> ee.Image:
    sub = imgcol.select(bands)
    median = sub.median().rename([f"{prefix}{b}_med" for b in bands])
    p10 = sub.reduce(ee.Reducer.percentile([10])).rename([f"{prefix}{b}_p10" for b in bands])
    p90 = sub.reduce(ee.Reducer.percentile([90])).rename([f"{prefix}{b}_p90" for b in bands])
    std = sub.reduce(ee.Reducer.stdDev()).rename([f"{prefix}{b}_std" for b in bands])
    mx = sub.max().rename([f"{prefix}{b}_max" for b in bands])
    mn = sub.min().rename([f"{prefix}{b}_min" for b in bands])
    amp = mx.subtract(mn).rename([f"{prefix}{b}_amp" for b in bands])
    return median.addBands([p10, p90, std, amp])


def landsat8_features(year: int, roi: ee.Geometry) -> ee.Image:
    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .map(_scale_l8)
    )
    annual_bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                    "NDVI", "EVI", "NDWI", "MNDWI", "NDBI", "BSI", "NBR", "SAVI"]
    img = _stats(col, "l8_", annual_bands)
    for season in ("mam", "jja", "son"):
        scol = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(roi)
            .filter(_season_filter(year, season))
            .map(_scale_l8)
        )
        med = scol.select(["NDVI", "EVI", "NDWI", "BSI", "SR_B5", "SR_B6"]).median()
        med = med.rename([f"l8_{season}_{b}" for b in ["NDVI", "EVI", "NDWI", "BSI", "NIR", "SWIR1"]])
        img = img.addBands(med)
    return img


def sentinel2_features(year: int, roi: ee.Geometry) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
        .map(_scale_s2)
    )
    annual_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
                    "NDVI", "EVI", "NDWI", "MNDWI", "NDBI", "BSI", "NDRE"]
    img = _stats(col, "s2_", annual_bands)
    for season in ("mam", "jja", "son"):
        scol = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filter(_season_filter(year, season))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
            .map(_scale_s2)
        )
        med = scol.select(["NDVI", "EVI", "NDRE", "B5", "B8", "B11"]).median()
        med = med.rename([f"s2_{season}_{b}" for b in ["NDVI", "EVI", "NDRE", "RE1", "NIR", "SWIR1"]])
        img = img.addBands(med)
    return img


def sentinel1_features(year: int, roi: ee.Geometry) -> ee.Image:
    def prep(img):
        return img.select(["VV", "VH"]).toFloat()

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .map(prep)
    )
    med = col.median()
    ratio = med.select("VV").subtract(med.select("VH")).rename("s1_VV_minus_VH")
    quot = med.select("VV").subtract(med.select("VH")).rename("s1_VV_VH_diff")
    annual = med.rename(["s1_VV_med", "s1_VH_med"]).addBands([ratio, quot])
    amp = col.max().subtract(col.min()).rename(["s1_VV_amp", "s1_VH_amp"])
    img = annual.addBands(amp)
    for season in ("mam", "jja", "son"):
        smed = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filter(_season_filter(year, season))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select(["VV", "VH"])
            .median()
            .rename([f"s1_{season}_VV", f"s1_{season}_VH"])
        )
        img = img.addBands(smed)
    return img


def alphaearth_features(year: int, roi: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(roi)
        .mosaic()
        .select([f"A{i:02d}" for i in range(64)])
    )


def terrain_features() -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").rename("elev")
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem)
    asin = aspect.multiply(3.14159265 / 180).sin().rename("aspect_sin")
    acos = aspect.multiply(3.14159265 / 180).cos().rename("aspect_cos")
    return dem.addBands([slope, asin, acos])


def climate_features(year: int, roi: ee.Geometry) -> ee.Image:
    precip = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select("precipitation")
        .sum()
        .rename("precip_annual")
    )
    t2m = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select("temperature_2m")
        .mean()
        .subtract(273.15)
        .rename("t2m_mean")
    )
    return precip.addBands(t2m)


def water_aux() -> ee.Image:
    return ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select(["occurrence", "seasonality"]).rename(
        ["jrc_occurrence", "jrc_seasonality"]
    )


def landsat8_features_lite(year: int, roi: ee.Geometry) -> ee.Image:
    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .map(_scale_l8)
    )
    med = col.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "NDVI", "EVI", "NDWI", "BSI"]).median()
    med = med.rename(["l8_B2_med", "l8_B3_med", "l8_B4_med", "l8_B5_med", "l8_B6_med", "l8_B7_med",
                      "l8_NDVI_med", "l8_EVI_med", "l8_NDWI_med", "l8_BSI_med"])
    ndvi = col.select("NDVI")
    p10 = ndvi.reduce(ee.Reducer.percentile([10])).rename("l8_NDVI_p10")
    p90 = ndvi.reduce(ee.Reducer.percentile([90])).rename("l8_NDVI_p90")
    amp = p90.subtract(p10).rename("l8_NDVI_amp")
    img = med.addBands([p10, p90, amp])
    for season in ("mam", "jja", "son"):
        smed = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(roi)
            .filter(_season_filter(year, season))
            .map(_scale_l8)
            .select(["NDVI", "EVI", "BSI"])
            .median()
            .rename([f"l8_{season}_NDVI", f"l8_{season}_EVI", f"l8_{season}_BSI"])
        )
        img = img.addBands(smed)
    return img


def sentinel2_features_lite(year: int, roi: ee.Geometry) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(_scale_s2)
    )
    med = col.select(["B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12", "NDVI", "EVI", "NDRE", "BSI"]).median()
    med = med.rename(["s2_B2_med", "s2_B3_med", "s2_B4_med", "s2_RE1_med", "s2_B8_med", "s2_B8A_med",
                      "s2_B11_med", "s2_B12_med", "s2_NDVI_med", "s2_EVI_med", "s2_NDRE_med", "s2_BSI_med"])
    ndvi = col.select("NDVI")
    p10 = ndvi.reduce(ee.Reducer.percentile([10])).rename("s2_NDVI_p10")
    p90 = ndvi.reduce(ee.Reducer.percentile([90])).rename("s2_NDVI_p90")
    amp = p90.subtract(p10).rename("s2_NDVI_amp")
    img = med.addBands([p10, p90, amp])
    for season in ("mam", "jja", "son"):
        smed = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filter(_season_filter(year, season))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(_scale_s2)
            .select(["NDVI", "NDRE", "BSI"])
            .median()
            .rename([f"s2_{season}_NDVI", f"s2_{season}_NDRE", f"s2_{season}_BSI"])
        )
        img = img.addBands(smed)
    return img


def stack_observational_features(year: int, roi: ee.Geometry, groups=None) -> ee.Image:
    groups = groups or ["l8", "s2", "s1", "aef", "terrain", "climate"]
    parts = []
    if "l8" in groups:
        parts.append(landsat8_features_lite(year, roi) if "lite" in groups else landsat8_features(year, roi))
    if "s2" in groups:
        parts.append(sentinel2_features_lite(year, roi) if "lite" in groups else sentinel2_features(year, roi))
    if "s1" in groups:
        parts.append(sentinel1_features(year, roi))
    if "aef" in groups:
        parts.append(alphaearth_features(year, roi))
    if "terrain" in groups:
        parts.append(terrain_features())
    if "climate" in groups:
        parts.append(climate_features(year, roi))
    if "water" in groups:
        parts.append(water_aux())
    img = parts[0]
    for p in parts[1:]:
        img = img.addBands(p)
    return img.clip(roi)

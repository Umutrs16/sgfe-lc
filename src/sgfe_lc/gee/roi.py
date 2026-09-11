from __future__ import annotations

import ee

from ..config import load_config

COUNTRY_ISO = {
    "KAZ": "Kazakhstan",
    "KGZ": "Kyrgyzstan",
    "TJK": "Tajikistan",
    "UZB": "Uzbekistan",
    "TKM": "Turkmenistan",
}


def country_fc(names=None) -> ee.FeatureCollection:
    cfg = load_config()
    names = names or list(COUNTRY_ISO.values())
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level0")
    return gaul.filter(ee.Filter.inList("ADM0_NAME", names))


def central_asia_roi() -> ee.Geometry:
    return country_fc().geometry().dissolve(maxError=100)


def country_code_image() -> ee.Image:
    """Integer country raster: 1=KAZ 2=KGZ 3=TJK 4=UZB 5=TKM."""
    mapping = [("Kazakhstan", 1), ("Kyrgyzstan", 2), ("Tajikistan", 3),
               ("Uzbekistan", 4), ("Turkmenistan", 5)]
    img = ee.Image(0).rename("country_id")
    for name, code in mapping:
        geom = country_fc([name]).geometry()
        img = img.where(ee.Image.constant(1).clip(geom), code)
    return img.updateMask(img.neq(0)).toByte()


def hex_grid(roi: ee.Geometry, scale_m: int = 100000) -> ee.FeatureCollection:
    return roi.coveringGrid("EPSG:3857", scale_m)


def site_hull_roi(lons, lats, buffer_m: float = 50000) -> ee.Geometry:
    coords = [[float(x), float(y)] for x, y in zip(lons, lats)]
    if len(coords) == 0:
        raise ValueError("site_hull_roi needs at least one point")
    if len(coords) == 1:
        return ee.Geometry.Point(coords[0]).buffer(buffer_m)
    if len(coords) == 2:
        return ee.Geometry.MultiPoint(coords).buffer(buffer_m)
    return ee.Geometry.MultiPoint(coords).convexHull(maxError=100).buffer(buffer_m)

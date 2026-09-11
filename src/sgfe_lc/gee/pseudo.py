from __future__ import annotations

import ee

from .products import product_agreement
from .roi import central_asia_roi


def consensus_candidates(year: int, roi: ee.Geometry | None = None, min_agree: int = 3, naboureh_asset=None) -> ee.Image:
    roi = roi or central_asia_roi()
    agr = product_agreement(year, roi, naboureh_asset=naboureh_asset)
    mask = agr.select("product_agree_n").gte(min_agree)
    return agr.select("product_mode").updateMask(mask).rename("pseudo_class")


def stratified_pseudo_points(class_img: ee.Image, roi: ee.Geometry, n_per_class: int = 400, scale: int = 30, seed: int = 20260902):
    samples = class_img.stratifiedSample(
        numPoints=n_per_class,
        classBand="pseudo_class",
        region=roi,
        scale=scale,
        seed=seed,
        geometries=True,
    )
    return samples.map(lambda f: f.set("tier", "C").set("pseudo_flag", 1).set("year", 2018))

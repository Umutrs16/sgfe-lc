from __future__ import annotations

import ee

from ..config import load_config
from .features import stack_observational_features
from .products import product_agreement
from .roi import central_asia_roi, country_fc


def remask_unofficial(img):
    """Drop Forest / Water / Snow-ice from the official map (no exact-year support)."""
    keep = img.eq(1).Or(img.eq(3)).Or(img.eq(4)).Or(img.eq(5)).Or(img.eq(7))
    return img.updateMask(keep)


def remask_unofficial_series(values):
    unofficial = {2, 6, 8}
    out = []
    for v in values:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            out.append(v)
            continue
        out.append(0 if iv in unofficial else iv)
    return out


def protocol_asset_tag(mode: str, threshold=None) -> str:
    """GEE asset IDs cannot contain '.'."""
    if mode == "hard":
        pct = 80 if threshold is None else int(round(float(threshold) * 100))
        return f"hard{pct:03d}"
    return str(mode or "eval")


def train_gee_rf(training: ee.FeatureCollection, predictors: list[str], class_property="class_id", n_trees=400):
    clf = ee.Classifier.smileRandomForest(n_trees).setOutputMode("CLASSIFICATION")
    return clf.train(training, class_property, predictors)


def train_gee_rf_prob(training: ee.FeatureCollection, predictors: list[str], class_property="class_id", n_trees=400):
    clf = ee.Classifier.smileRandomForest(n_trees).setOutputMode("MULTIPROBABILITY")
    return clf.train(training, class_property, predictors)


def classify_region(feature_img: ee.Image, classifier, predictors: list[str]) -> ee.Image:
    return feature_img.select(predictors).classify(classifier).rename("class_id").toByte()


def ensure_asset_folder(folder_id: str) -> str:
    try:
        ee.data.getAsset(folder_id)
    except Exception:
        ee.data.createAsset({"type": "Folder"}, folder_id)
    return folder_id


def country_degree_tiles(country: str, step=5.0, min_area_km2=50.0) -> list[tuple[int, int, int, int]]:
    """Lon/lat boxes that intersect a country. Coordinates are integer degrees."""
    geom = country_fc([country]).geometry()
    bounds = geom.bounds().getInfo()
    west, south, east, north = [float(v) for v in bounds]
    tiles = []
    lon0 = int(west // step * step)
    lat0 = int(south // step * step)
    lon = lon0
    while lon < east:
        lat = lat0
        while lat < north:
            box = ee.Geometry.Rectangle([lon, lat, lon + step, lat + step], None, False)
            inter = geom.intersection(box, 100)
            area = ee.Number(inter.area(100)).divide(1e6).getInfo()
            if area is not None and float(area) >= min_area_km2:
                tiles.append((int(lon), int(lat), int(lon + step), int(lat + step)))
            lat += step
        lon += step
    return tiles


def export_country_tiles(classifier, predictors, year=2018, country="Kazakhstan",
                         scale=30, groups=None, suffix="eval", step=5.0):
    cfg = load_config()
    groups = groups or ["aef", "terrain"]
    country_key = country.replace(" ", "")
    folder = f"{cfg['project']['asset_root']}/{country_key.lower()}_tiles_{suffix}"
    ensure_asset_folder(folder)
    geom = country_fc([country]).geometry()
    tiles = country_degree_tiles(country, step=step)
    tasks = {}
    for west, south, east, north in tiles:
        name = f"e{west:03d}n{south:03d}"
        box = ee.Geometry.Rectangle([west, south, east, north], None, False)
        roi = geom.intersection(box, 100)
        feats = stack_observational_features(year, roi, groups=groups)
        pred = classify_region(feats, classifier, predictors).clip(roi)
        task = ee.batch.Export.image.toAsset(
            image=pred,
            description=f"SGFE_LC_{year}_{country_key}_{scale}m_{suffix}_{name}",
            assetId=f"{folder}/{name}",
            region=roi,
            scale=scale,
            maxPixels=1e13,
            pyramidingPolicy={"class_id": "mode"},
        )
        task.start()
        tasks[name] = task
    return folder, tiles, tasks


def mosaic_tile_folder(folder_id: str):
    try:
        kids = ee.data.listAssets({"parent": folder_id}).get("assets", [])
    except Exception:
        return None
    imgs = [ee.Image(a["id"]) for a in kids if str(a.get("type", "")).upper() in {"IMAGE", "IMAGE_COLLECTION"}]
    if not imgs:
        return None
    return ee.ImageCollection(imgs).mosaic().rename("class_id")


def export_country_map(classifier, predictors, year=2018, country="Kazakhstan", scale=30, groups=None, suffix="eval"):
    cfg = load_config()
    roi = country_fc([country]).geometry()
    groups = groups or ["aef", "terrain"]
    feats = stack_observational_features(year, roi, groups=groups)
    pred = classify_region(feats, classifier, predictors)
    tag = f"_{suffix}" if suffix else ""
    task = ee.batch.Export.image.toAsset(
        image=pred.clip(roi),
        description=f"SGFE_LC_{year}_{country.replace(' ', '')}_{scale}m{tag}",
        assetId=f"{cfg['project']['asset_root']}/map_{year}_{country.replace(' ', '')}_{scale}m{tag}",
        region=roi,
        scale=scale,
        maxPixels=1e13,
        pyramidingPolicy={"class_id": "mode"},
    )
    task.start()
    return task


def export_uncertainty_layers(proba_img: ee.Image, train_fc: ee.FeatureCollection, predictors: list[str], roi, year=2018):
    """Export vote entropy companion layer. AOA is computed in Python at sites; regional AOA is optional."""
    cfg = load_config()
    # proba_img is array band from MULTIPROBABILITY
    # Approximate entropy via 1 - maxprob if array handling is awkward
    task = ee.batch.Export.image.toAsset(
        image=proba_img.clip(roi),
        description=f"SGFE_uncertainty_{year}",
        assetId=f"{cfg['project']['asset_root']}/uncertainty_{year}",
        region=roi,
        scale=30,
        maxPixels=1e13,
    )
    task.start()
    return task


def typical_zone_exports(classifier, predictors, year=2018, scale=10, groups=None, suffix="eval"):
    cfg = load_config()
    groups = groups or ["s2", "s1", "aef"]
    tasks = {}
    tag = f"_{suffix}" if suffix else ""
    for name, bbox in cfg["typical_zones"].items():
        roi = ee.Geometry.Rectangle(bbox)
        feats = stack_observational_features(year, roi, groups=groups)
        pred = classify_region(feats, classifier, predictors)
        task = ee.batch.Export.image.toAsset(
            image=pred.clip(roi),
            description=f"SGFE_zone_{name}_{scale}m{tag}",
            assetId=f"{cfg['project']['asset_root']}/zone_{name}_{year}_{scale}m{tag}",
            region=roi,
            scale=scale,
            maxPixels=1e13,
            pyramidingPolicy={"class_id": "mode"},
        )
        task.start()
        tasks[name] = task
    return tasks

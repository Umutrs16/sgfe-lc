from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

from ..config import load_config
from .features import (
    alphaearth_features,
    climate_features,
    landsat8_features_lite,
    sentinel1_features,
    sentinel2_features_lite,
    stack_observational_features,
    terrain_features,
    water_aux,
)
from .migration import stability_scores
from .products import dynamic_world_annual, esri_annual, glc_fcs30d_year, product_stack
from .roi import central_asia_roi, site_hull_roi


def sample_year_matched_training(train, properties=None, scale: int = 30, tile_scale: int = 8):
    """Option B: 2017 exact-year labels keep 2017 AEF; all other training rows use 2018 AEF."""
    from .features import alphaearth_features, terrain_features

    properties = properties or ["class_id"]
    year = pd.to_numeric(train.get("year"), errors="coerce")
    role = train["role"] if "role" in train.columns else pd.Series("exact_year", index=train.index)
    matched_2017 = train[year.eq(2017) & role.eq("exact_year")]
    other = train.drop(index=matched_2017.index)
    terrain = terrain_features()
    parts = []
    for sub, y in ((matched_2017, 2017), (other, 2018)):
        if sub.empty:
            continue
        roi = site_hull_roi(sub["longitude"], sub["latitude"], buffer_m=8000)
        feat = alphaearth_features(int(y), roi).addBands(terrain)
        fc = sites_to_fc(sub, properties=properties)
        parts.append(
            feat.sampleRegions(
                collection=fc, properties=properties, scale=scale, tileScale=tile_scale,
            )
        )
    if not parts:
        raise ValueError("No training rows for year-matched GEE sampling.")
    sampled = parts[0]
    for part in parts[1:]:
        sampled = sampled.merge(part)
    return sampled


def sites_to_fc(gdf: gpd.GeoDataFrame, properties=None) -> ee.FeatureCollection:
    properties = properties or [
        "site_id", "class_id", "country", "year", "split", "fold_id", "block_id", "tier"
    ]
    work = gdf.copy()
    if "geometry" not in work.columns:
        work = gpd.GeoDataFrame(
            work,
            geometry=gpd.points_from_xy(work["longitude"], work["latitude"]),
            crs="EPSG:4326",
        )
    elif work.crs is not None:
        try:
            if work.crs.to_epsg() != 4326:
                work = work.to_crs(4326)
        except Exception:
            pass
    keep = [c for c in properties if c in work.columns]
    slim = work[keep + ["geometry"]].copy()
    for p in ("class_id", "year", "fold_id"):
        if p in slim.columns:
            slim[p] = pd.to_numeric(slim[p], errors="coerce")
    records = json.loads(slim.to_json())
    for feat in records.get("features", []):
        props = feat.get("properties") or {}
        clean = {}
        for k, v in props.items():
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            clean[k] = v
        feat["properties"] = clean
    return ee.FeatureCollection(records)


def sample_image(img: ee.Image, sites: ee.FeatureCollection, scale: int = 30, tile_scale: int = 4):
    return img.sampleRegions(
        collection=sites,
        scale=scale,
        geometries=True,
        tileScale=tile_scale,
    )


def export_table(fc: ee.FeatureCollection, description: str, folder: str | None = None):
    cfg = load_config()
    folder = folder or cfg["project"]["drive_folder"]
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=folder,
        fileFormat="CSV",
    )
    task.start()
    return task


def export_site_features(gdf: gpd.GeoDataFrame, year: int = 2018, groups=None, scale: int = 30):
    roi = central_asia_roi()
    sites = sites_to_fc(gdf)
    tasks = {}
    bundles = groups or {
        "L8": ["l8"],
        "S2": ["s2"],
        "S1": ["s1"],
        "AEF": ["aef"],
        "ENV": ["terrain", "climate", "water"],
    }
    for name, grp in bundles.items():
        img = stack_observational_features(year, roi, groups=grp)
        fc = sample_image(img, sites, scale=scale)
        tasks[name] = export_table(fc, f"SGFE_sites_{year}_{name}")
    return tasks


def export_product_labels(gdf: gpd.GeoDataFrame, years=(2013, 2014, 2017, 2018), scale: int = 30):
    roi = central_asia_roi()
    sites = sites_to_fc(gdf)
    tasks = {}
    for y in years:
        img = product_stack(y, roi)
        fc = sample_image(img, sites, scale=scale)
        tasks[y] = export_table(fc, f"SGFE_products_{y}")
    return tasks


def download_fc_csv(fc: ee.FeatureCollection, dest: Path, selectors=None) -> pd.DataFrame:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    try:
        url = fc.getDownloadURL(filetype="CSV", selectors=selectors, filename=dest.stem)
        urllib.request.urlretrieve(url, dest)
        return pd.read_csv(dest)
    except Exception as e:
        last_err = e
    try:
        info = fc.getInfo()
        rows = []
        for feat in info.get("features", []):
            props = dict(feat.get("properties") or {})
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            if "longitude" not in props and coords and coords[0] is not None:
                props["longitude"] = coords[0]
                props["latitude"] = coords[1]
            rows.append(props)
        df = pd.DataFrame(rows)
        df.to_csv(dest, index=False)
        return df
    except Exception as e:
        raise RuntimeError(f"download failed: {last_err}; getInfo failed: {e}") from e


def reduce_sites(img: ee.Image, sites: ee.FeatureCollection, scale: int = 30, tile_scale: int = 4):
    # Single-band first() names the property "first"; pad so band names are kept.
    padded = img.addBands(ee.Image.constant(0).rename("zzpad"))
    return padded.reduceRegions(
        collection=sites,
        reducer=ee.Reducer.first(),
        scale=scale,
        tileScale=tile_scale,
    )


def _chunks(gdf: gpd.GeoDataFrame, n: int):
    for i in range(0, len(gdf), n):
        yield gdf.iloc[i : i + n]


def _chunk_roi(sub: gpd.GeoDataFrame, buffer_m: float = 8000):
    if "longitude" in sub.columns and "latitude" in sub.columns:
        return site_hull_roi(sub["longitude"], sub["latitude"], buffer_m=buffer_m)
    return site_hull_roi(sub.geometry.x, sub.geometry.y, buffer_m=buffer_m)


def extract_bundle_local(
    gdf,
    img_or_fn,
    dest: Path,
    chunk: int = 250,
    scale: int = 30,
    retries: int = 4,
    prefix_map: dict | None = None,
) -> pd.DataFrame:
    if dest.exists() and dest.stat().st_size > 100:
        return pd.read_csv(dest)
    parts = []
    for i, sub in enumerate(_chunks(gdf, chunk)):
        part_path = dest.with_name(f"{dest.stem}_part{i}{dest.suffix}")
        if part_path.exists() and part_path.stat().st_size > 50:
            parts.append(pd.read_csv(part_path))
            print(f"  cache {part_path.name}")
            continue
        img = img_or_fn(_chunk_roi(sub)) if callable(img_or_fn) else img_or_fn
        sites = sites_to_fc(sub)
        fc = reduce_sites(img, sites, scale=scale)
        last_err = None
        for attempt in range(retries):
            try:
                df = download_fc_csv(fc, part_path)
                parts.append(df)
                print(f"  {part_path.name} n={len(df)}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  retry {attempt + 1} {part_path.name}: {e}")
                time.sleep(10 * (attempt + 1))
        if last_err:
            raise last_err
    out = pd.concat(parts, ignore_index=True)
    if prefix_map:
        out = out.rename(columns={k: v for k, v in prefix_map.items() if k in out.columns})
    out.to_csv(dest, index=False)
    return out


def extract_all_local(gdf: gpd.GeoDataFrame, out_dir: Path, year: int = 2018) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = gdf.copy()
    if "longitude" in work.columns:
        work = work.sort_values(["longitude", "latitude"]).reset_index(drop=True)
    jobs = {
        "AEF": (lambda roi, y=year: alphaearth_features(y, roi), 30, 300, None),
        "ENV": (
            lambda roi, y=year: terrain_features().addBands(climate_features(y, roi)).addBands(water_aux()),
            30, 300, None,
        ),
        "GLC2018": (lambda roi: glc_fcs30d_year(2018, roi), 30, 300, None),
        "GLC2017": (lambda roi: glc_fcs30d_year(2017, roi), 30, 300, None),
        "GLC2014": (lambda roi: glc_fcs30d_year(2014, roi), 30, 300, None),
        "GLC2013": (lambda roi: glc_fcs30d_year(2013, roi), 30, 300, None),
        "ESRI2018": (lambda roi: esri_annual(2018, roi), 10, 250, None),
        "DW2018": (
            lambda roi: dynamic_world_annual(2018, roi).select(["dw_l1_2018", "dw_maxprob", "dw_entropy"]),
            10, 80, None,
        ),
        "L8": (lambda roi, y=year: landsat8_features_lite(y, roi), 30, 120, None),
        "S1": (lambda roi, y=year: sentinel1_features(y, roi), 10, 100, None),
        "S2": (lambda roi, y=year: sentinel2_features_lite(y, roi), 20, 50, None),
        "MIG2013": (
            lambda roi, y=year: stability_scores(2013, y, roi),
            30, 150,
            {c: f"{c}_2013" for c in ("C_mig", "S_product", "S_change", "S_pheno", "ndvi_abs_delta")},
        ),
        "MIG2014": (
            lambda roi, y=year: stability_scores(2014, y, roi),
            30, 150,
            {c: f"{c}_2014" for c in ("C_mig", "S_product", "S_change", "S_pheno", "ndvi_abs_delta")},
        ),
    }
    paths = {}
    failed = {}
    for name, (factory, scale, chunk, prefix_map) in jobs.items():
        dest = out_dir / f"SGFE_{name}.csv"
        print(f"=== {name} ===")
        try:
            extract_bundle_local(work, factory, dest, chunk=chunk, scale=scale, prefix_map=prefix_map)
            paths[name] = dest
        except Exception as e:
            failed[name] = str(e)
            print(f"FAILED {name}: {e}")
    if failed:
        (out_dir / "extract_failures.json").write_text(
            json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return paths


def extract_year_local(gdf: gpd.GeoDataFrame, out_dir: Path, year: int) -> dict[str, Path]:
    """Year-matched observational + product stacks (used for 2017 exact-year pairing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = gdf.copy()
    if "longitude" in work.columns:
        work = work.sort_values(["longitude", "latitude"]).reset_index(drop=True)
    jobs = {
        f"AEF{year}": (lambda roi, y=year: alphaearth_features(y, roi), 30, 300, None),
        f"ENV{year}": (
            lambda roi, y=year: terrain_features().addBands(climate_features(y, roi)).addBands(water_aux()),
            30, 300, None,
        ),
        f"ESRI{year}": (lambda roi, y=year: esri_annual(y, roi), 10, 250, {"first": f"esri_l1_{year}"}),
        f"DW{year}": (
            lambda roi, y=year: dynamic_world_annual(y, roi).select([f"dw_l1_{y}", "dw_maxprob", "dw_entropy"]),
            10, 80,
            {"dw_maxprob": f"dw_maxprob_{year}", "dw_entropy": f"dw_entropy_{year}"},
        ),
        f"L8_{year}": (lambda roi, y=year: landsat8_features_lite(y, roi), 30, 120, None),
        f"S1_{year}": (lambda roi, y=year: sentinel1_features(y, roi), 10, 100, None),
        f"S2_{year}": (lambda roi, y=year: sentinel2_features_lite(y, roi), 20, 50, None),
    }
    paths = {}
    failed = {}
    for name, (factory, scale, chunk, prefix_map) in jobs.items():
        dest = out_dir / f"SGFE_{name}.csv"
        print(f"=== {name} ===")
        try:
            extract_bundle_local(work, factory, dest, chunk=chunk, scale=scale, prefix_map=prefix_map)
            paths[name] = dest
        except Exception as e:
            failed[name] = str(e)
            print(f"FAILED {name}: {e}")
    if failed:
        (out_dir / f"extract_failures_{year}.json").write_text(
            json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return paths


def export_migration_scores(gdf: gpd.GeoDataFrame, survey_years=(2013, 2014), map_year: int = 2018):
    roi = central_asia_roi()
    sites = sites_to_fc(gdf)
    tasks = {}
    for y in survey_years:
        img = stability_scores(y, map_year, roi)
        fc = sample_image(img, sites, scale=30)
        tasks[y] = export_table(fc, f"SGFE_migration_{y}_to_{map_year}")
    return tasks

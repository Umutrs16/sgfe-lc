"""Harvest high-agreement 2018 consensus pixels as Tier C training extras.

Point-wise product extract (GLC → Esri → DW) avoids building a wall-to-wall
agreement image, which exceeds the GEE user-memory limit.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.gee.auth import init_ee
from sgfe_lc.gee.export import extract_bundle_local
from sgfe_lc.gee.features import alphaearth_features, terrain_features
from sgfe_lc.gee.products import dynamic_world_annual, esri_annual, glc_fcs30d_year


def _km_to_test(lon, lat, test_lon, test_lat) -> np.ndarray:
    tr = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    tx, ty = tr.transform(np.asarray(test_lon, float), np.asarray(test_lat, float))
    d = np.sqrt((x[:, None] - tx[None, :]) ** 2 + (y[:, None] - ty[None, :]) ** 2)
    return d.min(axis=1) / 1000.0


def _windows(train: pd.DataFrame, cfg: dict) -> list[list[float]]:
    boxes = [list(bbox) for bbox in cfg["typical_zones"].values()]
    rng = np.random.default_rng(cfg["project"]["seed"])
    for cid in (1, 3, 4, 5, 7):
        sub = train[pd.to_numeric(train.get("class_id"), errors="coerce") == cid]
        if sub.empty:
            continue
        take = sub.sample(n=min(2, len(sub)), random_state=int(rng.integers(1, 1_000_000)))
        for r in take.itertuples(index=False):
            lon, lat = float(r.longitude), float(r.latitude)
            boxes.append([lon - 0.40, lat - 0.30, lon + 0.40, lat + 0.30])
    return boxes


def _random_points(boxes: list[list[float]], n_per_box: int, seed: int) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, b in enumerate(boxes):
        lon = rng.uniform(b[0], b[2], n_per_box)
        lat = rng.uniform(b[1], b[3], n_per_box)
        for x, y in zip(lon, lat):
            rows.append({"site_id": f"PSEUDO_{i}_{len(rows)}", "longitude": float(x), "latitude": float(y)})
    gdf = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["longitude"] for r in rows], [r["latitude"] for r in rows]), crs="EPSG:4326")
    return gdf


def _first_numeric(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def main():
    cfg = load_config()
    init_ee()
    processed = resolve_path(cfg, "processed")
    dest_dir = processed / "gee_exports"
    dest_dir.mkdir(parents=True, exist_ok=True)
    sites = gpd.read_file(processed / "sites_exact_year.gpkg")
    test = sites[(sites.get("split") == "test") & (sites.get("role") == "exact_year")]
    train = sites[(sites.get("split") != "test") & (sites.get("role") == "exact_year")]
    if train.empty or test.empty:
        print("Need sites_exact_year.gpkg with locked splits first.")
        return 2

    cand = _random_points(_windows(train, cfg), n_per_box=80, seed=cfg["project"]["seed"])
    print("candidate points", len(cand))

    print("=== GLC 2018 ===")
    glc = extract_bundle_local(
        cand,
        lambda roi: glc_fcs30d_year(2018, roi),
        dest_dir / "SGFE_PSEUDO_GLC2018.csv",
        chunk=200,
        scale=30,
        prefix_map={"first": "glc_l1_2018", "glc_l1_2018": "glc_l1_2018"},
    )
    print("=== Esri 2018 ===")
    esri = extract_bundle_local(
        cand,
        lambda roi: esri_annual(2018, roi),
        dest_dir / "SGFE_PSEUDO_ESRI2018.csv",
        chunk=200,
        scale=10,
        prefix_map={"first": "esri_l1_2018", "esri_l1_2018": "esri_l1_2018"},
    )
    work = cand.merge(glc, on="site_id", how="left", suffixes=("", "_glc"))
    work = work.merge(esri, on="site_id", how="left", suffixes=("", "_esri"))
    work["glc_l1"] = _first_numeric(work, ["glc_l1_2018", "first", "glc_l1"])
    work["esri_l1"] = _first_numeric(work, ["esri_l1_2018", "esri_l1", "first_esri"])
    pair = work[work["glc_l1"].eq(work["esri_l1"]) & work["glc_l1"].isin([1, 3, 4, 5, 7])].copy()
    print("GLC==Esri core-class candidates", len(pair))
    if pair.empty:
        print("No two-product agreement.")
        return 1

    print("=== DW 2018 on two-product survivors ===")
    dw = extract_bundle_local(
        pair,
        lambda roi: dynamic_world_annual(2018, roi).select(["dw_l1_2018", "dw_maxprob"]),
        dest_dir / "SGFE_PSEUDO_DW2018.csv",
        chunk=60,
        scale=10,
    )
    pair = pair.merge(dw, on="site_id", how="left", suffixes=("", "_dw"))
    pair["dw_l1"] = _first_numeric(pair, ["dw_l1_2018", "dw_l1"])
    dw_p = _first_numeric(pair, ["dw_maxprob"])
    keep = pair[pair["dw_l1"].eq(pair["glc_l1"]) & (dw_p.isna() | (dw_p >= 0.50))].copy()
    keep["class_id"] = keep["glc_l1"].astype(int)
    dist = _km_to_test(keep["longitude"], keep["latitude"], test["longitude"], test["latitude"])
    keep = keep.loc[dist >= 5].copy()
    kept = [sub.head(80) for _, sub in keep.groupby("class_id")]
    keep = pd.concat(kept, ignore_index=True) if kept else keep
    keep["site_id"] = [f"PSEUDO_{i}" for i in range(len(keep))]
    print("three-product consensus after 5 km buffer", len(keep), keep["class_id"].value_counts().to_dict())
    if keep.empty:
        print("No pseudo points survived.")
        return 1

    gdf = gpd.GeoDataFrame(
        keep,
        geometry=gpd.points_from_xy(keep["longitude"], keep["latitude"]),
        crs="EPSG:4326",
    )
    print("=== AEF + terrain ===")
    feats = extract_bundle_local(
        gdf,
        lambda roi: alphaearth_features(2018, roi).addBands(terrain_features()),
        dest_dir / "SGFE_PSEUDO_AEF.csv",
        chunk=80,
        scale=30,
    )
    if "site_id" in feats.columns:
        gdf = gdf.drop(columns=[c for c in gdf.columns if c in feats.columns and c != "site_id"], errors="ignore")
        gdf = gdf.merge(feats, on="site_id", how="left")
    gdf["tier"] = "C"
    gdf["pseudo_flag"] = True
    gdf["role"] = "pseudo"
    gdf["year"] = 2018
    gdf["split"] = "traindev"
    gdf["w_rel"] = 0.5
    out = processed / "pseudo_tier_c.csv"
    gdf.drop(columns="geometry", errors="ignore").to_csv(out, index=False)
    print("Wrote", out, "n", len(gdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

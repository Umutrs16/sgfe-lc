from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from .legend import RANGELAND_ID


def write_gpkg(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    """Write GPKG without case-colliding column names (Country vs country)."""
    path = Path(path)
    keep, seen = [], set()
    for c in gdf.columns:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        keep.append(c)
    gdf[keep].to_file(path, driver="GPKG")
    csv_path = path.with_suffix(".csv")
    gdf[keep].drop(columns="geometry", errors="ignore").to_csv(csv_path, index=False)
    return path


KEY = ["site_id"]


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(path)


def merge_feature_tables(site_gdf: gpd.GeoDataFrame, feature_files: list[Path]) -> gpd.GeoDataFrame:
    out = site_gdf.copy()
    for path in feature_files:
        tab = _read_any(path)
        tab = tab.drop(columns=[c for c in tab.columns if c in {".geo", "system:index", "geo"}], errors="ignore")
        overlap = [c for c in KEY if c in tab.columns and c in out.columns]
        if not overlap:
            # fallback spatial join keys
            if {"longitude", "latitude"}.issubset(tab.columns) and {"longitude", "latitude"}.issubset(out.columns):
                tab["_xy"] = tab["longitude"].round(5).astype(str) + "_" + tab["latitude"].round(5).astype(str)
                out["_xy"] = out["longitude"].round(5).astype(str) + "_" + out["latitude"].round(5).astype(str)
                overlap = ["_xy"]
            else:
                continue
        drop_cols = [c for c in tab.columns if c in out.columns and c not in overlap]
        tab = tab.drop(columns=drop_cols)
        out = out.merge(tab, on=overlap, how="left")
        if "_xy" in out.columns:
            out = out.drop(columns="_xy")
    return out


def _year_matched_score(df: pd.DataFrame, base: str) -> pd.Series:
    year = pd.to_numeric(df.get("year"), errors="coerce")
    if base in df.columns:
        c = pd.to_numeric(df[base], errors="coerce")
    else:
        c = pd.Series(np.nan, index=df.index, dtype="float64")
    for y in (2013, 2014):
        col = f"{base}_{y}"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            c = vals.where(year.eq(y), c)
    return c


def _ontology_code(cid):
    """Grass/shrub share a rangeland code for field–product agreement."""
    if cid is None or (isinstance(cid, float) and np.isnan(cid)):
        return np.nan
    try:
        iv = int(cid)
    except (TypeError, ValueError):
        return np.nan
    if iv in (3, 4):
        return int(RANGELAND_ID)
    return iv


def field_anchored_s_product(df: pd.DataFrame, map_year: int = 2018) -> pd.Series:
    """Fraction of available yearly GLC L1 maps that agree with the field class.

    S_product,i = (1/N) Σ_y I(agree(g_iy, c_i)) over exported GLC years in
    [survey year, map year]. Grassland and shrubland are compared under the
    common rangeland ontology. A product that is stably wrong therefore scores 0.
    """
    year = pd.to_numeric(df.get("year"), errors="coerce")
    field = pd.to_numeric(df.get("class_id"), errors="coerce").map(_ontology_code)
    hits = pd.Series(0.0, index=df.index)
    counts = pd.Series(0.0, index=df.index)
    for y in range(2013, int(map_year) + 1):
        col = f"glc_l1_{y}"
        if col not in df.columns:
            continue
        in_window = year.notna() & (float(y) >= year) & (y <= map_year)
        g = pd.to_numeric(df[col], errors="coerce")
        g_ont = g.map(_ontology_code)
        valid = in_window & field.notna() & g.notna() & (g > 0) & g_ont.notna()
        agree = valid & (g_ont == field)
        hits = hits.add(agree.astype(float), fill_value=0)
        counts = counts.add(valid.astype(float), fill_value=0)
    return hits / counts.replace(0, np.nan)


def assign_roles_and_weights(df: pd.DataFrame, high=0.80) -> pd.DataFrame:
    """Exact-year 2017/2018 vs historical 2013/2014; class-aware reliability weights."""
    out = df.copy()
    year = pd.to_numeric(out.get("year"), errors="coerce")
    role = pd.Series("drop", index=out.index)
    role.loc[year.isin([2017, 2018])] = "exact_year"
    role.loc[year.isin([2013, 2014])] = "historical"
    out["role"] = role
    for key in ("S_product", "S_change", "S_pheno"):
        if key not in out.columns or pd.to_numeric(out[key], errors="coerce").isna().all():
            out[key] = _year_matched_score(out, key)
    if "S_product_self" not in out.columns:
        out["S_product_self"] = pd.to_numeric(out.get("S_product"), errors="coerce")
    out["S_product"] = field_anchored_s_product(out)
    cls = pd.to_numeric(out.get("class_id"), errors="coerce")
    s_prod = pd.to_numeric(out.get("S_product"), errors="coerce").fillna(0)
    s_chg = pd.to_numeric(out.get("S_change"), errors="coerce").fillna(0)
    s_ph = pd.to_numeric(out.get("S_pheno"), errors="coerce").fillna(0)
    # Vegetation classes: product–field agreement × change × phenology.
    w = s_prod * s_chg * s_ph
    # Cropland: NDVI median change can be rotation/irrigation, not a cover break.
    w = w.where(~cls.eq(1), s_prod * (0.5 + 0.5 * s_ph))
    # Built / Water: NDVI phenology is not a land-cover stability criterion.
    w = w.where(~cls.isin([6, 7]), s_prod)
    out["w_rel"] = w.clip(0, 1)
    out["C_mig"] = (s_prod * s_chg * s_ph).clip(0, 1)
    out.loc[out["role"] == "exact_year", "w_rel"] = 1.0
    out = assign_tiers(out, high=high, map_year=2018)
    out.loc[out["role"] == "exact_year", "w_rel"] = 1.0
    return out


def year_matched_product(df: pd.DataFrame, stem: str) -> pd.Series:
    year = pd.to_numeric(df.get("year"), errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for y in (2013, 2014, 2015, 2016, 2017, 2018):
        col = f"{stem}_{y}"
        if col in df.columns:
            out = pd.to_numeric(df[col], errors="coerce").where(year.eq(y), out)
    return out


def assemble_year_matched(sites: gpd.GeoDataFrame, export_dir: Path) -> gpd.GeoDataFrame:
    """Attach year-matched features: 2017 sites get 2017 stacks; 2018 and historical get 2018 stacks."""
    export_dir = Path(export_dir)
    year = pd.to_numeric(sites["year"], errors="coerce")
    files_2018 = [
        export_dir / "SGFE_AEF.csv",
        export_dir / "SGFE_ENV.csv",
        export_dir / "SGFE_L8.csv",
        export_dir / "SGFE_S1.csv",
        export_dir / "SGFE_S2.csv",
        export_dir / "SGFE_DW2018.csv",
        export_dir / "SGFE_ESRI2018.csv",
        export_dir / "SGFE_GLC2018.csv",
        export_dir / "SGFE_GLC2017.csv",
        export_dir / "SGFE_GLC2016.csv",
        export_dir / "SGFE_GLC2015.csv",
        export_dir / "SGFE_GLC2014.csv",
        export_dir / "SGFE_GLC2013.csv",
        export_dir / "SGFE_MIG2013.csv",
        export_dir / "SGFE_MIG2014.csv",
    ]
    files_2017 = [
        export_dir / "SGFE_AEF2017.csv",
        export_dir / "SGFE_ENV2017.csv",
        export_dir / "SGFE_L8_2017.csv",
        export_dir / "SGFE_S1_2017.csv",
        export_dir / "SGFE_S2_2017.csv",
        export_dir / "SGFE_DW2017.csv",
        export_dir / "SGFE_ESRI2017.csv",
        export_dir / "SGFE_GLC2017.csv",
        export_dir / "SGFE_GLC2018.csv",
    ]
    y2017 = sites.loc[year.eq(2017)].copy()
    y2018 = sites.loc[year.eq(2018)].copy()
    hist = sites.loc[year.isin([2013, 2014])].copy()
    other = sites.loc[~year.isin([2013, 2014, 2017, 2018])].copy()
    parts = []
    if len(y2017):
        parts.append(merge_feature_tables(y2017, [p for p in files_2017 if p.exists()]))
    if len(y2018):
        parts.append(merge_feature_tables(y2018, [p for p in files_2018 if p.exists()]))
    if len(hist):
        parts.append(merge_feature_tables(hist, [p for p in files_2018 if p.exists()]))
    if len(other):
        parts.append(other)
    out = pd.concat(parts, ignore_index=True)
    if "dw_l1_2017" in out.columns or "dw_l1_2018" in out.columns:
        out["dw_l1"] = year_matched_product(out, "dw_l1")
    if "esri_l1_2017" in out.columns or "esri_l1_2018" in out.columns:
        out["esri_l1"] = year_matched_product(out, "esri_l1")
    if any(f"glc_l1_{y}" in out.columns for y in (2013, 2014, 2015, 2016, 2017, 2018)):
        out["glc_l1"] = year_matched_product(out, "glc_l1")
    return gpd.GeoDataFrame(out, geometry=out["geometry"] if "geometry" in out.columns else None, crs=getattr(sites, "crs", "EPSG:4326"))


def assign_tiers(df: pd.DataFrame, high=0.80, map_year=2018) -> pd.DataFrame:
    out = df.copy()
    year = pd.to_numeric(out.get("year"), errors="coerce")
    if "C_mig" not in out.columns:
        out["C_mig"] = _year_matched_score(out, "C_mig")
    c_mig = pd.to_numeric(out.get("C_mig"), errors="coerce")
    tier = pd.Series("drop", index=out.index)
    tier.loc[year.isin([map_year, map_year - 1])] = "A"  # 2017–2018 same campaign
    hist = year.isin([2013, 2014])
    tier.loc[hist & (c_mig >= high)] = "B"
    tier.loc[hist & c_mig.notna() & (c_mig < high)] = "B_ungated"
    tier.loc[hist & c_mig.isna()] = "B_ungated"
    if "pseudo_flag" in out.columns:
        tier.loc[out["pseudo_flag"].astype(bool)] = "C"
    out["tier"] = tier
    return out

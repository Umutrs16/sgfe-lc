from __future__ import annotations

import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .config import load_config, resolve_path
from .legend import LEVEL1, harmonize_label

LON_ALIASES = {
    "lon", "long", "longitude", "lng", "x", "x_coord", "经度", "lon_dd", "long_dd",
}
LAT_ALIASES = {
    "lat", "latitude", "y", "y_coord", "纬度", "lat_dd",
}
YEAR_ALIASES = {
    "year", "survey_year", "yr", "date_year", "年份", "surveyyear",
}
DATE_ALIASES = {
    "date", "survey_date", "surveydate", "obs_date", "time", "调查日期",
}
CLASS_ALIASES = {
    "class", "lc", "landcover", "land_cover", "lulc", "type", "lc_type",
    "original_lc", "harmonized_lc", "classname", "class_name", "地类",
    "土地覆被", "覆被类型", "land_cover_type", "calc_l1", "calc_l1_id", "faolccs_3l",
}
COUNTRY_ALIASES = {
    "country", "nation", "adm0", "iso3", "国家", "country_name",
}
ID_ALIASES = {
    "site_id", "id", "sid", "plot_id", "sample_id", "fid", "样点", "编号", "point",
}

COUNTRY_MAP = {
    "kazakhstan": "KAZ",
    "kaz": "KAZ",
    "kz": "KAZ",
    "哈萨克斯坦": "KAZ",
    "kyrgyzstan": "KGZ",
    "kyrgyz": "KGZ",
    "kgz": "KGZ",
    "kg": "KGZ",
    "吉尔吉斯斯坦": "KGZ",
    "tajikistan": "TJK",
    "tjk": "TJK",
    "tj": "TJK",
    "塔吉克斯坦": "TJK",
    "uzbekistan": "UZB",
    "uzb": "UZB",
    "uz": "UZB",
    "乌兹别克斯坦": "UZB",
    "turkmenistan": "TKM",
    "tkm": "TKM",
    "tm": "TKM",
    "土库曼斯坦": "TKM",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(name).strip().lower())


def _find_col(columns, aliases) -> str | None:
    norm_alias = {_norm(a) for a in aliases}
    for c in columns:
        if _norm(c) in norm_alias:
            return c
    return None


def _read_table(path: Path) -> pd.DataFrame | gpd.GeoDataFrame:
    suffix = path.suffix.lower()
    if suffix in {".shp", ".gpkg", ".geojson", ".json"}:
        return gpd.read_file(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def discover_site_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    shps = [p for p in raw_dir.rglob("*.shp") if "CA_LC_Survey" in p.name]
    if shps:
        return shps
    files: list[Path] = []
    for pattern in ("*.csv", "*.xlsx", "*.xls", "*.shp", "*.gpkg", "*.geojson"):
        files.extend(raw_dir.rglob(pattern))
    skip = {"readme", "dictionary", "metadata", "compilation_sites&photos_ca_2013-2018"}
    return [p for p in files if p.stem.lower() not in skip]


def extract_archives(raw_dir: Path) -> None:
    for archive in list(raw_dir.glob("*.rar")) + list(raw_dir.glob("*.zip")):
        dest = raw_dir / archive.stem
        dest.mkdir(exist_ok=True)
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)


def _year_from_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    years = parsed.dt.year
    numeric = pd.to_numeric(series, errors="coerce")
    years = years.fillna(numeric)
    extracted = series.astype(str).str.extract(r"(20\d{2})", expand=False)
    years = years.fillna(pd.to_numeric(extracted, errors="coerce"))
    return years


ISO2_TO_ISO3 = {"KZ": "KAZ", "KG": "KGZ", "TJ": "TJK", "UZ": "UZB", "TM": "TKM"}


def _country_code(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip()
    if raw.upper() in {"KAZ", "KGZ", "TJK", "UZB", "TKM"}:
        return raw.upper()
    if raw.upper() in ISO2_TO_ISO3:
        return ISO2_TO_ISO3[raw.upper()]
    return COUNTRY_MAP.get(raw.lower())


def harmonize_sites(df: pd.DataFrame | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = list(df.columns)
    lon_c = _find_col(cols, LON_ALIASES)
    lat_c = _find_col(cols, LAT_ALIASES)
    year_c = _find_col(cols, YEAR_ALIASES)
    date_c = _find_col(cols, DATE_ALIASES)
    class_c = _find_col(cols, CLASS_ALIASES)
    country_c = _find_col(cols, COUNTRY_ALIASES)
    id_c = _find_col(cols, ID_ALIASES)

    if isinstance(df, gpd.GeoDataFrame) and df.geometry is not None and df.geometry.geom_type.isin(["Point"]).any():
        gdf = df.to_crs(4326) if df.crs and df.crs.to_epsg() != 4326 else df.copy()
        out = gdf.copy()
        out["longitude"] = out.geometry.x
        out["latitude"] = out.geometry.y
    else:
        if lon_c is None or lat_c is None:
            raise ValueError(
                f"Cannot find longitude/latitude columns. Columns={cols}"
            )
        out = df.copy()
        out["longitude"] = pd.to_numeric(out[lon_c], errors="coerce")
        out["latitude"] = pd.to_numeric(out[lat_c], errors="coerce")
        out = gpd.GeoDataFrame(
            out,
            geometry=[Point(xy) for xy in zip(out["longitude"], out["latitude"])],
            crs="EPSG:4326",
        )

    if year_c:
        out["year"] = pd.to_numeric(out[year_c], errors="coerce")
    elif date_c:
        out["year"] = _year_from_date(out[date_c])
    else:
        out["year"] = pd.NA

    l1_c = _find_col(cols, {"calc_l1"})
    l2_c = _find_col(cols, {"calc_l2"})
    if l1_c and l2_c:
        out["original_lc"] = out[l1_c].astype(str) + "|" + out[l2_c].astype(str)
        # Woodland/Shrubbery is shrub, not forest.
        shrub = out[l2_c].astype(str).str.lower().str.contains("shrub")
        out["class_id"] = out[l2_c].map(harmonize_label)
        still_na = out["class_id"].isna()
        out.loc[still_na, "class_id"] = out.loc[still_na, l1_c].map(harmonize_label)
        out.loc[shrub, "class_id"] = 4
    elif class_c:
        out["original_lc"] = out[class_c].astype(str)
        out["class_id"] = out[class_c].map(harmonize_label)
    else:
        out["original_lc"] = pd.NA
        out["class_id"] = pd.NA

    out["class_name"] = out["class_id"].map(LEVEL1)
    out["country"] = out[country_c].map(_country_code) if country_c else pd.NA
    out["site_id"] = out[id_c].astype(str) if id_c else [f"CA{i:04d}" for i in range(1, len(out) + 1)]
    out["survey_campaign"] = out["year"].map(
        lambda y: "2013-2014" if pd.notna(y) and int(y) <= 2014 else (
            "2017-2018" if pd.notna(y) and int(y) >= 2017 else "unknown"
        )
    )
    return out


def quality_control(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    flags = []
    work = gdf.copy()
    work["qc_keep"] = True
    work["qc_flag"] = ""

    coord_bad = (
        work["longitude"].isna()
        | work["latitude"].isna()
        | (work["longitude"] < 40)
        | (work["longitude"] > 95)
        | (work["latitude"] < 30)
        | (work["latitude"] > 60)
    )
    work.loc[coord_bad, "qc_keep"] = False
    work.loc[coord_bad, "qc_flag"] = work.loc[coord_bad, "qc_flag"] + "coord_outlier;"
    flags.append(("coord_outlier", int(coord_bad.sum())))

    dup = work.duplicated(["longitude", "latitude", "year"], keep="first")
    work.loc[dup, "qc_keep"] = False
    work.loc[dup, "qc_flag"] = work.loc[dup, "qc_flag"] + "duplicate_xy_year;"
    flags.append(("duplicate_xy_year", int(dup.sum())))

    no_class = work["class_id"].isna()
    work.loc[no_class, "qc_keep"] = False
    work.loc[no_class, "qc_flag"] = work.loc[no_class, "qc_flag"] + "unmapped_class;"
    flags.append(("unmapped_class", int(no_class.sum())))

    report = pd.DataFrame(flags, columns=["issue", "n"])
    report.loc[len(report)] = ["input_n", len(gdf)]
    report.loc[len(report)] = ["retained_n", int(work["qc_keep"].sum())]
    return work, report


def audit_crosstabs(gdf: gpd.GeoDataFrame) -> dict[str, pd.DataFrame]:
    keep = gdf[gdf["qc_keep"]].copy()
    return {
        "country_year": pd.crosstab(keep["country"].fillna("NA"), keep["year"].fillna(-1)),
        "country_class": pd.crosstab(keep["country"].fillna("NA"), keep["class_name"].fillna("NA")),
        "year_class": pd.crosstab(keep["year"].fillna(-1), keep["class_name"].fillna("NA")),
        "campaign_class": pd.crosstab(keep["survey_campaign"], keep["class_name"].fillna("NA")),
    }


def load_or_ingest(cfg=None) -> gpd.GeoDataFrame:
    cfg = cfg or load_config()
    processed = resolve_path(cfg, "processed") / "sites_harmonized.gpkg"
    if processed.exists():
        return gpd.read_file(processed)
    raw_dir = resolve_path(cfg, "raw_sites")
    extract_archives(raw_dir)
    files = discover_site_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No site table found in {raw_dir}. Place CA_LC_Sites&PhotosInfo.rar or a CSV/SHP/XLSX there."
        )
    frames = [harmonize_sites(_read_table(p)) for p in files]
    gdf = pd.concat(frames, ignore_index=True)
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf, _ = quality_control(gdf)
    return gdf

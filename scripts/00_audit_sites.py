"""Audit 902-site metadata, harmonize legend, and write processed tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.sites import audit_crosstabs, discover_site_files, extract_archives, harmonize_sites, quality_control, _read_table
from sgfe_lc.spatial import lock_spatial_splits


def main():
    cfg = load_config()
    raw = resolve_path(cfg, "raw_sites")
    out = resolve_path(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)

    extract_archives(raw)
    files = discover_site_files(raw)
    if not files:
        print("NO_SITES")
        print(f"Place CA_LC_Sites&PhotosInfo.rar or a CSV/SHP/XLSX in: {raw}")
        print("Download: https://data.dcxjegi.cn/portal/metadata/12bc429a-085d-47fa-bbab-e12081934b92")
        return 2

    frames = [harmonize_sites(_read_table(p)) for p in files]
    import pandas as pd
    import geopandas as gpd
    gdf = pd.concat(frames, ignore_index=True)
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf, qc = quality_control(gdf)
    gdf = lock_spatial_splits(
        gdf,
        block_km=cfg["splits"]["block_km"],
        test_fraction_blocks=cfg["splits"]["test_fraction_blocks"],
        n_folds=cfg["splits"]["n_folds"],
        seed=cfg["project"]["seed"],
    )

    keep = gdf[gdf["qc_keep"]].copy()
    export_cols = [c for c in [
        "site_id", "longitude", "latitude", "year", "country", "class_id", "class_name",
        "original_lc", "survey_campaign", "qc_keep", "qc_flag", "block_id", "split",
        "fold_id", "CALC_L1", "CALC_L2", "FAOLCCS_3L", "Point", "Location", "Alt(m)",
    ] if c in keep.columns]
    export = keep[export_cols + (["geometry"] if "geometry" in keep.columns else [])].copy()
    export.to_file(out / "sites_harmonized.gpkg", driver="GPKG")
    export.drop(columns="geometry", errors="ignore").to_csv(out / "sites_harmonized.csv", index=False)
    qc.to_csv(out / "qc_report.csv", index=False)

    tabs = audit_crosstabs(gdf)
    for name, tab in tabs.items():
        tab.to_csv(out / f"crosstab_{name}.csv")

    summary = {
        "source_files": [str(p) for p in files],
        "n_input": int(len(gdf)),
        "n_retained": int(keep.shape[0]),
        "years": keep["year"].value_counts(dropna=False).to_dict(),
        "countries": keep["country"].value_counts(dropna=False).to_dict(),
        "classes": keep["class_name"].value_counts(dropna=False).to_dict(),
        "n_test": int((keep["split"] == "test").sum()),
        "n_traindev": int((keep["split"] == "traindev").sum()),
        "go_nogo": _go_nogo(keep),
    }
    # JSON cannot serialize numpy ints from value_counts in some versions
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if hasattr(o, "item"):
            return o.item()
        return o

    (out / "audit_summary.json").write_text(json.dumps(_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_clean(summary), indent=2, ensure_ascii=False))
    return 0


def _go_nogo(keep):
    class_counts = keep["class_name"].value_counts()
    n_direct = int(keep["year"].isin([2017, 2018]).sum())
    countries = set(keep["country"].dropna())
    key = {"Grassland", "Shrubland", "Bare", "Cropland"}
    enough_classes = sum(int(class_counts.get(c, 0) >= 20) for c in key)
    return {
        "enough_level1_classes": bool(enough_classes >= 4),
        "direct_2017_2018_n": n_direct,
        "direct_enough_for_test": n_direct >= 80,
        "loco_feasible": countries.issuperset({"KAZ", "KGZ", "TJK", "UZB"}),
        "class_counts": class_counts.to_dict(),
    }


if __name__ == "__main__":
    raise SystemExit(main())

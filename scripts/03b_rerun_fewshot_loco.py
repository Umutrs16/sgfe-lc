"""Recompute E3/E8 after few-shot and matched-LOCO fixes; refresh figures and manuscript."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd

from sgfe_lc.eval_matrix import (
    _json_ready,
    e3_fewshot,
    e8_loco_matched,
    spatial_degradation,
)


def main():
    gdf = gpd.read_file(ROOT / "data" / "processed" / "sites_exact_year.gpkg")
    path = ROOT / "results" / "experiment_results_v2.json"
    store = json.loads(path.read_text(encoding="utf-8"))
    e3 = e3_fewshot(gdf)
    store["E3_fewshot"] = e3
    path.write_text(json.dumps(_json_ready(store), indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote E3")
    e8 = e8_loco_matched(gdf)
    store["E8_loco_matched"] = e8
    e4 = pd.DataFrame(store.get("E4_percent") or [])
    store["spatial_degradation"] = spatial_degradation(store.get("E2_features") or {}, e4)
    path.write_text(json.dumps(_json_ready(store), indent=2, ensure_ascii=False), encoding="utf-8")
    tables = ROOT / "results" / "tables"
    tables.mkdir(exist_ok=True)
    if isinstance(e3, pd.DataFrame):
        e3.to_csv(tables / "table_e3_fewshot.csv", index=False)
    if isinstance(e8, pd.DataFrame):
        e8.to_csv(tables / "table_e8_loco_matched.csv", index=False)
    print("E3 rows", 0 if e3 is None else len(e3))
    if isinstance(e3, pd.DataFrame):
        print(e3.to_string(index=False))
    print("E8 rows", 0 if e8 is None else len(e8))
    if isinstance(e8, pd.DataFrame):
        print(e8.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

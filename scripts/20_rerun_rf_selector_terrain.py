"""Rerun Track B RF 1-SE selector on AEF + elev + slope (same predictors as GEE)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.eval_matrix import (
    _json_ready,
    e12_map_2018,
    e6_nested_select,
    e6_nested_stability,
)
from sgfe_lc.models import select_feature_columns


def _protocol(stab: dict):
    rec = stab.get("one_se_recommendation") or stab.get("best_mean") or "hard_0.6"
    if isinstance(rec, str) and rec.startswith("hard"):
        thr = float(rec.split("_", 1)[1]) if "_" in rec else 0.6
        return "hard", thr
    return str(rec), None


def main():
    cfg = load_config()
    gdf = gpd.read_file(resolve_path(cfg, "processed") / "sites_exact_year.gpkg")
    cols = select_feature_columns(gdf.columns, "M4_AEF_TERRAIN")
    print("M4_AEF_TERRAIN n=", len(cols), "has elev", "elev" in cols, "has slope", "slope" in cols)
    store_path = resolve_path(cfg, "results") / "experiment_results_v2.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))

    print("reference seed RF selector...")
    sel = e6_nested_select(gdf, estimator="rf", seed=20260902, target_year=2018)
    print("  selected", sel.get("selected"), "n_pred", sel.get("n_predictors"), sel.get("feature_group"))
    e6 = store.get("E6_temporal") or {}
    e6["nested_selection_rf"] = sel
    store["E6_temporal"] = e6

    print("15-seed RF 1-SE...")
    stab = e6_nested_stability(gdf, n_seeds=15, estimator="rf", target_year=2018)
    store["E6_nested_stability_rf"] = stab
    print("  1-SE", stab.get("one_se_recommendation"), "set", stab.get("one_se_set"))
    print("  freq", stab.get("selection_frequency"))
    print("  means", {k: round(v["mean"], 4) for k, v in (stab.get("protocol_means") or {}).items()})

    protocol, thr = _protocol(stab)
    print("locked protocol", protocol, thr)
    store["E12_map_2018"] = e12_map_2018(gdf, protocol=protocol, high=thr or 0.6)
    print("  sklearn analogue OA", (store["E12_map_2018"].get("M4_AEF_rf_hard") or {}).get("oa"),
          "F1", (store["E12_map_2018"].get("M4_AEF_rf_hard") or store["E12_map_2018"].get(f"M4_AEF_rf_{protocol}") or {}).get("supported_macro_f1"))
    store_path.write_text(json.dumps(_json_ready(store), indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", store_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rerun only historical-label experiments after GLC 2015/2016 entered S_product.

Does not relock spatial splits. Does not retrain the GEE mosaic.
Does not rerun the 30-draw classifier-robustness grid.
"""
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
    e6_nested_select,
    e6_nested_stability,
    e6_temporal,
    e6_threshold_sensitivity,
    e12_map_2018,
)


def _protocol(rec: dict) -> tuple[str, float]:
    rec = rec or {}
    mode = rec.get("one_se_recommendation") or rec.get("best_mean") or "none"
    if isinstance(mode, str) and str(mode).startswith("hard"):
        try:
            return "hard", float(str(mode).split("_", 1)[1])
        except (IndexError, ValueError):
            return "hard", 0.80
    if mode in {"none", "ungated", "soft"}:
        return str(mode), 0.80
    return "none", 0.80


def main():
    cfg = load_config()
    gdf = gpd.read_file(resolve_path(cfg, "processed") / "sites_exact_year.gpkg")
    store_path = resolve_path(cfg, "results") / "experiment_results_v2.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    high = cfg["migration"]["high_confidence"]

    print("E6 temporal / threshold / XGB nested ...")
    e6 = e6_temporal(gdf, high=high)
    e6["nested_selection"] = e6_nested_select(gdf)
    store["E6_temporal"] = e6
    store["E6_threshold_sensitivity"] = e6_threshold_sensitivity(gdf)
    store["E6_nested_stability"] = e6_nested_stability(gdf, n_seeds=15)
    print("  XGB 1-SE", store["E6_nested_stability"].get("one_se_recommendation"),
          "freq", store["E6_nested_stability"].get("selection_frequency"))

    print("RF 2018-targeted nested CV ...")
    store["E6_nested_select_rf"] = e6_nested_select(gdf, estimator="rf", target_year=2018)
    store["E6_nested_stability_rf"] = e6_nested_stability(gdf, n_seeds=15, estimator="rf", target_year=2018)
    print("  RF 1-SE", store["E6_nested_stability_rf"].get("one_se_recommendation"),
          "set", store["E6_nested_stability_rf"].get("one_se_set"),
          "freq", store["E6_nested_stability_rf"].get("selection_frequency"))

    protocol, thr = _protocol(store["E6_nested_stability_rf"])
    print("E12 sklearn analogue", protocol, thr)
    store["E12_map_2018"] = e12_map_2018(gdf, protocol=protocol, high=thr)
    print("  train hist", store["E12_map_2018"].get("n_train_hist"),
          "classes", store["E12_map_2018"].get("training_class_support"))

    store_path.write_text(json.dumps(_json_ready(store), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

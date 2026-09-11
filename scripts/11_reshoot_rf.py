"""Recompute 5/10/20/25 few-shot, RF protocol selection, and Track B RF headlines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.eval_matrix import (
    HEADLINE_SHOTS,
    _json_ready,
    e3_classifier_robustness,
    e3_fewshot,
    e6_nested_select,
    e6_nested_stability,
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

    print("E3 few-shot", HEADLINE_SHOTS, "...")
    store["E3_fewshot"] = e3_fewshot(gdf, shots=HEADLINE_SHOTS, repeats=30)
    print("  support", store["E3_fewshot"].get("train_support"))
    print("  realized", store["E3_fewshot"].get("realized_max"))
    print("  capped", store["E3_fewshot"].get("capped_classes"))
    for r in store["E3_fewshot"].get("paired_delta") or []:
        print(" ", r)

    print("E3 classifier robustness ...")
    store["E3_classifier_robustness"] = e3_classifier_robustness(gdf, shots=HEADLINE_SHOTS, repeats=15)

    print("E6 RF nested-CV stability ...")
    store["E6_nested_select_rf"] = e6_nested_select(gdf, estimator="rf")
    store["E6_nested_stability_rf"] = e6_nested_stability(gdf, n_seeds=15, estimator="rf")
    print("  selected", store["E6_nested_select_rf"].get("selected"))
    print("  freq", store["E6_nested_stability_rf"].get("selection_frequency"))
    print("  1SE", store["E6_nested_stability_rf"].get("one_se_recommendation"))

    protocol, high = _protocol(store["E6_nested_stability_rf"])
    print("E12 Track B protocol", protocol, high, "...")
    store["E12_map_2018"] = e12_map_2018(gdf, protocol=protocol, high=high)
    rf = store["E12_map_2018"].get(f"M4_AEF_rf_{protocol}") or {}
    print("  RF OA", rf.get("oa"), "supported F1", rf.get("supported_macro_f1"), "shrub", rf.get("shrub_f1"))

    store_path.write_text(json.dumps(_json_ready(store), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

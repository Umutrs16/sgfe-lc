"""Compute 2018 locked-test block composition and unrounded Esri ΔF1 bounds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.eval_matrix import (
    TRACK_B_SUPPORTED,
    _json_ready,
    _to_rangeland,
    classification_report_dict,
    exact_year,
)
from sgfe_lc.metrics import block_bootstrap_delta


def main():
    cfg = load_config()
    gdf = gpd.read_file(resolve_path(cfg, "processed") / "sites_exact_year.gpkg")
    preds = pd.read_csv(resolve_path(cfg, "tables") / "gee_map_2018_preds.csv")
    ey = exact_year(gdf)
    year = pd.to_numeric(ey.get("year"), errors="coerce")
    test = ey[(ey["split"] == "test") & year.eq(2018)].copy()
    test["site_id"] = test["site_id"].astype(str)
    preds["site_id"] = preds["site_id"].astype(str)
    merged = test.merge(preds[["site_id", "gee_pred"]], on="site_id", how="left")
    yt = pd.to_numeric(merged["class_id"], errors="coerce")
    yp = pd.to_numeric(merged["gee_pred"], errors="coerce")
    ok = yt.notna() & yp.notna()
    labels = list(TRACK_B_SUPPORTED)
    rows = []
    for blk, sub in merged.loc[ok].groupby(merged.loc[ok, "block_id"].astype(str)):
        counts = sub["class_id"].astype(int).value_counts()
        rep = classification_report_dict(
            pd.to_numeric(sub["class_id"], errors="coerce").astype(int),
            pd.to_numeric(sub["gee_pred"], errors="coerce").astype(int),
            labels=labels,
        )
        rows.append({
            "block_id": str(blk),
            "n": int(len(sub)),
            "crop": int(counts.get(1, 0)),
            "grass": int(counts.get(3, 0)),
            "shrub": int(counts.get(4, 0)),
            "bare": int(counts.get(5, 0)),
            "oa": float(rep.get("oa")),
            "macro_f1": float(rep.get("macro_f1")),
        })
    rows = sorted(rows, key=lambda r: (-r["n"], r["block_id"]))
    n_total = int(sum(r["n"] for r in rows))
    n_max = int(rows[0]["n"]) if rows else 0
    share = float(n_max / n_total) if n_total else float("nan")

    yt_r = _to_rangeland(yt[ok].astype(int))
    gee_r = _to_rangeland(yp[ok].astype(int))
    esri_r = _to_rangeland(pd.to_numeric(merged.loc[ok, "esri_l1"], errors="coerce"))
    blocks = merged.loc[ok, "block_id"].astype(str)
    m = yt_r.notna() & gee_r.notna() & esri_r.notna() & (esri_r > 0)
    rec = block_bootstrap_delta(yt_r[m], gee_r[m], esri_r[m], blocks[m], labels, seed=20260902)
    # Keep full-precision copies; also count bootstrap mass at/above zero.
    yt_a = np.asarray(yt_r[m])
    pa = np.asarray(gee_r[m])
    pb = np.asarray(esri_r[m])
    blk = np.asarray(blocks[m])
    rng = np.random.default_rng(20260902)
    uniq = np.unique(blk)
    deltas = []
    from sklearn.metrics import f1_score
    for _ in range(1000):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(blk == b)[0] for b in drawn])
        if len(idx) < 8:
            continue
        fa = f1_score(yt_a[idx], pa[idx], labels=labels, average="macro", zero_division=0)
        fb = f1_score(yt_a[idx], pb[idx], labels=labels, average="macro", zero_division=0)
        deltas.append(fa - fb)
    arr = np.asarray(deltas)
    q025, q975 = np.quantile(arr, [0.025, 0.975])
    payload = {
        "n_test_2018": n_total,
        "n_blocks": int(len(rows)),
        "dominant_block": rows[0]["block_id"] if rows else None,
        "dominant_n": n_max,
        "dominant_share": share,
        "rows": rows,
        "esri_delta": rec,
        "esri_ci_lo_raw": float(q025),
        "esri_ci_hi_raw": float(q975),
        "esri_boot_n": int(len(arr)),
        "esri_boot_ge0": int((arr >= 0).sum()),
        "esri_boot_gt0": int((arr > 0).sum()),
        "esri_boot_min": float(arr.min()) if len(arr) else None,
        "esri_boot_max": float(arr.max()) if len(arr) else None,
    }
    print(json.dumps(payload, indent=2))
    store_path = resolve_path(cfg, "results") / "experiment_results_v2.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    store["E12_test_block_composition"] = payload
    esri = ((store.get("E12_paired_product_deltas") or {}).get("Esri") or {})
    esri["ci_lo"] = float(q025)
    esri["ci_hi"] = float(q975)
    esri["ci_lo_raw"] = float(q025)
    esri["ci_hi_raw"] = float(q975)
    esri["boot_ge0"] = payload["esri_boot_ge0"]
    store.setdefault("E12_paired_product_deltas", {})["Esri"] = esri
    store_path.write_text(json.dumps(_json_ready(store), indent=2, ensure_ascii=False), encoding="utf-8")
    print("Esri raw CI", q025, q975, "n>=0", payload["esri_boot_ge0"], "max", payload["esri_boot_max"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

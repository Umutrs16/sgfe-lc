"""Print GEE export task / asset status for the regional maps."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgfe_lc.config import load_config, resolve_path
from sgfe_lc.gee.auth import init_ee


def main():
    cfg = load_config()
    ee = init_ee()
    rec = {}
    for name in ("gee_export_tasks_hard060.json", "gee_export_tasks_hard060_kaz_tiles.json",
                 "gee_export_tasks_hard090_kaz_tiles.json", "gee_export_tasks_hard090.json",
                 "gee_export_tasks_hard_0.9.json", "gee_export_tasks.json",
                 "gee_export_tasks_ungated.json"):
        rec_path = resolve_path(cfg, "results") / name
        if rec_path.exists():
            print("reading", rec_path.name)
            rec.update(json.loads(rec_path.read_text(encoding="utf-8")))
    # Operations API
    ops = ee.data.listOperations()
    wanted = {v["id"] for v in rec.values() if isinstance(v, dict) and v.get("id")}
    print("recorded tasks", len(rec))
    for op in ops:
        meta = op.get("metadata") or {}
        desc = meta.get("description") or op.get("name", "")
        if "SGFE" in str(desc) or any(tid in str(op.get("name", "")) for tid in wanted):
            print(desc, op.get("done"), meta.get("state"), op.get("error", {}).get("message", ""))

    root = cfg["project"]["asset_root"]
    print("assets under", root)
    try:
        kids = ee.data.listAssets({"parent": root}).get("assets", [])
        for a in kids:
            print(" ", a.get("id"), a.get("type"))
    except Exception as e:
        print("listAssets failed:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

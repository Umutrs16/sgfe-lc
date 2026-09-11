"""Download simplified FAO GAUL 2015 Admin-0 polygons for the five republics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgfe_lc.gee.auth import init_ee


NAMES = ["Kazakhstan", "Kyrgyzstan", "Tajikistan", "Uzbekistan", "Turkmenistan"]
ISO = {
    "Kazakhstan": "KAZ",
    "Kyrgyzstan": "KGZ",
    "Tajikistan": "TJK",
    "Uzbekistan": "UZB",
    "Turkmenistan": "TKM",
}


def main():
    ee = init_ee()
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.inList("ADM0_NAME", NAMES))
    simple = gaul.map(lambda f: f.simplify(maxError=2000).set("iso3", f.get("ADM0_NAME")))
    info = simple.getInfo()
    for feat in info.get("features", []):
        name = feat.get("properties", {}).get("ADM0_NAME")
        feat.setdefault("properties", {})["iso3"] = ISO.get(name, name)
    dest = ROOT / "data" / "lookups" / "ca_gaul2015.geojson"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(info), encoding="utf-8")
    print("Wrote", dest, "features", len(info.get("features", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

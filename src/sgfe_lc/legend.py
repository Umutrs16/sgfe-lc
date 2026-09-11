from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT

LEVEL1 = {
    1: "Cropland",
    2: "Forest",
    3: "Grassland",
    4: "Shrubland",
    5: "Bare",
    6: "Water",
    7: "Built",
    8: "SnowIce",
}

LEVEL1_EVAL = [k for k in sorted(LEVEL1) if k != 8]  # SnowIce is absent in the 902-site table
# Headline quantitative classes (a priori): sufficient exact-year support.
BENCHMARK_CLASSES = [1, 3, 4, 5]  # crop, grass, shrub, bare
# Official map legend: exact-year support required. Built is weakly supported (n=2).
# Forest / Water / Snow-ice are never official map classes, even with historical extras.
OFFICIAL_MAP_CLASSES = [1, 3, 4, 5, 7]  # crop, grass, shrub, bare, built
UNOFFICIAL_MAP_CLASSES = [2, 6, 8]  # forest, water, snow/ice
RARE_MAP_CLASSES = [2, 6, 7]
CORE_CLASSES = BENCHMARK_CLASSES
RANGELAND_ID = 3  # common-ontology: grassland + shrubland → rangeland / grass-shrub complex

LEVEL1_NAME_TO_ID = {v.lower(): k for k, v in LEVEL1.items()}
LEVEL1_NAME_TO_ID.update(
    {
        "crop": 1,
        "cropland": 1,
        "farmland": 1,
        "agriculture": 1,
        "cultivated": 1,
        "cultivated land": 1,
        "耕地": 1,
        "农田": 1,
        "forest": 2,
        "tree": 2,
        "trees": 2,
        "tree cover": 2,
        "林地": 2,
        "森林": 2,
        "grass": 3,
        "grassland": 3,
        "rangeland": 3,
        "pasture": 3,
        "steppe": 3,
        "草地": 3,
        "草原": 3,
        "shrub": 4,
        "shrubland": 4,
        "shrub & scrub": 4,
        "shrub_and_scrub": 4,
        "灌丛": 4,
        "灌木": 4,
        "bare": 5,
        "bare land": 5,
        "bare ground": 5,
        "sparse": 5,
        "sparse vegetation": 5,
        "desert": 5,
        "裸地": 5,
        "稀疏植被": 5,
        "荒漠": 5,
        "water": 6,
        "wetland": 6,
        "flooded": 6,
        "flooded_vegetation": 6,
        "水体": 6,
        "湿地": 6,
        "built": 7,
        "built-up": 7,
        "builtup": 7,
        "urban": 7,
        "impervious": 7,
        "建设用地": 7,
        "城镇": 7,
        "snow": 8,
        "ice": 8,
        "snowice": 8,
        "snow/ice": 8,
        "snow_and_ice": 8,
        "冰雪": 8,
        "arable land": 1,
        "garden land": 1,
        "gardenland": 1,
        "woodland": 2,
        "forest land": 2,
        "shrubbery land": 4,
        "shrubbery": 4,
        "high coverage": 3,
        "moderate coverage": 3,
        "low coverage": 3,
        "residential and industrial land": 7,
        "urban land": 7,
        "rural settlements": 7,
        "other construction land": 7,
        "tranportation land": 7,
        "transportation land": 7,
        "highway land": 7,
        "railway land": 7,
        "rural land": 7,
        "unutilized land": 5,
        "bare land": 5,
        "barren lands": 5,
        "desert": 5,
        "gobi": 5,
        "saline land": 5,
        "canal": 6,
        "lake": 6,
        "pond and reservoir": 6,
    }
)

DW_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DW_TO_L1 = {
    "water": 6,
    "trees": 2,
    "grass": 3,
    "flooded_vegetation": 6,
    "crops": 1,
    "shrub_and_scrub": 4,
    "built": 7,
    "bare": 5,
    "snow_and_ice": 8,
}

# Esri 10 m annual (2017+). Class 11 rangeland mixes grass and shrub.
ESRI_TO_L1 = {
    1: 6,  # water
    2: 2,  # trees
    4: 6,  # flooded vegetation
    5: 1,  # crops
    7: 7,  # built
    8: 5,  # bare
    9: 8,  # snow/ice
    11: 3,  # rangeland -> grassland (flagged as mixed)
}

# GLC_FCS30D fine codes (Zhang et al., 2021/2024) -> Level-1
GLC_TO_L1: dict[int, int] = {}
for code in (10, 11, 12, 20):
    GLC_TO_L1[code] = 1
for code in (51, 52, 61, 62, 71, 72, 81, 82, 91, 92):
    GLC_TO_L1[code] = 2
for code in (120, 121, 122, 152):
    GLC_TO_L1[code] = 4
for code in (130, 140, 153):
    GLC_TO_L1[code] = 3
for code in (150, 200, 201, 202):
    GLC_TO_L1[code] = 5
for code in (181, 182, 183, 184, 185, 186, 187, 210):
    GLC_TO_L1[code] = 6
GLC_TO_L1[190] = 7
GLC_TO_L1[220] = 8

NABOUREH_TO_L1 = {
    "bare land": 5,
    "built-up": 7,
    "shrub lands": 4,
    "forest": 2,
    "cropland": 1,
    "grassland": 3,
    "wetland": 6,
    "water": 6,
    "snow/ice": 8,
    "snow/ice ": 8,
}


def load_crosswalk(path: Path | None = None) -> pd.DataFrame:
    path = path or (PROJECT_ROOT / "data" / "lookups" / "legend_crosswalk.csv")
    return pd.read_csv(path)


def harmonize_label(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        iv = int(value)
        if iv in LEVEL1:
            return iv
        if iv in GLC_TO_L1:
            return GLC_TO_L1[iv]
        if iv in ESRI_TO_L1:
            return ESRI_TO_L1[iv]
    key = str(value).strip().lower()
    if key in LEVEL1_NAME_TO_ID:
        return LEVEL1_NAME_TO_ID[key]
    if key in NABOUREH_TO_L1:
        return NABOUREH_TO_L1[key]
    if key in DW_TO_L1:
        return DW_TO_L1[key]
    return None


def remap_series(values, mapper: dict) -> pd.Series:
    return pd.Series(values).map(mapper)


def glc_remap_lists() -> tuple[list[int], list[int]]:
    from_codes = list(GLC_TO_L1.keys())
    to_codes = [GLC_TO_L1[c] for c in from_codes]
    return from_codes, to_codes

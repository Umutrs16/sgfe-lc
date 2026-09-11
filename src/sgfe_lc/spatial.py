from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.model_selection import GroupKFold


def lonlat_to_block(lon, lat, block_km: float = 100.0, crs: str = "EPSG:6933") -> pd.Series:
    """Equal-area blocks (default WGS 84 / NSIDC EASE-Grid 2.0). Avoid Web Mercator at mid-latitudes."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    size = block_km * 1000.0
    col = np.floor(x / size).astype(int)
    row = np.floor(y / size).astype(int)
    return pd.Series([f"{c}_{r}" for c, r in zip(col, row)], index=getattr(lon, "index", None))


def lock_spatial_splits_stratified(
    df: pd.DataFrame,
    block_km: float = 100.0,
    test_fraction_blocks: float = 0.35,
    min_test_sites: int = 130,
    n_folds: int = 5,
    seed: int = 20260902,
    crs: str = "EPSG:6933",
    eligible_mask=None,
    stratify_col: str = "country",
) -> pd.DataFrame:
    """Lock test blocks with country stratification and a minimum exact-year test size.

    Blocks that contain only ineligible (historical) sites cannot become test blocks.
    Historical sites that fall in a chosen test block are excluded from training later.
    """
    out = df.copy()
    out["block_id"] = lonlat_to_block(out["longitude"], out["latitude"], block_km, crs=crs)
    rng = np.random.default_rng(seed)
    elig = out if eligible_mask is None else out.loc[eligible_mask]
    test_blocks: set[str] = set()

    if stratify_col in elig.columns:
        for key, sub in elig.groupby(elig[stratify_col].fillna("NA")):
            blocks = list(sub["block_id"].dropna().unique())
            rng.shuffle(blocks)
            n_take = max(1, int(round(len(blocks) * test_fraction_blocks))) if blocks else 0
            test_blocks.update(blocks[:n_take])
    else:
        blocks = list(elig["block_id"].dropna().unique())
        rng.shuffle(blocks)
        n_take = max(1, int(round(len(blocks) * test_fraction_blocks)))
        test_blocks.update(blocks[:n_take])

    def _n_test():
        return int(elig["block_id"].isin(test_blocks).sum())

    remaining = [b for b in elig["block_id"].dropna().unique() if b not in test_blocks]
    # Prefer leftover blocks that contain non-cropland exact-year sites
    def _minority_score(block):
        sub = elig[elig["block_id"] == block]
        if "class_id" not in sub.columns:
            return len(sub)
        return int((pd.to_numeric(sub["class_id"], errors="coerce") != 1).sum())

    remaining = sorted(remaining, key=_minority_score, reverse=True)
    i = 0
    while _n_test() < min_test_sites and i < len(remaining):
        test_blocks.add(remaining[i])
        i += 1

    out["split"] = np.where(out["block_id"].isin(test_blocks), "test", "traindev")
    train = out["split"] == "traindev"
    groups = out.loc[train, "block_id"].astype(str)
    unique_groups = groups.nunique()
    folds = min(n_folds, max(2, unique_groups))
    out["fold_id"] = -1
    if unique_groups >= 2 and train.sum() >= folds:
        gkf = GroupKFold(n_splits=folds)
        dummy = np.zeros(int(train.sum()))
        for i, (_, val_idx) in enumerate(gkf.split(dummy, groups=groups)):
            idx = out.index[train][val_idx]
            out.loc[idx, "fold_id"] = i
    return out


def lock_spatial_splits_constrained(
    df: pd.DataFrame,
    block_km: float = 100.0,
    test_fraction_blocks: float = 0.20,
    n_folds: int = 5,
    seed: int = 20260902,
    crs: str = "EPSG:6933",
    eligible_mask=None,
    stratify_col: str = "country",
    headline_classes=(1, 3, 4, 5),
    min_train_per_class: int = 20,
    min_test_per_class: int = 10,
    max_test_site_fraction: float = 0.33,
) -> pd.DataFrame:
    """Country-stratified block lock with a priori per-class train/test floors.

    Rare classes that cannot meet the floors are left as-is and later excluded
    from headline statistics (not patched after seeing results).
    """
    out = df.copy()
    out["block_id"] = lonlat_to_block(out["longitude"], out["latitude"], block_km, crs=crs)
    rng = np.random.default_rng(seed)
    elig = out if eligible_mask is None else out.loc[eligible_mask]
    n_elig = max(1, len(elig))
    test_blocks: set[str] = set()

    if stratify_col in elig.columns:
        for _, sub in elig.groupby(elig[stratify_col].fillna("NA")):
            blocks = list(sub["block_id"].dropna().unique())
            rng.shuffle(blocks)
            n_take = max(1, int(round(len(blocks) * test_fraction_blocks))) if blocks else 0
            test_blocks.update(blocks[:n_take])
    else:
        blocks = list(elig["block_id"].dropna().unique())
        rng.shuffle(blocks)
        test_blocks.update(blocks[: max(1, int(round(len(blocks) * test_fraction_blocks)))])

    def _counts(tb):
        is_test = elig["block_id"].isin(tb)
        y = pd.to_numeric(elig.get("class_id"), errors="coerce")
        outc = {}
        for c in headline_classes:
            outc[c] = {
                "test": int(((y == c) & is_test).sum()),
                "train": int(((y == c) & ~is_test).sum()),
            }
        return outc, int(is_test.sum())

    remaining = [b for b in elig["block_id"].dropna().unique() if b not in test_blocks]

    def _blocks_with_class(blocks, cid, in_test: bool):
        pool = [b for b in blocks if b in test_blocks] if in_test else [b for b in blocks if b not in test_blocks]
        scored = []
        for b in pool:
            sub = elig[elig["block_id"] == b]
            y = pd.to_numeric(sub.get("class_id"), errors="coerce")
            n = int((y == cid).sum())
            if n > 0:
                scored.append((n, len(sub), b))
        scored.sort()
        return [b for _, _, b in scored]

    for cid in headline_classes:
        counts, n_test = _counts(test_blocks)
        need = min_test_per_class - counts[cid]["test"]
        if need <= 0:
            continue
        for b in _blocks_with_class(remaining, cid, in_test=False):
            if (n_test / n_elig) >= max_test_site_fraction:
                break
            test_blocks.add(b)
            if b in remaining:
                remaining.remove(b)
            counts, n_test = _counts(test_blocks)
            if counts[cid]["test"] >= min_test_per_class:
                break

    for cid in headline_classes:
        counts, _ = _counts(test_blocks)
        while counts[cid]["train"] < min_train_per_class:
            donors = _blocks_with_class(list(test_blocks), cid, in_test=True)
            moved = False
            for b in reversed(donors):
                trial = set(test_blocks)
                trial.discard(b)
                c2, _ = _counts(trial)
                if c2[cid]["test"] >= min_test_per_class:
                    test_blocks.discard(b)
                    remaining.append(b)
                    moved = True
                    break
            if not moved:
                break
            counts, _ = _counts(test_blocks)

    # Drop surplus cropland-only test blocks if the site fraction is still high.
    counts, n_test = _counts(test_blocks)
    while (n_test / n_elig) > max_test_site_fraction:
        crop_blocks = []
        for b in list(test_blocks):
            sub = elig[elig["block_id"] == b]
            y = pd.to_numeric(sub.get("class_id"), errors="coerce")
            if int((y.isin(list(headline_classes)) & (y != 1)).sum()) == 0:
                crop_blocks.append((len(sub), b))
        if not crop_blocks:
            break
        crop_blocks.sort(reverse=True)
        test_blocks.discard(crop_blocks[0][1])
        counts, n_test = _counts(test_blocks)

    out["split"] = np.where(out["block_id"].isin(test_blocks), "test", "traindev")
    train = out["split"] == "traindev"
    groups = out.loc[train, "block_id"].astype(str)
    unique_groups = groups.nunique()
    folds = min(n_folds, max(2, unique_groups))
    out["fold_id"] = -1
    if unique_groups >= 2 and train.sum() >= folds:
        gkf = GroupKFold(n_splits=folds)
        dummy = np.zeros(int(train.sum()))
        for i, (_, val_idx) in enumerate(gkf.split(dummy, groups=groups)):
            idx = out.index[train][val_idx]
            out.loc[idx, "fold_id"] = i
    return out


def lock_spatial_splits(
    df: pd.DataFrame,
    block_km: float = 100.0,
    test_fraction_blocks: float = 0.20,
    n_folds: int = 5,
    seed: int = 20260902,
    crs: str = "EPSG:6933",
    eligible_mask=None,
) -> pd.DataFrame:
    """Lock a held-out spatial test set, then assign development folds on the rest.

    Test blocks are never used to tune migration thresholds or pseudo-labels.
    """
    out = df.copy()
    out["block_id"] = lonlat_to_block(out["longitude"], out["latitude"], block_km, crs=crs)
    rng = np.random.default_rng(seed)
    pool = out if eligible_mask is None else out.loc[eligible_mask]
    blocks = pool["block_id"].dropna().unique()
    rng.shuffle(blocks)
    n_test = max(1, int(round(len(blocks) * test_fraction_blocks)))
    test_blocks = set(blocks[:n_test])
    out["split"] = np.where(out["block_id"].isin(test_blocks), "test", "traindev")

    train = out["split"] == "traindev"
    groups = out.loc[train, "block_id"].astype(str)
    unique_groups = groups.nunique()
    folds = min(n_folds, max(2, unique_groups))
    out["fold_id"] = -1
    if unique_groups >= 2 and train.sum() >= folds:
        gkf = GroupKFold(n_splits=folds)
        dummy = np.zeros(int(train.sum()))
        for i, (_, val_idx) in enumerate(gkf.split(dummy, groups=groups)):
            idx = out.index[train][val_idx]
            out.loc[idx, "fold_id"] = i
    return out


def loco_masks(df: pd.DataFrame, countries=("KAZ", "KGZ", "TJK", "UZB")) -> dict[str, pd.Series]:
    masks = {}
    for c in countries:
        masks[c] = df["country"] == c
    return masks


def environmental_stratum(elevation, precip, aridity=None) -> pd.Series:
    """Climatic–topographic bins (not formal WWF ecoregions)."""
    elev = pd.Series(elevation)
    rain = pd.Series(precip)
    elev_bin = pd.cut(elev, [-np.inf, 500, 1500, np.inf], labels=["low", "mid", "high"])
    rain_bin = pd.cut(rain, [-np.inf, 150, 350, np.inf], labels=["arid", "semiarid", "subhumid"])
    return elev_bin.astype(str) + "_" + rain_bin.astype(str)


def ecoregion_bin(elevation, precip, aridity=None) -> pd.Series:
    """Backward-compatible alias for environmental_stratum."""
    return environmental_stratum(elevation, precip, aridity)

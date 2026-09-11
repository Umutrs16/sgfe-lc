import numpy as np
import pandas as pd

from sgfe_lc.experiments import e2_feature_ablation, e3_label_efficiency, e6_loco
from sgfe_lc.spatial import lock_spatial_splits


def _synthetic(n=160, seed=0):
    rng = np.random.default_rng(seed)
    cls = rng.integers(1, 6, n)
    df = pd.DataFrame({
        "site_id": [f"S{i}" for i in range(n)],
        "longitude": rng.uniform(65, 80, n),
        "latitude": rng.uniform(38, 52, n),
        "class_id": cls,
        "country": rng.choice(["KAZ", "KGZ", "TJK", "UZB"], n),
        "year": rng.choice([2013, 2014, 2018], n, p=[0.3, 0.2, 0.5]),
        "tier": "A",
    })
    for i in range(8):
        df[f"A{i:02d}"] = rng.normal(cls, 0.8, n)
    for i, name in enumerate(["l8_NDVI_med", "l8_BSI_med", "s2_NDVI_med", "s1_VV_med"]):
        df[name] = rng.normal(cls * 0.1, 0.3, n)
    return lock_spatial_splits(df, block_km=250, seed=1)


def test_ablation_and_loco_on_synthetic():
    df = _synthetic()
    e2 = e2_feature_ablation(df, estimator="rf", seed=1)
    assert "M4_AEF" in e2
    assert e2["M4_AEF"]["macro_f1_mean"] == e2["M4_AEF"]["macro_f1_mean"]
    loco = e6_loco(df, estimator="rf", seed=1)
    assert not loco.empty
    le = e3_label_efficiency(df, fractions=[0.4, 1.0], repeats=2, estimator="rf", seed=1)
    assert not le.empty

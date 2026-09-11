import numpy as np
import pandas as pd
import pytest

from sgfe_lc.aoa import aoa_threshold, dissimilarity_index
from sgfe_lc.metrics import block_bootstrap_delta, classification_report_dict, mcnemar_test
from sgfe_lc.spatial import lock_spatial_splits, lock_spatial_splits_constrained, lonlat_to_block


def test_perfect_classification():
    y = [1, 2, 3, 1, 2, 3]
    r = classification_report_dict(y, y, labels=[1, 2, 3])
    assert r["oa"] == 1
    assert r["macro_f1"] == 1


def test_mcnemar_detects_difference():
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    a = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    b = np.array([2, 2, 2, 2, 1, 1, 1, 1])
    out = mcnemar_test(y, a, b)
    assert out["p_value"] < 0.05


def test_spatial_blocks_lock_test():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "longitude": rng.uniform(66, 80, 80),
        "latitude": rng.uniform(40, 50, 80),
        "class_id": rng.integers(1, 6, 80),
    })
    out = lock_spatial_splits(df, block_km=200, seed=1)
    assert set(out["split"]) <= {"test", "traindev"}
    assert out["block_id"].nunique() >= 2
    assert (out["split"] == "test").any()
    blocks = lonlat_to_block(df["longitude"], df["latitude"], 200)
    assert len(blocks) == 80


def test_aoa_threshold_excludes_self():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4))
    di_loo = dissimilarity_index(x, x, k=5, leave_one_out=True)
    di_self = dissimilarity_index(x, x, k=5, leave_one_out=False)
    assert di_loo.mean() > di_self.mean()
    assert aoa_threshold(x, k=5) > 0


def test_constrained_lock_has_both_splits():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "longitude": rng.uniform(66, 80, n),
        "latitude": rng.uniform(40, 52, n),
        "class_id": rng.choice([1, 3, 4, 5], n, p=[0.4, 0.3, 0.15, 0.15]),
        "country": rng.choice(["KAZ", "UZB"], n, p=[0.8, 0.2]),
    })
    out = lock_spatial_splits_constrained(df, block_km=200, seed=2, test_fraction_blocks=0.2)
    assert set(out["split"]) <= {"test", "traindev"}
    assert (out["split"] == "test").any()
    assert (out["split"] == "traindev").any()


def test_block_bootstrap_delta_uses_observed_f1():
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    a = y.copy()
    b = np.array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1])
    blocks = np.array(["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"])
    out = block_bootstrap_delta(y, a, b, blocks, labels=[1, 2, 3], n_boot=200, seed=1)
    from sklearn.metrics import f1_score
    obs = f1_score(y, a, labels=[1, 2, 3], average="macro") - f1_score(y, b, labels=[1, 2, 3], average="macro")
    assert abs(out["delta"] - obs) < 1e-12
    assert out["ci_lo"] <= out["ci_hi"]


def test_fewshot_uses_unique_traindev_only_and_caps_k():
    from sgfe_lc.eval_matrix import HEADLINE_SHOTS, e3_fewshot

    rng = np.random.default_rng(0)
    n = 80
    # Bare (5) has only 8 traindev rows so a requested 40-shot must cap.
    class_id = np.array([1] * 24 + [3] * 24 + [4] * 16 + [5] * 16)
    split = np.array(["traindev"] * 20 + ["test"] * 4 +
                     ["traindev"] * 20 + ["test"] * 4 +
                     ["traindev"] * 12 + ["test"] * 4 +
                     ["traindev"] * 8 + ["test"] * 8)
    df = pd.DataFrame({
        "role": ["exact_year"] * n,
        "class_id": class_id,
        "split": split,
        "block_id": [f"b{i % 8}" for i in range(n)],
    })
    for i in range(8):
        df[f"A{i:02d}"] = rng.normal(size=n)
        df[f"l8_b{i}"] = rng.normal(size=n)
        df[f"s1_b{i}"] = rng.normal(size=n)
        df[f"s2_b{i}"] = rng.normal(size=n)
    out = e3_fewshot(df, shots=(5, 10, 40), repeats=3, estimator="logreg")
    assert out["train_support"][5] == 8
    assert out["realized_max"][5] == 8
    assert "5" in out["capped_classes"]
    assert max(HEADLINE_SHOTS) <= 25


def test_class_support_keeps_forest_water_historical_only():
    from sgfe_lc.eval_matrix import _class_support, e1_exact_year_products

    df = pd.DataFrame({
        "role": ["exact_year"] * 8 + ["historical"] * 4,
        "class_id": [1, 1, 3, 3, 4, 5, 5, 7, 2, 6, 1, 3],
        "dw_l1": [1, 1, 3, 3, 4, 5, 5, 7, 2, 6, 1, 3],
        "esri_l1": [1, 1, 3, 3, 3, 5, 5, 7, 3, 6, 1, 3],
        "glc_l1": [1, 1, 3, 3, 4, 5, 5, 7, 2, 6, 1, 3],
    })
    rows = {r["class"]: r for r in _class_support(df)}
    assert rows["Forest"]["exact_year_n"] == 0
    assert rows["Water"]["exact_year_n"] == 0
    assert rows["Forest"]["historical_n"] == 1
    assert rows["Water"]["historical_n"] == 1
    assert rows["Built"]["exact_year_n"] == 1
    assert rows["Cropland"]["headline"] is True
    assert rows["Forest"]["headline"] is False
    out = e1_exact_year_products(df)
    assert out["n_exact_year"] == 8
    assert out["n_headline"] == 7  # 8 exact-year minus Built
    assert out["n_headline"] == out["n_exact_year"] - rows["Built"]["exact_year_n"]


def test_e1_common_ontology_excludes_built():
    from sgfe_lc.eval_matrix import e1_exact_year_products

    df = pd.DataFrame({
        "role": ["exact_year"] * 12,
        "class_id": [1, 1, 1, 3, 3, 3, 4, 4, 5, 5, 7, 7],
        "dw_l1": [1, 1, 3, 3, 3, 4, 4, 5, 5, 5, 7, 1],
        "esri_l1": [1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 7, 7],
        "glc_l1": [1, 3, 1, 3, 4, 3, 4, 5, 5, 1, 7, 7],
    })
    out = e1_exact_year_products(df)
    assert 7 not in (out.get("headline_classes_common") or [])
    for name, rep in (out.get("common_ontology") or {}).items():
        assert 7 not in (rep.get("labels") or [])
        assert name or True
    snow = next(r for r in out["class_support"] if r["class_id"] == 8)
    assert snow["exact_year_n"] == 0 and snow["historical_n"] == 0
    assert snow["map_class"] is False


def test_classic_one_se_prefers_ungated_over_hard():
    from sgfe_lc.eval_matrix import _one_se_recommend

    stats = {
        "none": {"mean": 0.433468, "se": 0.002075},
        "ungated": {"mean": 0.438138, "se": 0.002395},
        "hard_0.9": {"mean": 0.438570, "se": 0.002665},
    }
    rec, floor, in_set = _one_se_recommend(stats)
    assert rec == "ungated"
    assert "ungated" in in_set
    assert "hard_0.9" in in_set
    assert "none" not in in_set
    assert floor < 0.4386


def test_field_anchored_s_product_uses_2015_2016_when_present():
    from sgfe_lc.merge import field_anchored_s_product

    df = pd.DataFrame({
        "year": [2013],
        "class_id": [3],
        "glc_l1_2013": [3],
        "glc_l1_2014": [3],
        "glc_l1_2015": [1],
        "glc_l1_2016": [1],
        "glc_l1_2017": [3],
        "glc_l1_2018": [3],
    })
    s = field_anchored_s_product(df)
    assert float(s.iloc[0]) == pytest.approx(4 / 6)


def test_field_anchored_s_product_rejects_stable_wrong_glc():
    from sgfe_lc.merge import assign_roles_and_weights, field_anchored_s_product

    df = pd.DataFrame({
        "year": [2013, 2013, 2014, 2018],
        "class_id": [3, 3, 7, 1],
        "glc_l1_2013": [5, 3, 7, 1],
        "glc_l1_2014": [5, 3, 7, 1],
        "glc_l1_2017": [5, 4, 7, 1],
        "glc_l1_2018": [5, 3, 7, 1],
        "S_change": [1.0, 1.0, 0.2, 1.0],
        "S_pheno": [1.0, 1.0, 0.2, 1.0],
    })
    s = field_anchored_s_product(df)
    assert float(s.iloc[0]) == 0.0
    assert float(s.iloc[1]) == 1.0
    assert float(s.iloc[2]) == 1.0
    out = assign_roles_and_weights(df)
    assert float(out.loc[0, "w_rel"]) == 0.0
    assert float(out.loc[1, "w_rel"]) == 1.0
    assert float(out.loc[2, "w_rel"]) == 1.0
    assert out.loc[2, "class_id"] == 7
    assert float(out.loc[3, "w_rel"]) == 1.0


def test_nested_cv_drops_all_years_from_validation_blocks():
    from sgfe_lc.eval_matrix import e6_nested_select

    rng = np.random.default_rng(0)
    rows = []
    for block in ("b0", "b1", "b2", "b3"):
        for year in (2017, 2018):
            for cls, n in ((1, 8), (3, 8), (5, 6)):
                for _ in range(n):
                    rows.append({"block_id": block, "year": year, "class_id": cls, "split": "traindev", "role": "exact_year"})
    for _ in range(8):
        rows.append({"block_id": "b0", "year": 2013, "class_id": 1, "split": "traindev", "role": "historical", "w_rel": 1.0})
    df = pd.DataFrame(rows)
    for i in range(8):
        df[f"A{i:02d}"] = rng.normal(size=len(df))
    out = e6_nested_select(df, estimator="logreg", seed=1, target_year=2018)
    assert "2017 exact-year" in (out.get("note") or "")
    assert out.get("table")


def test_classic_one_se_keeps_hard_when_ungated_below_floor():
    from sgfe_lc.eval_matrix import _one_se_recommend

    stats = {
        "none": {"mean": 0.3838, "se": 0.0041},
        "ungated": {"mean": 0.4133, "se": 0.0044},
        "hard_0.8": {"mean": 0.4213, "se": 0.0052},
        "hard_0.9": {"mean": 0.4268, "se": 0.0049},
    }
    rec, floor, in_set = _one_se_recommend(stats)
    assert rec == "hard_0.9"
    assert "ungated" not in in_set
    assert "hard_0.9" in in_set
    assert floor == pytest.approx(0.4219, abs=1e-3)

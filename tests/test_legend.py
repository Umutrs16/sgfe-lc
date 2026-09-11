from sgfe_lc.legend import DW_TO_L1, GLC_TO_L1, harmonize_label


def test_chinese_and_english_labels():
    assert harmonize_label("草地") == 3
    assert harmonize_label("Grassland") == 3
    assert harmonize_label("cropland") == 1
    assert harmonize_label("灌丛") == 4
    assert harmonize_label("bare") == 5


def test_glc_and_dw_maps():
    assert GLC_TO_L1[10] == 1
    assert GLC_TO_L1[130] == 3
    assert GLC_TO_L1[120] == 4
    assert GLC_TO_L1[200] == 5
    assert DW_TO_L1["crops"] == 1
    assert DW_TO_L1["shrub_and_scrub"] == 4

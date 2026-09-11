# SGFE-LC

Python and Google Earth Engine pipeline for year-matched feature extraction, locked spatial-block splits, few-shot and historical-label experiments, and 2018 land-cover mapping in arid Central Asia.

This repository contains **data-processing and evaluation code only**. Manuscript writers, journal-figure builders, and Word export scripts are not included.

## Field data

The original 902-site field dataset is Li et al. (2026), DOI [10.3974/geodb.2026.06.06.V1](https://doi.org/10.3974/geodb.2026.06.06.V1). Download the site metadata (not the photographs) into `data/raw/`, then run the audit script.

`data/processed/site_split_ids.csv` stores the quality-controlled site IDs, 100 km block IDs, train/dev/test assignment, and roles used by the locked analysis (seed `20260902`).

## Setup

```bash
pip install -e .
pytest -q
```

Earth Engine authentication is required for feature extraction and mapping. Set `project.gee_project` in `config/project.yaml` to a Cloud project you can access.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| Site QC and locked 100 km split | `scripts/00_audit_sites.py` | `data/processed/sites_harmonized.gpkg` |
| GEE connectivity | `scripts/01_smoke_gee.py` | probe samples |
| Export year-matched predictors | `scripts/02_export_gee_samples.py`, `scripts/02b_extract_year_2017.py` | CSVs under `data/processed/gee_exports/` |
| Assemble features and run experiments | `scripts/03_run_redesign.py` | `sites_exact_year.gpkg`, `results/experiment_results_v2.json` |
| Track B RF protocol check | `scripts/20_rerun_rf_selector_terrain.py`, `scripts/21_gee_smile_protocol_cv.py` | selector / Smile RF tables |
| 2018 mapping export | `scripts/04_export_regional_map.py` | GEE assets |
| Score the 2018 map at field sites | `scripts/13_score_gee_map.py` | site-level OA / Macro-F1 |

A Code Editor helper script is in `gee/SGFE_LC_CodeEditor.js`.

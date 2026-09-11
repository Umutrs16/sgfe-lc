from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .legend import LEVEL1


def _figdir() -> Path:
    cfg = load_config()
    d = resolve_path(cfg, "figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def fig1_site_map(gdf, path: Path | None = None):
    path = path or _figdir() / "fig1_sites.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    if "year" in gdf:
        sc = ax.scatter(gdf["longitude"], gdf["latitude"], c=gdf["year"], s=12, cmap="viridis", alpha=0.85)
        fig.colorbar(sc, ax=ax, label="Survey year")
    else:
        ax.scatter(gdf["longitude"], gdf["latitude"], s=12, c="#1b4d3e", alpha=0.85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Central Asia in-situ sites (Turkmenistan is an extrapolation domain)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig3_product_bars(e1: dict, path: Path | None = None):
    path = path or _figdir() / "fig3_product_benchmark.png"
    names, oa, f1 = [], [], []
    for k, r in e1.items():
        if not isinstance(r, dict) or "oa" not in r:
            continue
        names.append(k.replace("_l1_2018", ""))
        oa.append(r["oa"])
        f1.append(r["macro_f1"])
    if not names:
        return None
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, oa, 0.35, label="OA")
    ax.bar(x + 0.18, f1, 0.35, label="Macro-F1")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Exact-year 2018 product accuracy on field sites")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig4_label_efficiency(df: pd.DataFrame, path: Path | None = None):
    path = path or _figdir() / "fig4_label_efficiency.png"
    if df is None or df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, sub in df.groupby("group"):
        ax.plot(sub["frac"], sub["macro_f1"], marker="o", label=group)
        if "ci_lo" in sub and sub["ci_lo"].notna().any():
            ax.fill_between(sub["frac"], sub["ci_lo"], sub["ci_hi"], alpha=0.15)
    ax.set_xlabel("Training label fraction")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Label efficiency of foundation embeddings vs traditional features")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig5_loco_heatmap(df: pd.DataFrame, path: Path | None = None):
    path = path or _figdir() / "fig5_loco.png"
    if df is None or df.empty:
        return None
    pivot = df.pivot(index="group", columns="test_country", values="macro_f1")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="YlGn")
    ax.set_xticks(range(len(pivot.columns)), list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Macro-F1")
    ax.set_title("Leave-one-country-out generalization")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig6_ablation_forest(named_scores: dict, path: Path | None = None):
    path = path or _figdir() / "fig6_ablation.png"
    items = [(k, v) for k, v in named_scores.items() if isinstance(v, (int, float))]
    if not items:
        return None
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1.5))
    y = np.arange(len(labels))
    ax.scatter(vals, y, s=40)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Module ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def class_legend_note() -> str:
    return ", ".join(f"{k}={v}" for k, v in LEVEL1.items())

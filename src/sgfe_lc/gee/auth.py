from __future__ import annotations

import ee

from ..config import gee_project, load_config


def init_ee(project: str | None = None):
    cfg = load_config()
    project = project or gee_project(cfg)
    ee.Initialize(project=project)
    return ee

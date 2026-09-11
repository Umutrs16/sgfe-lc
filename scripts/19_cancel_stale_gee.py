"""Cancel PENDING/RUNNING GEE tasks whose description contains a stale protocol tag."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgfe_lc.gee.auth import init_ee


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ungated"
    ee = init_ee()
    n = 0
    for op in ee.data.listOperations():
        meta = op.get("metadata") or {}
        desc = str(meta.get("description") or "")
        state = meta.get("state")
        if tag in desc and state in {"PENDING", "RUNNING"}:
            ee.data.cancelOperation(op["name"])
            print("cancelled", desc, state)
            n += 1
    print("cancelled", n, "tasks matching", tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

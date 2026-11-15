"""Render docs/architecture/opm/*.dot to SVG (requires Graphviz `dot` on PATH).

    .venv/bin/python scripts/render_opm.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

OPM_DIR = Path(__file__).resolve().parents[1] / "docs" / "architecture" / "opm"


def main() -> int:
    dot = shutil.which("dot")
    if dot is None:
        print("Graphviz 'dot' not found on PATH (brew install graphviz)", file=sys.stderr)
        return 1
    out_dir = OPM_DIR / "svg"
    out_dir.mkdir(exist_ok=True)
    failures = 0
    for src in sorted(OPM_DIR.glob("*.dot")):
        target = out_dir / f"{src.stem}.svg"
        result = subprocess.run([dot, "-Tsvg", str(src), "-o", str(target)], capture_output=True, text=True, timeout=60)
        print(f"[{'ok' if result.returncode == 0 else 'FAIL'}] {src.name} -> svg/{target.name}")
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        failures += result.returncode != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

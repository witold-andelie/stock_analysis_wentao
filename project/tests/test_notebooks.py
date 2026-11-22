"""Offline notebook generation, validation and read-only German evidence."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"
GERMAN = "06_german_extension.ipynb"


@pytest.fixture
def selected_build(tmp_path):
    before = {p.name: p.stat().st_mtime_ns for p in NOTEBOOKS.glob("*.ipynb")}
    output = tmp_path / "notebooks"
    env = {**os.environ, "NB_OUT": str(output), "PYTHONPATH": "src"}
    subprocess.run([sys.executable, "scripts/build_notebooks.py", "06"], cwd=PROJECT,
                   env=env, check=True, capture_output=True, text=True)
    after = {p.name: p.stat().st_mtime_ns for p in NOTEBOOKS.glob("*.ipynb")}
    return output, before, after


def test_build_selected_notebook_only(selected_build):
    output, before, after = selected_build
    assert sorted(p.name for p in output.iterdir()) == [GERMAN]
    assert before == after


def test_notebook_06_valid_and_error_free():
    """The committed notebook is executed on the coordinator's German warehouse (like 00-05): outputs are allowed,
    error outputs are not."""
    notebook = nbformat.read(NOTEBOOKS / GERMAN, as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert cells
    for cell in cells:
        assert not [out for out in cell.outputs if out.get("output_type") == "error"]
        ast.parse(cell.source)

    first = ast.parse(cells[0].source)
    assignments = [node for node in ast.walk(first) if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Constant) and node.value.value == "de"
                   and any(ast.unparse(target) == "os.environ['DRIFTREV_MARKET']" for target in node.targets)]
    imports = [node for node in ast.walk(first)
               if (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "driftrev")
               or (isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "driftrev" for alias in node.names))]
    assert len(assignments) == 1
    assert imports
    assert assignments[0].lineno < min(node.lineno for node in imports)


def test_notebook_06_is_read_only():
    notebook = nbformat.read(NOTEBOOKS / GERMAN, as_version=4)
    forbidden = ("update_", "ingest", "run_pipeline", "research_de", "train_and_backtest", "replace_table", "upsert", "save_")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for name in forbidden:
                assert name not in cell.source, name


def test_existing_notebooks_untouched_by_selection(selected_build):
    _, before, after = selected_build
    for prefix in ("00", "01", "02", "03", "04", "05"):
        names = [name for name in before if name.startswith(prefix)]
        assert len(names) == 1, prefix
        for name in names:
            assert after[name] == before[name], name

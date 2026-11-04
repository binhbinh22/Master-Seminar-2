"""Execute all EDA notebooks and export every Matplotlib figure as a 300-DPI PNG."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def export_notebook(path: Path, output: Path, dpi: int, save_notebook: bool = False) -> dict:
    notebook = nbformat.read(path, as_version=4)
    original_sources = [cell.source for cell in notebook.cells]
    destination = output / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    code_cells = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        code_cells += 1
        cell.source = f"_eda_source_cell = {index}\n" + cell.source + "\n_eda_capture_figures()\n"
        cell.outputs = []
        cell.execution_count = None

    setup = f"""
import json as _eda_json
from pathlib import Path as _EDAPath
from weakref import WeakSet as _EDAWeakSet
import matplotlib.pyplot as _eda_plt
_eda_output = _EDAPath({str(destination)!r})
_eda_source_cell = -1
_eda_saved = _EDAWeakSet()
_eda_figures = []
_eda_counts = {{}}
_eda_original_show = _eda_plt.show
def _eda_capture_figures():
    for number in _eda_plt.get_fignums():
        figure = _eda_plt.figure(number)
        if figure in _eda_saved:
            continue
        _eda_counts[_eda_source_cell] = _eda_counts.get(_eda_source_cell, 0) + 1
        filename = f"cell_{{_eda_source_cell + 1:03d}}_figure_{{_eda_counts[_eda_source_cell]:02d}}.png"
        figure.savefig(_eda_output / filename, dpi={dpi}, bbox_inches='tight', facecolor='white')
        _eda_saved.add(figure)
        _eda_figures.append({{'file': filename, 'source_cell_index': _eda_source_cell,
                             'titles': [ax.get_title() for ax in figure.axes if ax.get_title()]}})
def _eda_show(*args, **kwargs):
    _eda_capture_figures()
    return _eda_original_show(*args, **kwargs)
_eda_plt.show = _eda_show
"""
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    notebook.cells.append(nbformat.v4.new_code_cell(
        "_eda_capture_figures()\n"
        "print('EDA_EXPORT_MANIFEST=' + _eda_json.dumps(_eda_figures, ensure_ascii=False))"
    ))
    client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    last_outputs = notebook.cells[-1].outputs
    marker = "EDA_EXPORT_MANIFEST="
    manifest = next(line[len(marker):] for item in last_outputs
                    if item.output_type == "stream" for line in item.text.splitlines()
                    if line.startswith(marker))
    figures = json.loads(manifest)
    for figure in figures:
        if not (destination / figure["file"]).is_file():
            raise RuntimeError(f"Figure export missing: {figure['file']}")
    if save_notebook:
        notebook.cells = notebook.cells[1:-1]
        for cell, original in zip(notebook.cells, original_sources):
            cell.source = original
        nbformat.write(notebook, path)
    return {"notebook": str(path.relative_to(ROOT)), "output_directory": path.stem,
            "executed_code_cells": code_cells, "figure_count": len(figures), "dpi": dpi,
            "source_sha256": hashlib.sha256("\n".join(original_sources).encode()).hexdigest(),
            "execution_adjustments": [], "figures": figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/EDA")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save-notebooks", action="store_true", help="Keep executed tables and plots in the source notebooks")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    output = args.output_dir.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        parser.error("Choose a dedicated figure directory, not the repository or its parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".eda-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "figures"
        staging.mkdir()
        records = []
        for path in sorted((ROOT / "notebooks").glob("EDA*.ipynb")):
            print(f"Running {path.name} ...", flush=True)
            record = export_notebook(path, staging, args.dpi, args.save_notebooks)
            records.append(record)
            print(f"  {record['executed_code_cells']} code cells; {record['figure_count']} PNG figures", flush=True)
        if not records:
            raise RuntimeError("No EDA notebooks found")
        manifest = {"notebooks": records, "total_figures": sum(r["figure_count"] for r in records),
                    "shared_utilities_sha256": hashlib.sha256((ROOT / "notebooks/eda_utils.py").read_bytes()).hexdigest(),
                    "raw_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted((ROOT / "data/raw/nlbse2023").glob("*.csv"))},
                    "note": "Notebooks executed without source rewrites. Cell indices are zero-based; filenames use one-based cell numbers."}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        # Replace the previous generated figure set only after every notebook succeeds.
        if output.exists():
            if not (output / "manifest.json").is_file():
                raise ValueError(f"Refusing to replace a directory without an EDA export manifest: {output}")
            shutil.rmtree(output)
        staging.rename(output)
    print(f"Exported {manifest['total_figures']} figures to {output}")


if __name__ == "__main__":
    main()

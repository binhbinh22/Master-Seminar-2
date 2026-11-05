# Validation results

Date: 5 November 2026. Scope: report artifacts and archived evidence, not a new research-training run.

## Document build

- **Succeeded:** MiKTeX XeLaTeX + BibTeX, with repeated XeLaTeX passes to settle contents/cross-references. XeLaTeX preserves Vietnamese team names from the existing manuscript.
- Final report: **32 PDF pages**, including cover, contents, 10 main sections, references and appendices. The page count changed from the initial 33-page build after switching the text font to support the restored cover metadata.
- **9 vector figures/diagrams**, each also supplied as PNG; **17 numbered tables** (13 generated table files, four tables embedded in section sources).
- Final log: no LaTeX/package warnings, undefined citations/references, missing figures/characters, or overfull boxes. A few underfull-line notices are cosmetic; rendered text remains within margins.
- PDF text bounds and nonempty-page checks passed. All input paths and diagram/figure references exist.
- Every page was rendered into contact sheets and visually inspected; important data/results/architecture pages were additionally inspected at higher resolution. Diagram text was fitted to its boxes and the CV/test legend moved outside bars. No clipping, broken table, or overlapping diagram text remained in the reviewed final assets.
- `pdf_validation.json` records automated outcomes. `build/rendered/` contains visual-review intermediates.
- The updated Overleaf ZIP was extracted into a fresh temporary directory and compiled independently with XeLaTeX/BibTeX: **32 pages**, successful. It contains all required section, table, bibliography and figure files without depending on original repository paths for document compilation.
- Institution, school, instructor, course, academic year and all five Vietnamese names/student IDs were checked against `docs/report/report.tex`, verified in extracted PDF text, and visually reviewed on the rendered cover. University of L'Aquila is the instructor's affiliation; the report's institution is Hanoi University of Science and Technology.

## Numerical and dataset verification

- All 6,738 sentence targets agree with a pivot of the original category-wise CSVs.
- Final splits reproduce the same sentence population and targets as screening data.
- Every sentence appears in CV validation once; exact normalized-text groups do not overlap within a CV split or between final splits.
- All 69 stored CV mean/sample-SD records recomputed from folds.
- All 30 final micro-/macro-F1 records checked against stored per-label evidence.
- All neural summary CSV micro-F1 values checked against source records; no discrepancies.
- All six staged datasets independently regenerated in temporary storage from the checked-in raw CSVs using the actual preparation/loader functions. Local byte fingerprints, population, sentence order, labels and folds matched exactly. `split_verification.json` records these checks. Stored historical run fingerprints were also checked for agreement across comparable runs; Windows file-byte conversion is not silently treated as a match to historical byte hashes.
- Missing predictions/checkpoints, incomplete final coverage and the selection-manifest mismatch were explicitly preserved as findings.

## Existing software tests

Command executed from repository root:

```powershell
$env:USE_TF='0'
$env:MODEL_ZOO_PROGRESS='off'
python -m unittest discover -s tests -p test_two_stage.py -v
```

Outcome: **2 passed, 1 error**. `test_actual_data_matches_raw_and_reproduces` fails because its hard-coded `data/raw/` does not exist in the checkout. `test_rejects_overlap_and_overwrite` and the offline staged-training smoke test passed. The latter trains tiny synthetic fixtures with mocked backbones, not the research models. Original tests and datasets were not modified. The independent full-data regeneration check used the actual available `data/` CSVs and passed.

The full repository test suite was not claimed as passing or executed. Research models were not retrained, EDA notebooks were not rerun, and original checkpoints were not replayed.

## Remaining limitations

The PDF is complete and reviewable, but full course/submission readiness is **not certified**. Switchable memory/file persistence is missing. Independent method-selection evaluation, unbiased outer checkpoint scoring, repeated full-protocol seeds, significance analysis, historical environment lock, sentence predictions and complete final Transformer coverage remain unavailable. See `implementation_gaps.md` and the report's threats-to-validity section.

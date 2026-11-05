# Validation results

Revision date: 5 November 2026. This record distinguishes archived scientific experiments from revision-time software checks and report generation.

## Document

- XeLaTeX + BibTeX build succeeds: **35 pages**, **14 figures**, **20 numbered tables**, **19 references**.
- The complete institutional/instructor/team cover metadata matches the old manuscript, including Vietnamese names and student IDs. The compiler preserves their diacritics.
- The cover contains no commit hash, GitHub link or technical readiness note; reproducibility details are in Section 9 and the appendix.
- Final log has no unresolved citations/references, missing characters/figures, LaTeX/package warnings or overfull boxes. Cosmetic underfull-line notices are checked in rendered output.
- Every PDF page is rendered and reviewed for layout. Text bounds and nonempty-page checks are automated in `scripts/validate_pdf.py`; outcomes are stored in `pdf_validation.json`.
- The PDF is tracked in Git. The generated Overleaf ZIP is self-contained and checked through a fresh-directory XeLaTeX/BibTeX build.

## Numerical and content checks

- All 6,738 sentence targets match raw category-wise CSVs; final populations/targets match screening populations.
- No exact normalized-text group overlaps within CV or final splits; CV validates each sentence once.
- All 69 CV summaries and 30 final F1 records are checked arithmetically. Original scientific metric files are unchanged.
- All six prepared datasets regenerate with equal bytes, order, labels, folds and fingerprints.
- Three EDA figures use actual prepared targets/text, two learning figures use stored neural histories/ML refit records, and four illustrative comments retain actual sentence IDs and gold labels. They are not presented as model-error examples.
- Java rational counts are checked directly: E1 TP=3/28, FP=2, FN=25; E17 TP=12/28, FP=2, FN=16.
- Prose Random Forest Python F1 is corrected to 0.5882. Displayed-value percentage-point arithmetic gives Pharo E3−E1 = 1.08 and E15−E11 = 1.51; this rounding convention is stated in the report.
- E4's duplicate screening rows are replaced by notes; all original E4 result files remain preserved.
- Official benchmark context is sourced to NLBSE'23 organizers' Table III: RF 0.309, submitted LR 0.547, CodeT5 0.657, STACC 0.744 average category F1. It is explicitly separated from our per-language micro-F1 protocol.

## Software

```powershell
$env:USE_TF='0'
$env:MODEL_ZOO_PROGRESS='off'
python -m unittest discover -s tests -v
```

**35 tests pass.** Six new storage tests cover read/write contracts, snapshot isolation, missing paths/backend validation, shared corruption checks, ML consumption, and equivalence across all six actual datasets. Existing offline tests exercise small mocked neural models; these are software fixtures, not new scientific experiments.

The earlier actual-data test assumed an absent `data/raw/` download directory. It now selects that source if present and otherwise uses the checked-in long CSVs under `data/`, preserving the original equivalence assertions. Temporary actual-data regeneration passes independently.

Dataset input/manifests can switch between FileStore and MemoryStore through the same loader and `--storage` CLI option. Run plots, tokenizers, metrics/checkpoint exports remain filesystem outputs. This scope is documented without treating the new adapter layer as the implementation used by archived training runs.

## Scientific scope

No research model or original EDA notebook was retrained/re-executed for this report revision. Report assets are recomputed from existing evidence. Selection dependence, checkpoint bias, limited seeds, missing original predictions/weights and incomplete final coverage are collected in Section 8. The report makes no new significance or competition-superiority claim. A Vietnamese edition was explicitly skipped at the user's request.

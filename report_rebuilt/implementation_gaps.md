# Implementation and evidence gaps

This checklist distinguishes code, documentation, experiments, and artifacts. It is not a record of completed improvements. Course requirements follow the supplied prompt; no official rubric was available locally.

## Missing implementation

- [ ] **Course requirement: interchangeable persistence.** File readers/writers and in-memory bundle/tensor objects exist, but there is no injected interface with memory and filesystem adapters, backend selection, and unchanged business logic. Implementing this requires code work beyond report authoring.
- [ ] **Neural row-level prediction export.** The staged neural runner produces aggregate metrics, not a final per-sentence decision file. Add sentence ID, split, ordered labels, truth, probabilities, hard predictions, thresholds and checkpoint identity.
- [ ] **Documented checkpoint evaluation/inference entry point.** No verified standalone final checkpoint replay CLI is supplied. Current evaluation is integrated into training runners.

## Missing documentation / distribution

- [ ] Original training Python/package lock, GPU model, driver/CUDA versions and exact execution environment.
- [ ] Explanation and provenance for choosing the common E1/E3/E9/E15/E17 final subset instead of the language-specific top four in the saved selection manifest.
- [ ] Distributed checkpoints, joblib models/vectorizers, or links with checksums. Saving is implemented, but these files are not in the current snapshot.
- [ ] Classical prediction CSVs expected by the runner's output contract. No such files are in current results.
- [ ] Verified annotation instructions and agreement information. Label names/targets are available; descriptive glosses do not replace an annotation guide.
- [ ] An official course template/rubric beyond the existing manuscript layout. Institution, instructor, course, academic year and all five author names/student IDs are available in `docs/report/report.tex` and have been restored on the rebuilt cover.

## Missing experiments

- [ ] Untouched outer holdout or nested/project-grouped evaluation with all model selection confined to training.
- [ ] Neural CV with independent inner checkpoint selection, or a fixed schedule selected without consulting the reported outer fold.
- [ ] Repeated complete protocol seeds; fold seeds 42–46 are not repeated independent protocol seeds.
- [ ] Appropriate paired uncertainty/significance analysis accounting for selection and multiple comparisons.
- [ ] Final evaluation of the remaining twelve Transformer configurations per language, if scientifically justified after freezing a protocol. Do not simply run more comparisons on the same observed test.
- [ ] Final threshold-tuning experiment (E4); E4 screening uses 0.5 and equals E1.
- [ ] Qualitative FP/FN/rare-label analysis from actual sentence predictions. It cannot be completed from aggregate metrics.
- [ ] Controlled backbone/regularization/optimization ablations and hardware-normalized cost/latency measurements.

## Unverified functionality / claims

- [ ] End-to-end Colab setup and execution. No dedicated setup/record found.
- [ ] Full JupyterLab re-execution in a clean environment. Saved EDA outputs exist, but were not regenerated in this report task.
- [ ] Generalization to unseen projects/classes, developer productivity benefits, or superiority over official competition baselines.
- [ ] Near-duplicate mitigation beyond exact whitespace-normalized text grouping.

## Recommended improvements (proposed only)

1. Introduce a narrow data/artifact store interface and equivalent memory/file adapters with contract tests.
2. Preserve immutable sentence-level predictions, checkpoints, split IDs, code revision, package lock and device metadata for every run.
3. Freeze candidate configurations, then evaluate a fresh outer dataset partition, preferably by project/class, with nested selection where required.
4. Run a prespecified seed set and report uncertainty, label supports, and precision/recall tradeoffs.
5. Tune thresholds on validation alone and tie evaluation criteria to actual downstream error costs.
6. Replace stale manuscript/README statements about missing ML results after project owners review the new report. Original files were preserved here.

**Readiness:** the reconstructed report can be compiled and reviewed, but it does not establish full course compliance or submission readiness while the mandatory persistence requirement and critical evidence gaps remain unresolved.

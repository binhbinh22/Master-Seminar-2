# Stage 2: final training results

Stage-1 CV used all data, including future stage-2 test rows; final test is exploratory, not independent of method selection.

Rows are ordered by validation micro F1 within each language/data fingerprint. Test is reporting only.
The saved checkpoint is selected on validation at the baseline threshold; E4 subsequently tunes thresholds on validation.

| Language | Data fingerprint | Model | Seed | Best epoch | Val micro F1 | Test micro F1 | Test macro F1 |
|---|---|---|---:|---:|---:|---:|---:|
| java | 508818d4c42f | E15 | 42 | 3 | 0.6402 | 0.6914 | 0.7284 |
| java | 508818d4c42f | E17 | 42 | 3 | 0.6338 | 0.7034 | 0.7342 |
| pharo | e585d5a37586 | E15 | 42 | 6 | 0.7566 | 0.7050 | 0.6194 |
| pharo | e585d5a37586 | E17 | 42 | 10 | 0.7166 | 0.7080 | 0.6283 |
| python | a17f2a4465ac | E15 | 42 | 9 | 0.6541 | 0.6541 | 0.5965 |
| python | a17f2a4465ac | E17 | 42 | 4 | 0.6133 | 0.6248 | 0.5669 |

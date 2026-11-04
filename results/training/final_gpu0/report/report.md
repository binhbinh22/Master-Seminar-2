# Stage 2: final training results

Stage-1 CV used all data, including future stage-2 test rows; final test is exploratory, not independent of method selection.

Rows are ordered by validation micro F1 within each language/data fingerprint. Test is reporting only.
The saved checkpoint is selected on validation at the baseline threshold; E4 subsequently tunes thresholds on validation.

| Language | Data fingerprint | Model | Seed | Best epoch | Val micro F1 | Test micro F1 | Test macro F1 |
|---|---|---|---:|---:|---:|---:|---:|
| java | 508818d4c42f | E3 | 42 | 5 | 0.6055 | 0.6839 | 0.7111 |
| java | 508818d4c42f | E9 | 42 | 8 | 0.5965 | 0.6762 | 0.7371 |
| java | 508818d4c42f | E1 | 42 | 6 | 0.5761 | 0.6947 | 0.6946 |
| pharo | e585d5a37586 | E1 | 42 | 8 | 0.7735 | 0.7371 | 0.6720 |
| pharo | e585d5a37586 | E3 | 42 | 5 | 0.7680 | 0.6919 | 0.6015 |
| pharo | e585d5a37586 | E9 | 42 | 7 | 0.7363 | 0.7157 | 0.6058 |
| python | a17f2a4465ac | E1 | 42 | 9 | 0.6397 | 0.6074 | 0.5547 |
| python | a17f2a4465ac | E3 | 42 | 10 | 0.6362 | 0.5900 | 0.5496 |
| python | a17f2a4465ac | E9 | 42 | 6 | 0.6302 | 0.6285 | 0.5634 |

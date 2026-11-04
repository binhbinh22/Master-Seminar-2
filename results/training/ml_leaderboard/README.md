# Classical ML leaderboard

Sinh tự động bởi `python -m src.evaluation.ml_leaderboard`. Xếp hạng CV theo `cv_micro_f1_mean`; xếp hạng test theo `test_micro_f1`.

## Giai đoạn 1 — CV (100% dữ liệu, sàng lọc phương pháp)

| Language | Method | CV micro-F1 (mean) | CV micro-F1 (std) | CV macro-F1 | CV Hamming loss |
|---|---|---|---|---|---|
| java | ovr_logreg | 0.5972 | 0.0566 | 0.639 | 0.1315 |
| java | ovr_sgd | 0.5932 | 0.0556 | 0.6363 | 0.1354 |
| java | ovr_linear_svc | 0.5811 | 0.0577 | 0.6241 | 0.1354 |
| java | classifier_chain_logreg | 0.5789 | 0.0675 | 0.6282 | 0.1387 |
| java | ovr_random_forest | 0.5726 | 0.0405 | 0.6146 | 0.1417 |
| pharo | ovr_sgd | 0.6486 | 0.0188 | 0.5604 | 0.1149 |
| pharo | ovr_random_forest | 0.6458 | 0.0226 | 0.5196 | 0.1105 |
| pharo | ovr_logreg | 0.6438 | 0.0223 | 0.5545 | 0.1164 |
| pharo | ovr_linear_svc | 0.6372 | 0.021 | 0.5474 | 0.1149 |
| pharo | classifier_chain_logreg | 0.6209 | 0.0247 | 0.5279 | 0.1262 |
| python | ovr_random_forest | 0.5882 | 0.0126 | 0.5428 | 0.176 |
| python | ovr_logreg | 0.5833 | 0.0126 | 0.547 | 0.1885 |
| python | ovr_sgd | 0.5811 | 0.0093 | 0.5459 | 0.1925 |
| python | ovr_linear_svc | 0.573 | 0.0095 | 0.5381 | 0.1915 |
| python | classifier_chain_logreg | 0.5623 | 0.0193 | 0.5233 | 0.1951 |

## Giai đoạn 2 — Validation / Test (80/10/10, held-out)

| Language | Method | Validation micro-F1 | Test micro-F1 | Test macro-F1 | Test Hamming loss |
|---|---|---|---|---|---|
| java | classifier_chain_logreg | 0.6433 | 0.6964 | 0.747 | 0.1004 |
| java | ovr_sgd | 0.6041 | 0.695 | 0.7441 | 0.1015 |
| java | ovr_logreg | 0.6046 | 0.6871 | 0.7344 | 0.1027 |
| java | ovr_linear_svc | 0.5974 | 0.6838 | 0.7335 | 0.1015 |
| java | ovr_random_forest | 0.5438 | 0.636 | 0.6495 | 0.1216 |
| pharo | ovr_random_forest | 0.6011 | 0.6447 | 0.5607 | 0.1136 |
| pharo | ovr_sgd | 0.6228 | 0.642 | 0.5788 | 0.1177 |
| pharo | ovr_logreg | 0.6294 | 0.64 | 0.5653 | 0.1169 |
| pharo | ovr_linear_svc | 0.628 | 0.6308 | 0.5601 | 0.1169 |
| pharo | classifier_chain_logreg | 0.599 | 0.6161 | 0.5197 | 0.1274 |
| python | ovr_linear_svc | 0.5486 | 0.5704 | 0.5295 | 0.1906 |
| python | ovr_sgd | 0.5685 | 0.569 | 0.5268 | 0.1953 |
| python | ovr_logreg | 0.5654 | 0.5675 | 0.5244 | 0.1953 |
| python | classifier_chain_logreg | 0.5887 | 0.5577 | 0.5275 | 0.1945 |
| python | ovr_random_forest | 0.5772 | 0.5421 | 0.5019 | 0.1953 |


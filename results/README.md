# Kết quả E1–E12 đã giữ lại

Bộ kết quả chính gồm **36 run hoàn chỉnh: 12 exp × 3 ngôn ngữ**, mỗi cặp một run.
Các run dùng budget 10 epoch với early stopping; E9 giữ head v2. Không chọn run theo điểm test cao nhất.
Đã kiểm tra data fingerprint, thứ tự nhãn, 5 fold, test metrics, đồ thị và khả năng nạp checkpoint.

| Exp | Python | Java | Pharo |
|---|---|---|---|
| E1 | [python](training/epochs10_gpu0/exp1/python/metrics.json) | [java](training/epochs10_gpu0/exp1/java/metrics.json) | [pharo](training/epochs10_gpu0/exp1/pharo/metrics.json) |
| E2 | [python](training/epochs10_gpu0/exp2/python/metrics.json) | [java](training/epochs10_gpu0/exp2/java/metrics.json) | [pharo](training/epochs10_gpu0/exp2/pharo/metrics.json) |
| E3 | [python](training/epochs10_gpu0/exp3/python/metrics.json) | [java](training/epochs10_gpu0/exp3/java/metrics.json) | [pharo](training/epochs10_gpu0/exp3/pharo/metrics.json) |
| E4 | [python](training/epochs10_gpu1/exp4/python/metrics.json) | [java](training/epochs10_gpu1/exp4/java/metrics.json) | [pharo](training/epochs10_gpu1/exp4/pharo/metrics.json) |
| E5 | [python](training/epochs10_gpu1/exp5/python/metrics.json) | [java](training/epochs10_gpu1/exp5/java/metrics.json) | [pharo](training/epochs10_gpu1/exp5/pharo/metrics.json) |
| E6 | [python](training/epochs10_gpu1/exp6/python/metrics.json) | [java](training/epochs10_gpu1/exp6/java/metrics.json) | [pharo](training/epochs10_gpu1/exp6/pharo/metrics.json) |
| E7 | [python](training/20261029-140821-309637/exp7/python/metrics.json) | [java](training/20261029-140821-309637/exp7/java/metrics.json) | [pharo](training/20261029-140821-309637/exp7/pharo/metrics.json) |
| E8 | [python](training/20261029-140821-309637/exp8/python/metrics.json) | [java](training/20261029-140821-309637/exp8/java/metrics.json) | [pharo](training/20261029-140821-309637/exp8/pharo/metrics.json) |
| E9 | [python](training/20261029-162942-782531/exp9/python/metrics.json) | [java](training/20261029-162942-782531/exp9/java/metrics.json) | [pharo](training/20261029-162942-782531/exp9/pharo/metrics.json) |
| E10 | [python](training/20261029-140844-311400/exp10/python/metrics.json) | [java](training/20261029-140844-311400/exp10/java/metrics.json) | [pharo](training/20261029-140844-311400/exp10/pharo/metrics.json) |
| E11 | [python](training/20261029-140844-311400/exp11/python/metrics.json) | [java](training/20261029-140844-311400/exp11/java/metrics.json) | [pharo](training/20261029-140844-311400/exp11/pharo/metrics.json) |
| E12 | [python](training/20261029-140844-311400/exp12/python/metrics.json) | [java](training/20261029-140844-311400/exp12/java/metrics.json) | [pharo](training/20261029-140844-311400/exp12/pharo/metrics.json) |

## Báo cáo tổng hợp

- [Báo cáo model zoo](model_zoo/report.md)
- [CSV metric tổng hợp](model_zoo/leaderboard.csv)
- [CSV theo nhãn](model_zoo/per_label.csv)
- [Nhật ký dọn dẹp và đường dẫn đã giữ/xóa](training/CLEANUP_MANIFEST.json)

Đã loại E9 head v1 bị collapse, phần E9–Java không khởi chạy được và một lượt E9–Python dở dang.
Giữ nguyên code/config E1–E12 và kết quả E9 v2 của cả ba ngôn ngữ.
`diagnostics/` và `epoch_review/` là tài liệu kiểm tra/phân tích riêng, không được tính là run CV hoàn chỉnh.

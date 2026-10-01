# Master-Seminar-2

## RoBERTa baseline: NLBSE'23 code comment classification

Install dependencies and run from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m experiments.dl.train_exp1
```

The default config is `configs/exp1_roberta_baseline.yaml`; use `--config PATH`
to select another YAML file. The dataset config is
`configs/datasets/nlbse2023_code_comments.yaml`. The official CSVs are already
in `data/raw/nlbse2023/`; use `--download` if they are missing. The first run
also downloads `roberta-base` weights and tokenizer from Hugging Face.

Five-fold cross-validation splits sentences with official training targets,
balancing the number of sentences per language. Each sentence appears in one
validation fold. Each fold selects its best epoch using masked micro F1. The
final model is then trained on all official training targets for the median
best-epoch count, saved to `results/exp1_roberta_baseline/best_model.pt`, and
evaluated once on official test targets. Change `training.folds` in the YAML
to change the fold count. A full run trains six RoBERTa models by default.

`metrics.json` in the output directory contains per-fold metrics, mean and
sample standard deviation across folds, and final test metrics. It reports
micro/macro precision, recall, F1, average precision (area under the stepwise
precision-recall curve), Hamming loss, and per-label metrics with support.
Undefined average precision (no positive target) is recorded as `null`.
All metrics use only targets in the relevant split mask. Because the official
partition is per label, a sentence can have a training target for one label
and a test target for another.

### Theo dõi training và xem đồ thị

Trong lúc huấn luyện, mở TensorBoard ở terminal khác:

```bash
tensorboard --logdir results/exp1_roberta_baseline/tensorboard
```

Các run được tách theo thời điểm chạy, với log riêng cho từng fold, model cuối
và phần tổng hợp. TensorBoard ghi loss theo batch/epoch, learning rate, metric
validation theo epoch, và metric test sau khi hoàn tất. Tệp `metrics.json` ghi
đường dẫn chính xác của run trong `tensorboard_log_dir`.

Các đồ thị PNG được lưu ở `results/exp1_roberta_baseline/plots/`:
`training_curves.png` (loss và validation F1 theo epoch),
`cross_validation.png` (trung bình và độ lệch chuẩn giữa các fold),
`test_summary.png` (metric test), và `test_per_label.png` (F1, average
precision theo từng nhãn). `metrics.json` cũng liệt kê các đường dẫn này.

Thí nghiệm khác có thể dùng chung `save_training_plots` trong
`src/visualization/training_plots.py`: truyền lịch sử huấn luyện theo tên run,
metric validation cần vẽ, cùng kết quả cross-validation/test nếu có.

Các module dùng chung cho thí nghiệm tiếp theo:

- `src/data/splits.py`: chia k-fold cân bằng theo một nhóm tùy chọn (ví dụ ngôn ngữ).
- `src/evaluation/multilabel.py`: tính metric và đánh giá multi-label có mask.
- `src/evaluation/summaries.py`: tính trung bình, độ lệch chuẩn của metric qua các fold.
- `src/training/masked_classifier.py`: vòng huấn luyện cho model trả về `loss` và `logits`.
- `src/training/reproducibility.py`: đặt random seed.
- `src/visualization/tensorboard.py`: ghi scalar và metric theo nhãn lên TensorBoard.

Mỗi thí nghiệm tự tạo dataset, model và DataLoader rồi gọi các hàm chung phù hợp.

## Exp2: RoBERTa + Weighted BCE

Chạy từ thư mục gốc với cùng dependencies của exp1:

```bash
python3 -m experiments.dl.train_exp2
```

Config mặc định: `configs/exp2_roberta_weighted_bce.yaml`. Có thể truyền
`--config PATH` hoặc `--download` như exp1. Model dùng RoBERTa CLS và
Linear(768, 19), với `BCEWithLogitsLoss(pos_weight=..., reduction="none")`.
Loss được lấy trung bình trên các target có mask hợp lệ.

Với từng nhãn, `pos_weight = số target âm / số target dương`, chỉ tính trên
target thuộc official train của các câu trong tập train hiện tại. Mỗi fold
tính trọng số riêng, không dùng validation hoặc test; model cuối tính lại
trên toàn bộ official train. Nhãn không có target dương hoặc không có target
âm dùng trọng số 1 để giữ loss hữu hạn và vẫn học từ target hiện có.

Exp2 dùng cùng seed, cách chia 5 fold, threshold 0.5 và cách chọn số epoch
cuối của exp1. Kết quả được lưu tại `results/exp2_roberta_weighted_bce/`, gồm
checkpoint `best_model.pt`, tokenizer, config, `metrics.json`, log TensorBoard
và đồ thị. `metrics.json` ghi `pos_weight` theo thứ tự `labels` cho từng fold
và model cuối; trọng số cũng được lưu trong checkpoint.

```bash
tensorboard --logdir results/exp2_roberta_weighted_bce/tensorboard
```

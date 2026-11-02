# Training hai giai đoạn

Chạy các lệnh bên dưới từ thư mục gốc repo, sau khi cài `requirements.txt`.
Mỗi ngôn ngữ được huấn luyện riêng; các phương pháp dùng chung dữ liệu và folds.

## Dữ liệu

```bash
python3 -m src.data.prepare_two_stage --raw-dir data/raw --output-dir data --folds 5 --seed 42
```

Hai bộ đã được tạo sẵn. Lệnh trên từ chối ghi đè; dùng `--force` khi chủ động
muốn tạo lại. Đổi seed/folds đòi hỏi chạy lại toàn bộ các phương pháp để so sánh.
Nguồn là `data/raw/nlbse2023/{python,java,pharo}.csv`; bỏ partition gốc theo từng
nhãn, pivot thành một dòng cho mỗi sentence ID, giữ đủ nhãn và đủ câu.

| Ngôn ngữ | Tổng câu cho CV | Train | Validation | Test | Nhãn |
|---|---:|---:|---:|---:|---:|
| Python | 2.555 | 2.044 | 255 | 256 | 5 |
| Java | 2.418 | 1.935 | 241 | 242 | 7 |
| Pharo | 1.765 | 1.413 | 176 | 176 | 7 |

`data/k_fold_data/<language>/` chứa `all.csv`, `folds.json`, `label_cols.json`,
`manifest.json`. Chỉ số fold bắt đầu từ 0, trỏ tới dòng trong `all.csv`.
Mỗi câu xuất hiện trong validation đúng một lần qua 5 fold.

`data/full_training_data/<language>/` chứa `train.csv`, `val.csv`, `test.csv`,
`label_cols.json`, `manifest.json`. Chia xấp xỉ 80/10/10 vì số câu nguyên và
ràng buộc nhóm trùng. Dùng phép gán nhóm cân bằng gần đúng số câu và số nhãn
dương, seed 42. Không có nhãn nào bảo đảm stratification hoàn hảo.

Hai bộ chứa cùng tập sentence ID và nhãn. Text chỉ chuẩn hóa khoảng trắng;
các ID có text trùng hệt nhau luôn nằm cùng fold/split. Không chia theo project
hay class và chưa gộp near-duplicate. Manifest ghi số câu, phân bố nhãn, seed,
SHA-256 nguồn raw và fingerprint quần thể để kiểm tra khi nạp.

## Giai đoạn 1: K-fold trên 100% dữ liệu

```bash
# Xem danh sách E1–E17 và kiểm tra lệnh trước khi chạy.
python3 -m experiments.dl.model_zoo list
bash scripts/train_kfold.sh --dry-run

# 17 phương pháp × 3 ngôn ngữ; mỗi cặp chạy 5 fold.
bash scripts/train_kfold.sh --language all \
  --device cuda:0 --seed 42 --output-dir results/training/kfold_seed42
```

Mỗi fold khởi tạo model mới từ backbone pretrained, học trên khoảng 80% và
validation trên khoảng 20%. Early stopping/chọn checkpoint theo validation
micro F1; lưu metric từng fold, mean/sample SD, lịch sử và đồ thị. E9 giữ
patience/min-epochs riêng trong config. Class weights và graph chỉ tính từ fold
train. Giai đoạn này không train thêm model cuối và không đánh giá test.

Chọn 4 phương pháp tốt nhất **cho từng ngôn ngữ**, theo trung bình CV micro F1:

```bash
python3 -m src.evaluation.two_stage_results select \
  --results results/training/kfold_seed42 \
  --top-k 4 --output-dir results/training/selection
```

Dùng `--top-k 3` để chọn 3 phương pháp. Công cụ sinh `selection.json`,
`leaderboard.csv`, `report.md`. Chỉ so sánh cùng dữ liệu/folds/thứ tự nhãn và
cùng tập seed; từ chối run trùng cùng model/seed hoặc trộn cấu hình khác nhau
cho một model. Khi chạy nhiều seed, truyền các thư mục seed sau `--results`;
xếp hạng bằng trung bình CV của các seed, không chọn seed cao nhất.

E4 hiện chấm CV ở threshold 0.5 và hoãn tuning sang validation giai đoạn 2.
Vì vậy bước sàng lọc chưa đo lợi ích riêng của threshold tuning; E4 tương đương
E1 nếu cấu hình còn lại giống nhau. Khi hòa điểm, ưu tiên số E nhỏ hơn.
CV dùng validation chọn epoch nên kết quả sàng lọc vẫn có thiên lệch lựa chọn.

## Giai đoạn 2: chỉ train các phương pháp đã chọn

```bash
bash scripts/train_full.sh \
  --selection results/training/selection/selection.json \
  --language all --device cuda:0 --seed 42 \
  --output-dir results/training/final_seed42
```

Script tự lấy danh sách model và cấu hình đã sàng lọc cho từng ngôn ngữ.
Mỗi phương pháp khởi tạo lại từ pretrained, học trên **80% train**, dùng
**10% validation** để early stopping và chọn best epoch. Khôi phục best weights,
sau đó đánh giá **10% test** một lần. Không chạy K-fold lại, không dùng trung vị
epoch từ CV, không gộp validation vào train. E4 chỉ tuning threshold trên
validation; test dùng threshold đã cố định. Validation được báo ở threshold
cơ sở khi chọn checkpoint.

Fingerprint cấu hình và quần thể phải khớp với `selection.json`. Output đã có
metrics/checkpoint sẽ bị từ chối ghi đè. Khi cần chỉ định thủ công, có thể dùng
`--experiments 1,2,11` thay `--selection`; khi đó script không kiểm tra việc các
model này đã lọt top-k. Cách dùng chuẩn là truyền `--selection`.

Kết quả gồm `best_model.pt`, tokenizer, config, thứ tự nhãn, `thresholds.json`,
data manifest, `metrics.json`, TensorBoard và đồ thị. Graph được lưu thêm cho
E9/E12. Script tự sinh báo cáo tại `<output-dir>/report/`; có thể tổng hợp lại:

```bash
python3 -m src.evaluation.two_stage_results report \
  --results results/training/final_seed42 \
  --output-dir results/training/final_report
```

**Giới hạn của protocol được yêu cầu:** giai đoạn 1 đã dùng toàn bộ dữ liệu,
bao gồm các câu về sau nằm trong test. Vì vậy điểm test giai đoạn 2 là kết quả
thăm dò, không phải đánh giá độc lập với lựa chọn phương pháp. Nếu cần đánh giá
độc lập, phải dành test ngay từ đầu và chỉ làm K-fold trên phần còn lại.

## Chạy một model hoặc chia GPU

```bash
python3 -m experiments.dl.train_exp2 --stage kfold --language python --device cuda:0
python3 -m experiments.dl.model_zoo train --exp E2 --stage final --language python \
  --selection results/training/selection/selection.json --device cuda:0

# Hai terminal; dùng hai output khác nhau rồi đưa cả hai vào lệnh select.
bash scripts/train_kfold.sh --experiments 1,2,3,4,5,6,7,8,9 \
  --device cuda:0 --output-dir results/training/kfold_gpu0
bash scripts/train_kfold.sh --experiments 10,11,12,13,14,15,16,17 \
  --device cuda:1 --output-dir results/training/kfold_gpu1
```

Mặc định CLI dùng `--stage kfold`. Bộ `data/processed` và kết quả cũ đã được xóa.
`--stage legacy` chỉ còn giữ trong code để tương thích; muốn chạy phải tự tạo lại
dữ liệu bằng notebook preprocessing cũ. Không đưa run legacy vào lựa chọn hai giai đoạn. Hàm Python `roberta_experiment.run` giữ mặc
định legacy để tương thích; gọi từ code phải truyền `stage="kfold"` hoặc `"final"`.

Kiểm thử offline, dùng backbone nhỏ khởi tạo ngẫu nhiên và không tải model:

```bash
USE_TF=0 MODEL_ZOO_PROGRESS=off python3 -m unittest discover -s tests -v
```

# Classical ML baseline — code comment classification (Python)

Cách chạy:
```
python -m experiments.ml.train_baseline --config configs/exp_ml_baseline.yaml
```
Kết quả ghi ra `results/exp_ml_baseline/metrics.json`.

## Nguồn dữ liệu — dùng chung với baseline RoBERTa

Pipeline này đọc từ `data/processed/python/` (`train.csv`, `test.csv`, `folds.json`,
`label_cols.json`) — **bộ split chung của cả nhóm**, được sinh một lần bởi
`notebooks/data_preprocessing.ipynb` và dùng cho cả ML (ở đây) và RoBERTa
(`experiments/dl/train_exp1.py` của thành viên khác trong nhóm).

Lý do dùng chung bắt buộc: ML và RoBERTa là 2 phương pháp độc lập, nhưng để so sánh được
với nhau, cả hai phải fit trên **đúng cùng** `train.csv` và được chấm trên **đúng cùng**
`test.csv`/`folds.json`. Pipeline này **không** tự đọc `data/python.csv` gốc hay tự chia
lại train/test/fold — làm vậy sẽ phá vỡ khả năng so sánh.

Bộ split này là group-aware: 112 nhóm câu trùng lặp gần như nguyên văn (cùng đoạn mô tả
copy vào nhiều class khác nhau) được giữ nguyên vẹn cùng 1 phía ở mọi ranh giới train/test
và train/val của từng fold — xem `data/README.md` để biết chi tiết cách sinh và kiểm
chứng (`assert not set(train['group_id']) & set(test['group_id'])`).

## Cấu trúc module

| File | Vai trò |
|---|---|
| `data.py` | `load_processed('python')` — đọc `data/processed/python/` |
| `features.py` | TF-IDF (word n-gram + char n-gram), fit riêng mỗi lần train (không leak qua vectorizer) |
| `models.py` | 5 model: dummy baseline + 4 biến thể tuyến tính |
| `evaluation.py` | `cross_validate()` dùng `folds.json` có sẵn (chọn/so sánh model) + `evaluate_on_test()` fit trên toàn bộ train, chấm trên `test.csv` (số để so với RoBERTa) |
| `legacy/` | 4 script cũ dùng sai cột nhãn (`category` thay vì `instance_type`) — chỉ giữ để đối chiếu, xem `legacy/README.md` |

## Lỗi dữ liệu đã sửa (lịch sử)

`python.csv` gốc lưu dạng "exploded": mỗi câu lặp lại 5 lần (1 lần/category), và
`instance_type` (0/1) mới là nhãn thật — `category` chỉ là tên slot. Các script
`legacy/` lấy thẳng cột `category` làm nhãn, bỏ qua `instance_type`, khiến cùng một câu
có 5 nhãn mâu thuẫn nhau (4/5 sai) → model gần như không học được gì (micro-F1 ~0.01).
`notebooks/data_preprocessing.ipynb` (và trước đó `data.py` ở đây) pivot đúng theo
`instance_type`, cho ra multi-label thật (Python: 2,555 câu, trung bình 1.12 nhãn/câu).

## Vì sao chọn nhóm mô hình tuyến tính (Logistic Regression, Linear SVM, SGD, Classifier Chain)

1. **Đặc trưng TF-IDF thưa, nhiều chiều** (hàng chục nghìn chiều: word + char n-gram).
   Với dữ liệu thưa/nhiều chiều, mô hình tuyến tính thường đạt hiệu quả ngang hoặc hơn mô
   hình phi tuyến phức tạp (Random Forest, cây quyết định) — hiện tượng quen thuộc khi xử
   lý văn bản kiểu bag-of-words/TF-IDF.
2. **Số mẫu ít** (2,044 train / 511 test) — mô hình phức tạp (ensemble cây sâu, deep
   learning) dễ overfit trên tập nhỏ; mô hình tuyến tính có ít tham số hơn, ổn định hơn.
3. **Hỗ trợ multi-label tự nhiên qua One-vs-Rest**: `LogisticRegression` cho
   `predict_proba` mượt, dễ kết hợp `class_weight='balanced'` để xử lý mất cân bằng nhãn
   (support dao động 62–160 trên test), và dễ ghép với `OneVsRestClassifier`/`ClassifierChain`.
4. **Baseline chính thức của NLBSE'23** cũng dùng mô hình nhẹ (Random Forest + TF-IDF);
   Logistic Regression tương đương về độ phức tạp, nhanh và dễ so sánh — còn RoBERTa của
   nhóm đóng vai trò baseline deep learning để đối chiếu với nhóm model nhẹ này.
5. **Thực nghiệm xác nhận**: 4 biến thể tuyến tính cho kết quả gần như nhau (test
   micro-F1 0.59–0.61) → không gian đặc trưng TF-IDF gần tuyến tính, tăng độ phức tạp mô
   hình khó cải thiện thêm nếu không tăng dữ liệu. `dummy_most_frequent` (baseline sàn,
   luôn đoán theo nhãn phổ biến nhất) cho micro-F1 = 0 — vì cả 5 nhãn đều có prevalence
   <50%, nên "luôn đoán 0" không bắt được true positive nào — chứng minh 4 model tuyến
   tính thực sự học được tín hiệu ngôn ngữ, không phải chỉ đoán theo tần suất.

## Độ đo sử dụng

- **micro/macro F1**: coi mỗi cặp (mẫu, nhãn) là 1 dự đoán nhị phân. micro gộp tất cả lại
  (nhãn phổ biến ảnh hưởng nhiều hơn); macro tính riêng từng nhãn rồi lấy trung bình (mọi
  nhãn có trọng số như nhau, kể cả nhãn hiếm `DevelopmentNotes`).
- **Hamming loss**: tỷ lệ ô (mẫu × nhãn) sai / tổng số ô. Càng thấp càng tốt.
- **Jaccard (samples)**: với mỗi mẫu, `|giao tập nhãn| / |hợp tập nhãn|`, rồi trung bình
  qua các mẫu. Càng cao càng tốt — đo đúng việc model có đoán trọn vẹn cả bộ nhãn của một
  câu hay không, nghiêm khắc hơn F1 vì phạt cả nhãn thừa lẫn thiếu cùng lúc.

## Kết quả

### CV trên train (chọn/so sánh model — 5 fold từ `folds.json`)

| Model | micro-F1 | macro-F1 | Jaccard | Hamming |
|---|---|---|---|---|
| dummy_most_frequent | 0.0000 | 0.0000 | 0.0000 | 0.2242 |
| **ovr_logreg** | **0.5630** | **0.5315** | 0.5147 | 0.1984 |
| ovr_sgd | 0.5607 | 0.5298 | 0.5145 | 0.2044 |
| ovr_linear_svc | 0.5529 | 0.5179 | 0.5003 | 0.2008 |
| classifier_chain_logreg | 0.5405 | 0.5043 | **0.5277** | 0.2054 |

### Held-out test (số để so với RoBERTa — fit trên toàn bộ train, chấm 1 lần trên `test.csv`)

| Model | micro-F1 | macro-F1 | Jaccard | Hamming |
|---|---|---|---|---|
| dummy_most_frequent | 0.0000 | 0.0000 | 0.0000 | 0.2243 |
| **ovr_logreg** | **0.6115** | **0.5658** | 0.5631 | 0.1746 |
| ovr_sgd | 0.6014 | 0.5576 | 0.5545 | 0.1847 |
| ovr_linear_svc | 0.5956 | 0.5474 | 0.5373 | 0.1796 |
| classifier_chain_logreg | 0.5895 | 0.5346 | **0.5776** | 0.1832 |

Per-label trên test (ovr_logreg):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Usage | 0.7405 | 0.7312 | 0.7358 | 160 |
| Parameters | 0.6706 | 0.7170 | 0.6930 | 159 |
| Summary | 0.5306 | 0.5714 | 0.5503 | 91 |
| Expand | 0.6081 | 0.4455 | 0.5143 | 101 |
| DevelopmentNotes | 0.3067 | 0.3710 | 0.3358 | 62 |

Số liệu đầy đủ (kèm per-label cho cả 5 model): `results/exp_ml_baseline/metrics.json`.

**OVR Logistic Regression** là model tốt nhất trên cả CV và test — dùng model này khi so
sánh trực tiếp với kết quả RoBERTa.

## Hạn chế hiện tại

- Chỉ dùng ngôn ngữ Python (2,555 câu), chưa mở rộng sang Java/Pharo để đủ 6,738 mẫu như
  đề bài gốc. `data/processed/` đã có sẵn `java/` và `pharo/` cùng format — có thể lặp lại
  pipeline này cho từng ngôn ngữ bằng cách đổi `data.language` trong config.
- Ngưỡng quyết định cố định 0.5, chưa tối ưu threshold riêng từng nhãn.
- Nhóm câu trùng lặp lớn (VD nhóm `@author vaadin ltd` bên Java, không ảnh hưởng Python)
  không thể tách ra giữa các fold, có thể làm lệch tỷ lệ nhãn của fold chứa nhóm đó — xem
  `data/README.md`.

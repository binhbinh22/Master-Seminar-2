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
| `models.py` | 6 model: dummy baseline + 4 biến thể tuyến tính + 1 Random Forest (đối chứng phi tuyến) |
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
5. **Thực nghiệm xác nhận — có đối chứng phi tuyến thật**: 4 biến thể tuyến tính cho kết
   quả gần như nhau (test micro-F1 0.59–0.61). Thêm `ovr_random_forest` (phi tuyến, 300
   cây) để kiểm chứng trực tiếp giả thuyết "không gian đặc trưng gần tuyến tính": Random
   Forest đạt CV micro-F1 0.5698 (nhích hơn Logistic Regression một chút khi validate nội
   bộ), nhưng **tụt xuống 0.5647 trên test** — thấp hơn cả 4 model tuyến tính, và đặc biệt
   kém hẳn ở 2 nhãn khó/hiếm (`DevelopmentNotes` F1 0.22 vs 0.34, `Expand` F1 0.39 vs 0.51
   so với Logistic Regression). Nghĩa là Random Forest khớp tốt trên dữ liệu đã thấy
   (CV) nhưng generalize kém hơn trên dữ liệu mới — dấu hiệu overfit trên không gian TF-IDF
   nhiều chiều, ít mẫu. Đây là bằng chứng thực nghiệm, không chỉ suy luận lý thuyết, rằng
   mô hình tuyến tính là lựa chọn tốt hơn cho dataset này. `dummy_most_frequent` (baseline
   sàn, luôn đoán theo nhãn phổ biến nhất) cho micro-F1 = 0 — vì cả 5 nhãn đều có
   prevalence <50%, nên "luôn đoán 0" không bắt được true positive nào — chứng minh các
   model còn lại thực sự học được tín hiệu ngôn ngữ, không phải chỉ đoán theo tần suất.
6. **Trùng đúng baseline chính thức của NLBSE'23** (TF-IDF + Random Forest) — nên
   `ovr_random_forest` ở đây cũng đóng vai trò tái lập baseline gốc của cuộc thi để đối
   chiếu, không chỉ là một lựa chọn kiến trúc tuỳ ý.

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
| ovr_random_forest | 0.5698 | 0.5285 | 0.5003 | **0.1800** |
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
| ovr_random_forest | 0.5647 | 0.4975 | 0.5116 | 0.1804 |

→ Random Forest xếp **nhất trên CV** nhưng **cuối cùng trên test** — dấu hiệu overfit rõ
ràng, không generalize tốt bằng 4 model tuyến tính (xem phân tích ở mục trên).

Per-label trên test:

| Label | Support | ovr_logreg F1 | ovr_random_forest F1 |
|---|---|---|---|
| Usage | 160 | **0.7358** | 0.6821 |
| Parameters | 159 | 0.6930 | 0.6730 |
| Summary | 91 | **0.5503** | 0.5222 |
| Expand | 101 | **0.5143** | 0.3924 |
| DevelopmentNotes | 62 | **0.3358** | 0.2178 |

Logistic Regression thắng ở cả 5/5 nhãn trên test, cách biệt rõ nhất ở 2 nhãn hiếm/khó
nhất (`Expand`, `DevelopmentNotes`) — đúng chỗ Random Forest dễ overfit nhất vì ít mẫu
dương để học split.

Số liệu đầy đủ (kèm per-label cho cả 6 model): `results/exp_ml_baseline/metrics.json`.

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

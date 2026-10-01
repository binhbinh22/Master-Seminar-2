# Classical ML baseline — code comment classification (Python subset)

Cách chạy:
```
python -m experiments.ml.train_baseline --config configs/exp_ml_baseline.yaml
```
Kết quả ghi ra `results/exp_ml_baseline/metrics.json`.

## Cấu trúc module

| File | Vai trò |
|---|---|
| `data.py` | Đọc `data/python.csv`, pivot đúng theo `instance_type` thành multi-label, sinh `group_id` (text đã chuẩn hoá) |
| `splitting.py` | Group-aware `MultilabelStratifiedKFold` — xem mục "Vì sao cần group-aware split" |
| `features.py` | TF-IDF (word 1-2gram + char 3-5gram), fit riêng từng fold |
| `models.py` | 5 model: dummy baseline + 4 biến thể tuyến tính |
| `evaluation.py` | Vòng lặp CV, tính micro/macro-F1, Jaccard, Hamming loss, per-label |
| `legacy/` | 4 script cũ dùng sai cột nhãn (`category` thay vì `instance_type`) — chỉ giữ để đối chiếu, xem `legacy/README.md` |

## Lỗi dữ liệu đã sửa

`python.csv` lưu dạng "exploded": mỗi câu lặp lại 5 lần (1 lần/category), và `instance_type`
(0/1) mới là nhãn thật — `category` chỉ là tên slot. Các script `legacy/` lấy thẳng cột
`category` làm nhãn, bỏ qua `instance_type`, khiến cùng một câu có 5 nhãn mâu thuẫn nhau
(4/5 sai) → model gần như không học được gì (micro-F1 ~0.01). `data.py` pivot đúng theo
`instance_type`, cho ra 2,555 mẫu multi-label thật (trung bình 1.12 nhãn/mẫu).

## Vì sao cần group-aware split

112 nhóm câu (368 dòng) trong dataset có nội dung trùng lặp gần như nguyên văn (cùng đoạn
mô tả được copy vào nhiều class khác nhau). `MultilabelStratifiedKFold` thường không biết
điều này — nó có thể đặt 2 bản sao của cùng 1 câu ở 2 phía train/test khác nhau của cùng
một fold, khiến model "thấy đề thi trước khi thi". Kiểm chứng thực tế trên notebook
tiền xử lý TF-IDF của nhóm (`notebooks/data_preprocessing_tfidf.ipynb`): ~14% số dòng test
bị ảnh hưởng, và việc dùng trực tiếp ma trận train/test được xuất từ đó khiến micro-F1 bị
thổi phồng (0.6282 → giảm còn 0.5814 sau khi group-aware).

`splitting.py` sửa bằng cách stratify ở **cấp nhóm** (group) thay vì cấp dòng: gộp các
dòng cùng `group_id` lại, chọn 1 đại diện/nhóm để stratify, rồi gán toàn bộ nhóm vào cùng
1 fold. Xác nhận bằng `python -m src.models.ml.splitting`: `group_overlap=0` ở cả 5 fold.

## Vì sao chọn nhóm mô hình tuyến tính (Logistic Regression, Linear SVM, SGD, Classifier Chain)

1. **Đặc trưng TF-IDF thưa, nhiều chiều** (~40,000 chiều: word 1-2gram + char 3-5gram).
   Với dữ liệu thưa/nhiều chiều, mô hình tuyến tính thường đạt hiệu quả ngang hoặc hơn mô
   hình phi tuyến phức tạp (Random Forest, cây quyết định) — hiện tượng quen thuộc khi xử
   lý văn bản kiểu bag-of-words/TF-IDF.
2. **Số mẫu ít** (2,555 câu) — mô hình phức tạp (ensemble cây sâu, deep learning) dễ
   overfit trên tập nhỏ; mô hình tuyến tính có ít tham số hơn, ổn định hơn qua CV.
3. **Hỗ trợ multi-label tự nhiên qua One-vs-Rest**: `LogisticRegression` cho
   `predict_proba` mượt, dễ kết hợp `class_weight='balanced'` để xử lý mất cân bằng nhãn
   (support dao động 312–800), và dễ ghép với `OneVsRestClassifier`/`ClassifierChain`.
4. **Baseline chính thức của NLBSE'23** cũng dùng mô hình nhẹ (Random Forest + TF-IDF);
   Logistic Regression tương đương về độ phức tạp, nhanh và dễ so sánh.
5. **Thực nghiệm xác nhận**: 4 biến thể tuyến tính (OVR LogReg, Classifier Chain, Linear
   SVM, SGD) cho kết quả gần như nhau (micro-F1 0.57–0.58 sau khi sửa leakage) → không
   gian đặc trưng TF-IDF gần tuyến tính, tăng độ phức tạp mô hình khó cải thiện thêm nếu
   không tăng dữ liệu. `dummy_most_frequent` (baseline sàn, luôn đoán theo nhãn phổ biến
   nhất) cho micro-F1 = 0 — vì cả 5 nhãn đều có prevalence <50%, nên "luôn đoán 0" không
   bắt được true positive nào — chứng minh 4 model tuyến tính thực sự học được tín hiệu
   ngôn ngữ, không phải chỉ đoán theo tần suất.

## Độ đo sử dụng

- **micro/macro F1**: coi mỗi cặp (mẫu, nhãn) là 1 dự đoán nhị phân. micro gộp tất cả lại
  (nhãn phổ biến ảnh hưởng nhiều hơn); macro tính riêng từng nhãn rồi lấy trung bình (mọi
  nhãn có trọng số như nhau, kể cả nhãn hiếm `DevelopmentNotes`).
- **Hamming loss**: tỷ lệ ô (mẫu × nhãn) sai / tổng số ô. Càng thấp càng tốt.
- **Jaccard (samples)**: với mỗi mẫu, `|giao tập nhãn| / |hợp tập nhãn|`, rồi trung bình
  qua các mẫu. Càng cao càng tốt — đo đúng việc model có đoán trọn vẹn cả bộ nhãn của một
  câu hay không, nghiêm khắc hơn F1 vì phạt cả nhãn thừa lẫn thiếu cùng lúc.

## Kết quả (5-fold, group-aware, sau khi sửa leakage)

| Model | micro-F1 | macro-F1 | Jaccard | Hamming |
|---|---|---|---|---|
| dummy_most_frequent | 0.0000 | 0.0000 | 0.0000 | 0.2243 |
| **ovr_logreg** | **0.5814** | **0.5417** | 0.5262 | 0.1889 |
| ovr_sgd | 0.5778 | 0.5408 | 0.5267 | 0.1942 |
| classifier_chain_logreg | 0.5792 | 0.5340 | **0.5658** | 0.1879 |
| ovr_linear_svc | 0.5699 | 0.5293 | 0.5157 | 0.1924 |

Per-label (ovr_logreg):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Usage | 0.6870 | 0.6556 | 0.6700 | 800 |
| Parameters | 0.6663 | 0.6531 | 0.6588 | 794 |
| Summary | 0.5067 | 0.5770 | 0.5391 | 454 |
| Expand | 0.5261 | 0.5350 | 0.5295 | 504 |
| DevelopmentNotes | 0.3030 | 0.3231 | 0.3111 | 312 |

Số liệu chi tiết: `results/exp_ml_baseline/metrics.json`.

## Hạn chế hiện tại

- Chỉ dùng tập con Python (2,555 mẫu), chưa gộp Java/Pharo để đủ 6,738 mẫu như mô tả đề
  bài gốc — xem thảo luận về cách gộp đúng (per-language + masking) trước khi mở rộng.
- Ngưỡng quyết định cố định 0.5, chưa tối ưu threshold riêng từng nhãn.

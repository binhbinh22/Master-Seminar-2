# Báo cáo: Code Comment Classification (NLBSE'23, Python subset)

Ngày: 2026-09-24

## Nhiệm vụ

> Project 2: Code Comment Classification
> Dataset: NLBSE'23 tool competition (Code Comment Classification), 6,738 samples, multi-label.
> Các bước: implement ML/DL classifier → đánh giá bằng metric phù hợp → k-fold CV → chọn kỹ thuật ML phù hợp → chạy evaluation → so sánh với baseline (RoBERTa) → phân tích kết quả.

---

## Bước 1 — Khảo sát dữ liệu thô

File có 12,775 dòng, 6 cột: `comment_sentence_id`, `class`, `comment_sentence`, `partition`, `instance_type`, `category`.

Kiểm tra nhanh phát hiện:
- Chỉ có 2,555 `comment_sentence_id` duy nhất, nhưng mỗi id lặp lại đúng 5 lần.
- 5 lần lặp đó ứng với 5 category cố định: `Usage`, `Parameters`, `DevelopmentNotes`, `Expand`, `Summary`.
- Cột `instance_type` (0/1) mới là **nhãn thật**: 1 nghĩa là câu đó thực sự thuộc category ghi ở hàng đó, 0 nghĩa là không.

→ Kết luận: dữ liệu ở dạng "exploded" (mỗi câu × mỗi category = 1 hàng), không phải "mỗi hàng là 1 mẫu với 1 nhãn `category`.

### Ý nghĩa 5 nhãn (category)

Taxonomy dành riêng cho comment ở mức class trong Python (NLBSE'23):

| Nhãn | Ý nghĩa | Support |
|---|---|---|
| **Summary** | Tóm tắt chung, mục đích/chức năng chính của class | 454 |
| **Expand** | Giải thích/mở rộng chi tiết thêm ngoài phần tóm tắt (cách hoạt động, ngữ cảnh) | 504 |
| **Parameters** | Mô tả tham số, thuộc tính (attributes) của class/hàm | 794 |
| **Usage** | Hướng dẫn cách dùng, ví dụ sử dụng class | 800 |
| **DevelopmentNotes** | Ghi chú cho developer: TODO, cảnh báo, lưu ý kỹ thuật, hạn chế | 312 |

Một câu comment có thể mang **nhiều nhãn cùng lúc** (VD: vừa `Usage` vừa `Parameters`) — đây chính là lý do bài toán là **multi-label**, không phải multi-class (mỗi mẫu chỉ 1 nhãn duy nhất).

## Bước 2 — data loader ([data_utils.py](data_utils.py))

Viết hàm `load_multilabel()`: pivot 12,775 dòng "exploded" về lại 2,555 dòng — mỗi dòng là 1 câu comment duy nhất, với vector nhãn multi-hot 5 chiều được dựng từ `instance_type` (lấy max theo `category`).

Kết quả sau khi sửa:
- 2,555 mẫu (đúng số câu duy nhất).
- Trung bình 1.12 nhãn/mẫu → xác nhận đúng là bài toán **multi-label**.
- Phân bố nhãn: `Usage`=800, `Parameters`=794, `Expand`=504, `Summary`=454, `DevelopmentNotes`=312.

## Bước 3 — Trích xuất đặc trưng

Dùng TF-IDF kết hợp 2 loại n-gram qua `FeatureUnion` (trong [ml_pipeline.py](ml_pipeline.py)):
- Word n-gram (1–2), `min_df=2`, tối đa 20,000 chiều.
- Char n-gram (3–5, `char_wb`), `min_df=2`, tối đa 20,000 chiều — giúp bắt các mẫu hình thái/cấu trúc câu ngắn (comment code thường ngắn, nhiều thuật ngữ kỹ thuật).

Vectorizer được **fit lại trên từng fold** (không fit trên toàn bộ dữ liệu trước) để tránh rò rỉ thông tin từ tập test sang train.

## Bước 4 — Chọn kỹ thuật Machine Learning

Bài toán multi-label + đặc trưng TF-IDF thưa (sparse), số mẫu vừa phải (2,555) → ưu tiên các mô hình tuyến tính, huấn luyện nhanh, ít overfit:

| Mô hình | Cách xử lý multi-label |
|---|---|
| One-vs-Rest Logistic Regression | 1 classifier nhị phân/nhãn, độc lập |
| Classifier Chain (Logistic Regression) | Chuỗi classifier, nhãn sau dùng dự đoán nhãn trước làm đặc trưng → mô hình hoá tương quan giữa các nhãn |
| One-vs-Rest Linear SVM | 1 classifier margin-max/nhãn |
| One-vs-Rest SGD (log-loss) | Xấp xỉ logistic regression, huấn luyện online/nhanh hơn |

Tất cả đều dùng `class_weight='balanced'` để bù cho mất cân bằng nhãn (312 vs 800 mẫu).

### Vì sao chọn nhóm mô hình tuyến tính (Logistic Regression...) thay vì thuật toán khác

1. **Đặc trưng TF-IDF thưa, nhiều chiều** (~40,000 chiều: word 1-2gram + char 3-5gram). Với dữ liệu thưa/nhiều chiều, mô hình tuyến tính thường đạt hiệu quả ngang hoặc hơn mô hình phi tuyến phức tạp (Random Forest, cây quyết định) — hiện tượng quen thuộc khi xử lý văn bản kiểu bag-of-words/TF-IDF.
2. **Số mẫu ít** (2,555 câu) — mô hình phức tạp (deep learning, ensemble cây sâu) dễ overfit trên tập nhỏ; mô hình tuyến tính có ít tham số hơn, ổn định hơn khi train/test qua CV.
3. **Hỗ trợ multi-label tự nhiên qua One-vs-Rest**: `LogisticRegression` cho `predict_proba` mượt (xác suất), dễ kết hợp `class_weight='balanced'` để xử lý mất cân bằng nhãn, và dễ ghép với `OneVsRestClassifier`/`ClassifierChain`.
4. **Baseline chính thức của NLBSE'23** cũng dùng mô hình nhẹ (Random Forest, TF-IDF); Logistic Regression là lựa chọn tương đương về độ phức tạp, nhanh và dễ so sánh.
5. **Thực nghiệm xác nhận lựa chọn đúng**: chạy song song 4 biến thể tuyến tính (OVR LogReg, Classifier Chain, Linear SVM, SGD) đều cho kết quả gần như nhau (micro-F1 0.61–0.63, xem Bước 6) → không gian đặc trưng TF-IDF gần tuyến tính, tăng độ phức tạp mô hình khó cải thiện thêm nếu không tăng dữ liệu. Logistic Regression được chọn làm đại diện chính vì đơn giản, dễ diễn giải (hệ số → từ nào ảnh hưởng nhãn nào), và có `predict_proba` để tune threshold về sau.

## Bước 5 — Thiết lập k-fold cross-validation

Dùng `MultilabelStratifiedKFold` (thư viện `iterative-stratification`), k=5. Đây là kỹ thuật stratify chuyên cho multi-label — đảm bảo tỷ lệ từng nhãn được giữ tương đối đều giữa các fold train/test, khác với `KFold` thường hay `StratifiedKFold` (chỉ hoạt động đúng với single-label).

## Bước 6 — Chạy evaluation

### Giải thích các độ đo (metric)

Với multi-label, một mẫu có thể đúng "một phần" (đúng vài nhãn, sai/thiếu vài nhãn khác), nên accuracy thường gây hiểu nhầm. Các độ đo dùng ở đây:

- **Precision / Recall / F1 (micro & macro)**: coi mỗi cặp (mẫu, nhãn) là một dự đoán nhị phân độc lập.
  - *micro*: gộp tất cả (mẫu, nhãn) lại rồi tính 1 lần → nhãn phổ biến (nhiều support) ảnh hưởng nhiều hơn.
  - *macro*: tính F1 riêng từng nhãn rồi lấy trung bình cộng → mọi nhãn có trọng số như nhau, nhãn hiếm (`DevelopmentNotes`) ảnh hưởng ngang nhãn phổ biến.
- **Hamming loss**: tỷ lệ ô (mẫu × nhãn) bị dự đoán sai trên tổng số ô.
  `Hamming loss = (số ô sai) / (số mẫu × số nhãn)`. Càng **thấp** càng tốt (0 = hoàn hảo). Đo lỗi ở mức chi tiết nhất — từng quyết định nhị phân riêng lẻ.
- **Jaccard (samples)**: với mỗi mẫu, so **tập nhãn dự đoán** với **tập nhãn thật**: `Jaccard = |Giao| / |Hợp|`, rồi lấy trung bình qua các mẫu. Càng **cao** càng tốt (1 = trùng khớp tuyệt đối cả bộ nhãn). Đây là độ đo nghiêm khắc nhất vì phạt cùng lúc cả nhãn thừa lẫn nhãn thiếu trong một mẫu — khác với F1 vốn đánh giá từng nhãn tách rời.

Ba nhóm độ đo bổ sung cho nhau: F1 cho biết mô hình mạnh/yếu ở nhãn nào, Hamming loss cho biết mức lỗi tổng thể chi tiết, Jaccard cho biết mô hình có đoán **đúng trọn bộ nhãn của từng câu** hay không — quan trọng nếu ứng dụng thực tế cần cả bộ nhãn chính xác (chứ không chỉ đúng từng nhãn riêng lẻ).

Chạy `python3 ml_pipeline.py --folds 5`. Kết quả trung bình 5-fold:

| Model | micro-F1 | macro-F1 | Jaccard (samples) |
|---|---|---|---|
| **OVR Logistic Regression** | **0.6282** | **0.5928** | 0.5843 |
| OVR SGD (log-loss) | 0.6266 | 0.5932 | 0.5876 |
| OVR Linear SVM | 0.6236 | 0.5854 | 0.5801 |
| Classifier Chain (LR) | 0.6098 | 0.5667 | **0.5977** |

Chi tiết: [results/ml_pipeline_results.json](results/ml_pipeline_results.json)

Phân tích per-label với mô hình tốt nhất (OVR Logistic Regression), trung bình 5 fold ([per_label_analysis.py](per_label_analysis.py)):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Usage | 0.7251 | 0.7000 | 0.7118 | 800 |
| Parameters | 0.7035 | 0.7090 | 0.7050 | 794 |
| Expand | 0.5620 | 0.6032 | 0.5815 | 504 |
| Summary | 0.5551 | 0.6079 | 0.5793 | 454 |
| DevelopmentNotes | 0.3639 | 0.4138 | 0.3862 | 312 |



## Bước 7 — Phân tích kết quả

- **F1 tỉ lệ thuận với số lượng mẫu/nhãn**: `Usage` và `Parameters` (support cao nhất, ~800) đạt F1 ~0.70–0.71; `DevelopmentNotes` (ít mẫu nhất, 312) chỉ đạt F1 ~0.39 — thấp hơn hẳn. Đây là dấu hiệu rõ ràng của bottleneck dữ liệu/mất cân bằng nhãn, không phải do mô hình yếu.
- **Các mô hình tuyến tính cho kết quả gần như nhau** (micro-F1 0.61–0.63) → không gian đặc trưng TF-IDF gần như tách tuyến tính được cho bài toán này; tăng độ phức tạp mô hình (phi tuyến) khó cải thiện nhiều nếu không tăng dữ liệu.
- **Classifier Chain đánh đổi**: micro/macro-F1 thấp hơn OVR một chút, nhưng Jaccard (đo đúng toàn bộ tập nhãn của từng mẫu) lại cao nhất — mô hình hoá tương quan giữa các nhãn (VD: câu có `Usage` thường cũng có `Parameters`) giúp dự đoán đúng "trọn bộ nhãn" tốt hơn dù từng nhãn riêng lẻ hơi kém đi.
- **`DevelopmentNotes` là nhãn khó nhất**: vừa ít mẫu nhất, vừa mang ngữ nghĩa "ghi chú khác/không thuộc loại nào khác" — nội dung đa dạng, khó nhận diện qua đặc trưng từ vựng.

## Hạn chế

1. Mới chỉ dùng tập con **Python** (2,555 mẫu), chưa gộp Java + Pharo để đủ 6,738 mẫu như đề bài — cần `java.csv`/`pharo.csv` từ repo NLBSE'23; mỗi ngôn ngữ có bộ category riêng nên cần xử lý cẩn thận khi gộp.
2. Chưa có baseline deep learning (RoBERTa) do thiếu dung lượng đĩa — script đã sẵn sàng, chỉ cần chạy lại.
3. Ngưỡng quyết định cố định 0.5 cho các mô hình OVR/SGD, chưa tối ưu threshold riêng từng nhãn (có thể cải thiện thêm recall cho nhãn hiếm như `DevelopmentNotes`).


# Báo cáo: Code Comment Classification (NLBSE'23, Python subset)

Ngày: 2026-09-24

## Nhiệm vụ

> Project 2: Code Comment Classification
> Dataset: NLBSE'23 tool competition (Code Comment Classification), 6,738 samples, multi-label.
> Các bước: implement ML/DL classifier → đánh giá bằng metric phù hợp → k-fold CV → chọn kỹ thuật ML phù hợp → chạy evaluation → so sánh với baseline (RoBERTa) → phân tích kết quả.

---

## Bước 1 — Khảo sát dữ liệu thô

File `python.csv` có 12,775 dòng, 6 cột: `comment_sentence_id`, `class`, `comment_sentence`, `partition`, `instance_type`, `category`.

Kiểm tra nhanh phát hiện:
- Chỉ có 2,555 `comment_sentence_id` duy nhất, nhưng mỗi id lặp lại đúng 5 lần.
- 5 lần lặp đó ứng với 5 category cố định: `Usage`, `Parameters`, `DevelopmentNotes`, `Expand`, `Summary`.
- Cột `instance_type` (0/1) mới là **nhãn thật**: 1 nghĩa là câu đó thực sự thuộc category ghi ở hàng đó, 0 nghĩa là không.

→ Kết luận: dữ liệu ở dạng "exploded" (mỗi câu × mỗi category = 1 hàng), không phải "mỗi hàng là 1 mẫu với 1 nhãn `category`" như các script gốc trong repo giả định.

## Bước 2 — Sửa lại data loader ([data_utils.py](data_utils.py))

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

## Bước 5 — Thiết lập k-fold cross-validation

Dùng `MultilabelStratifiedKFold` (thư viện `iterative-stratification`), k=5. Đây là kỹ thuật stratify chuyên cho multi-label — đảm bảo tỷ lệ từng nhãn được giữ tương đối đều giữa các fold train/test, khác với `KFold` thường hay `StratifiedKFold` (chỉ hoạt động đúng với single-label).

## Bước 6 — Chạy evaluation

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



## Bước 8 — Phân tích kết quả

- **F1 tỉ lệ thuận với số lượng mẫu/nhãn**: `Usage` và `Parameters` (support cao nhất, ~800) đạt F1 ~0.70–0.71; `DevelopmentNotes` (ít mẫu nhất, 312) chỉ đạt F1 ~0.39 — thấp hơn hẳn. Đây là dấu hiệu rõ ràng của bottleneck dữ liệu/mất cân bằng nhãn, không phải do mô hình yếu.
- **Các mô hình tuyến tính cho kết quả gần như nhau** (micro-F1 0.61–0.63) → không gian đặc trưng TF-IDF gần như tách tuyến tính được cho bài toán này; tăng độ phức tạp mô hình (phi tuyến) khó cải thiện nhiều nếu không tăng dữ liệu.
- **Classifier Chain đánh đổi**: micro/macro-F1 thấp hơn OVR một chút, nhưng Jaccard (đo đúng toàn bộ tập nhãn của từng mẫu) lại cao nhất — mô hình hoá tương quan giữa các nhãn (VD: câu có `Usage` thường cũng có `Parameters`) giúp dự đoán đúng "trọn bộ nhãn" tốt hơn dù từng nhãn riêng lẻ hơi kém đi.
- **`DevelopmentNotes` là nhãn khó nhất**: vừa ít mẫu nhất, vừa mang ngữ nghĩa "ghi chú khác/không thuộc loại nào khác" — nội dung đa dạng, khó nhận diện qua đặc trưng từ vựng.

## Hạn chế

1. Mới chỉ dùng tập con **Python** (2,555 mẫu), chưa gộp Java + Pharo để đủ 6,738 mẫu như đề bài — cần `java.csv`/`pharo.csv` từ repo NLBSE'23; mỗi ngôn ngữ có bộ category riêng nên cần xử lý cẩn thận khi gộp.
2. Chưa có baseline deep learning (RoBERTa) do thiếu dung lượng đĩa — script đã sẵn sàng, chỉ cần chạy lại.
3. Ngưỡng quyết định cố định 0.5 cho các mô hình OVR/SGD, chưa tối ưu threshold riêng từng nhãn (có thể cải thiện thêm recall cho nhãn hiếm như `DevelopmentNotes`).

## File liên quan

| File | Vai trò |
|---|---|
| [data_utils.py](data_utils.py) | Load & pivot dữ liệu đúng (multi-label) |
| [ml_pipeline.py](ml_pipeline.py) | 4 mô hình ML cổ điển + 5-fold CV |
| [per_label_analysis.py](per_label_analysis.py) | Phân tích precision/recall/F1 theo từng nhãn |
| [roberta_pipeline.py](roberta_pipeline.py) | Fine-tune RoBERTa baseline (sẵn sàng, chưa chạy) |
| [results/ANALYSIS.md](results/ANALYSIS.md) | Bản tóm tắt kỹ thuật (ngắn hơn) |
| `run_ml_baseline.py`, `full_pipeline.py`, `detailed_analysis.py`, `multiclass_pipeline.py` | Pipeline gốc, **đã lỗi**, giữ lại để đối chiếu — nên xoá hoặc archive |

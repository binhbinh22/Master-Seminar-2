# So sánh ML cổ điển vs RoBERTa — Code Comment Classification (Python)

Ngày: 2026-10-31

## 1. Mục tiêu

Báo cáo này so sánh 2 nhánh làm việc độc lập trong cùng nhóm cho đề bài **Project 2:
Code Comment Classification (NLBSE'23)**:

- **ML cổ điển** (nhánh `binh/ML`): TF-IDF + các mô hình tuyến tính/cây, triển khai ở
  `src/models/ml/` và `experiments/ml/train_baseline.py`.
- **RoBERTa / Deep Learning** (nhánh `dl/roberta_baseline`): fine-tune RoBERTa-base với
  12 biến thể kỹ thuật (E1–E12), triển khai và báo cáo đầy đủ trong `report.pdf` /
  `docs/report/`.

Hai bên **dùng chung đúng một bộ dữ liệu đã xử lý** (`data/processed/python/`), đã xác
minh khớp số liệu (xem Mục 2) — nên các con số test dưới đây so sánh được trực tiếp,
không cần quy đổi hay giả định gì thêm.

## 2. Dataset

### Nguồn chung

Cả hai nhánh đọc từ `data/processed/python/` (`train.csv`, `test.csv`, `folds.json`,
`label_cols.json`), sinh ra bởi `notebooks/data_preprocessing.ipynb` — pivot đúng theo
`instance_type` (không dùng nhầm cột `category`), group-aware theo text chuẩn hoá (không
rò rỉ câu trùng giữa train/test hay giữa các fold).

| | ML (`binh/ML`) | RoBERTa (`report.pdf`) |
|---|---|---|
| Train | 2.044 | 2.044 |
| Test | 511 | 511 |
| Fold 0–4 (val size) | 409, 409, 408, 409, 409 | 409, 409, 408, 409, 409 |

→ Khớp tuyệt đối ở cả 2 nguồn độc lập — xác nhận cùng 1 split, không có sai lệch dữ liệu
giữa 2 nhánh.

### Đặc điểm dữ liệu

- 5 nhãn: `Usage`, `Parameters`, `Expand`, `Summary`, `DevelopmentNotes`.
- Multi-label: trung bình 1.121 nhãn dương/câu, 11.1% câu có ≥2 nhãn, không câu nào rỗng
  nhãn.
- Support trên test (giống hệt ở cả 2 báo cáo): Usage=160, Parameters=159, Expand=101,
  Summary=91, DevelopmentNotes=62 — lệch nhau rõ rệt (tỷ lệ lớn nhất/nhỏ nhất ≈ 2.6×).
- 95% câu ≤ 13 từ / ≤ 19 token RoBERTa — câu rất ngắn, nhiều câu chỉ 1–3 từ (27.6%).

### Phạm vi — khác biệt quan trọng

| | ML | RoBERTa |
|---|---|---|
| Ngôn ngữ đã chạy | Python (2.555 câu) | Python + Java + Pharo (6.738 câu) |
| Số nhãn | 5 | 5 / 7 / 7 (19 tổng) |

RoBERTa đã phủ **toàn bộ phạm vi 6.738 mẫu** mà đề bài nêu; ML mới dừng ở Python. So sánh
trong báo cáo này **chỉ giới hạn ở Python** — phần duy nhất 2 bên overlap.

## 3. Mô hình

### 3.1 ML cổ điển

Đặc trưng: TF-IDF word (1-2gram) + char (3-5gram, `char_wb`), ~20k chiều/loại, fit lại
mỗi lần train (không leak qua vectorizer).

| Model | Mô tả |
|---|---|
| `dummy_most_frequent` | Baseline sàn, luôn đoán nhãn phổ biến nhất mỗi cột |
| `ovr_logreg` | One-vs-Rest Logistic Regression, `class_weight='balanced'` |
| `classifier_chain_logreg` | Chuỗi classifier, mô hình hoá tương quan giữa nhãn |
| `ovr_linear_svc` | One-vs-Rest Linear SVM |
| `ovr_sgd` | One-vs-Rest SGD (log-loss) |
| `ovr_random_forest` | One-vs-Rest Random Forest (300 cây) — đối chứng phi tuyến, tái lập baseline chính thức NLBSE'23 |

### 3.2 RoBERTa (tóm tắt, chi tiết xem `report.pdf`)

Backbone `roberta-base` (768 chiều), fine-tune toàn bộ, đầu ra sigmoid + BCE (không
softmax), max 128 token, AdamW lr=2e-5, tối đa 10 epoch/fold, early stopping theo
validation micro-F1.

| Exp | Kỹ thuật | Vai trò |
|---|---|---|
| **E1** | RoBERTa + BCE | **Baseline — dùng để so sánh trực tiếp với ML ở đây** |
| E2 | Weighted BCE | Xử lý mất cân bằng nhãn |
| E3 | Focal Loss | Tập trung ví dụ khó |
| E4 | Threshold tuning (OOF) | Hiệu chỉnh ngưỡng quyết định |
| E5/E6 | Label-aware attention (+contrastive) | Biểu diễn theo nhãn |
| E7 | Asymmetric Loss | Cân bằng precision/recall bất đối xứng |
| E8 | R-Drop | Regularization |
| E9 | Label GNN | Mô hình hoá tương quan nhãn qua graph |
| E10 | Adversarial FGM | Regularization embedding |
| E11 | CodeBERT + BCE | Backbone pretrain trên code |
| E12 | Attention + GNN | Kết hợp E5 + E9 |

## 4. Thiết lập đánh giá

Cả 2 bên dùng **đúng cùng quy trình 2 bước**:

1. **CV trên train** (5 fold từ `folds.json`) — dùng để chọn/so sánh model, không chạm
   vào test.
2. **Held-out test** (511 câu) — fit lại trên toàn bộ train, chấm **đúng 1 lần** — đây là
   con số dùng để so sánh ML vs RoBERTa trong báo cáo này.

Metric: micro-F1, macro-F1, Hamming loss (cả 2 bên); Jaccard-samples (chỉ bên ML — bên
RoBERTa không báo cáo).

## 5. Kết quả

### 5.1 Tổng thể trên test (511 câu)

| Model | micro-F1 | macro-F1 | Hamming loss |
|---|---|---|---|
| **RoBERTa E1 (baseline)** | **0.6466** | **0.5755** | **0.1472** |
| RoBERTa E9 (Label GNN, cao nhất trong 12 biến thể) | 0.6591 | 0.5867 | 0.1413 |
| RoBERTa E11 (CodeBERT, chọn theo CV) | 0.6467 | 0.5627 | 0.1479 |
| ML — `ovr_logreg` (tốt nhất bên ML) | 0.6115 | 0.5658 | 0.1746 |
| ML — `ovr_sgd` | 0.6014 | 0.5576 | 0.1847 |
| ML — `ovr_linear_svc` | 0.5956 | 0.5474 | 0.1796 |
| ML — `classifier_chain_logreg` | 0.5895 | 0.5346 | 0.1832 |
| ML — `ovr_random_forest` | 0.5647 | 0.4975 | 0.1804 |
| Dummy / `ovr_random_forest` (RF bên ML) không vượt baseline | — | — | — |

→ **RoBERTa E1 thắng Logistic Regression khoảng 3.5 điểm % micro-F1** (0.6466 vs 0.6115)
và **~1 điểm % macro-F1** (0.5755 vs 0.5658). Khoảng cách tồn tại nhưng không áp đảo —
ML cổ điển vẫn là baseline cạnh tranh được, không bị bỏ xa.

### 5.2 Per-label trên test

| Label | Support | ML `ovr_logreg` F1 | RoBERTa E1 F1 | Chênh lệch |
|---|---|---|---|---|
| DevelopmentNotes | 62 | **0.3358** | 0.32 | **ML thắng** (+1.6 điểm %) |
| Expand | 101 | **0.5143** | 0.38 | **ML thắng** (+13.4 điểm %) |
| Parameters | 159 | 0.6930 | **0.73** | RoBERTa thắng (+3.7 điểm %) |
| Summary | 91 | 0.5503 | **0.67** | RoBERTa thắng (+12.0 điểm %) |
| Usage | 160 | 0.7358 | **0.79** | RoBERTa thắng (+5.4 điểm %) |

→ Phát hiện đáng chú ý: **ML thắng rõ ở đúng 2 nhãn hiếm/khó nhất** (`DevelopmentNotes`,
`Expand` — support thấp nhất), trong khi RoBERTa thắng ở 3 nhãn phổ biến hơn
(`Parameters`, `Summary`, `Usage`). Vì micro-F1 gộp theo tần suất nhãn (nhãn phổ biến
đóng góp nhiều hơn), RoBERTa thắng rõ hơn trên micro-F1 (3.5 điểm %) so với macro-F1
(~1 điểm %, vì macro coi mọi nhãn ngang nhau — bù trừ bởi 2 nhãn ML thắng).

### 5.3 CV không dự đoán đúng thứ hạng trên test — hiện tượng xảy ra ở cả 2 bên

| Bên | Model CV cao nhất | CV micro-F1 | Nhưng xếp hạng test |
|---|---|---|---|
| ML | `ovr_random_forest` (0.5698) | cao nhất | **thấp nhất** trong 5 model thật (0.5647) |
| RoBERTa | E11 CodeBERT (0.6410±0.0047) | cao nhất, cách biệt rõ | chỉ ngang E1 (0.6467 vs 0.6466) |

Cả 2 nhánh độc lập quan sát cùng 1 hiện tượng: **model tốt nhất trên validation không
chắc tốt nhất trên test** — củng cố lẫn nhau rằng đây là đặc điểm thật của dataset nhỏ
(2.555 câu Python), không phải lỗi quy trình riêng của bên nào.

### 5.4 Chi phí

| | ML (6 model, CV+test) | RoBERTa (1 experiment, CV+test) |
|---|---|---|
| Thời gian | **~63 giây** (CPU) | **~14.8 phút** (GPU A30) |

ML rẻ hơn RoBERTa khoảng **14×** cho 1 model, và vì ML chạy 6 model trong thời gian đó,
chênh lệch thực tế theo "số model thử được/phút" còn lớn hơn nhiều (~85×).

## 6. Nhận xét tổng hợp

1. **RoBERTa thắng nhưng không áp đảo.** Khoảng cách micro-F1 ~3.5 điểm % là có ý nghĩa
   nhưng không lớn đến mức ML trở nên vô dụng — đặc biệt khi cân nhắc chi phí (Mục 5.4).
2. **ML cổ điển mạnh hơn ở nhãn hiếm, RoBERTa mạnh hơn ở nhãn phổ biến.** Giả thuyết hợp
   lý: TF-IDF + Logistic Regression học trực tiếp từ các từ khoá đặc trưng xuất hiện dù
   hiếm (feature sparse, không bị pha loãng); RoBERTa với không gian biểu diễn dày đặc có
   thể cần nhiều mẫu hơn để hội tụ đúng hướng cho nhãn ít dữ liệu (62–101 mẫu dương).
   Đây là giả thuyết — cần thêm thực nghiệm (nhiều seed) để khẳng định chắc chắn.
3. **Không nhãn nào "miễn nhiễm"**: kể cả RoBERTa với 12 biến thể kỹ thuật khác nhau cũng
   không có cấu hình nào thắng tuyệt đối mọi ngôn ngữ/nhãn (Hình 3 trong `report.pdf`) —
   nhất quán với kết quả ML: không có model nào thắng toàn diện.
4. **Trade-off chi phí/hiệu năng rõ ràng.** Nếu mục tiêu là baseline nhanh, dễ triển khai,
   dễ diễn giải (hệ số TF-IDF → từ khoá) — Logistic Regression là lựa chọn hợp lý. Nếu cần
   tối đa hiệu năng và có sẵn GPU — RoBERTa (đặc biệt biến thể E9/E11) đáng đầu tư.
5. **Cả 2 bên đều xác nhận**: Random Forest/CodeBERT đạt CV cao nhất nhưng không giữ được
   lợi thế đó trên test — bài học chung cho cả nhóm: **không chọn model cuối cùng chỉ dựa
   vào CV**, luôn cần nhìn test độc lập trước khi kết luận.

## 7. Hạn chế

- So sánh chỉ ở **Python** — RoBERTa đã có số liệu Java/Pharo (`report.pdf` Bảng 8–9)
  nhưng ML chưa mở rộng sang 2 ngôn ngữ này để so sánh tương tự.
- 1 seed (42) ở cả 2 bên — chênh lệch nhỏ (ví dụ RoBERTa E11 vs E1 chỉ 0.01 điểm % test
  dù CV cách nhau 6.6 điểm %) cần lặp nhiều seed mới khẳng định ý nghĩa thống kê.
- ML chưa threshold-tune (`0.5` cố định) trong khi RoBERTa có hẳn 1 biến thể (E4) làm việc
  này — nếu muốn so sánh công bằng hơn ở khía cạnh threshold, cần thêm bước này cho ML.
- `ovr_random_forest` tái lập tinh thần baseline NLBSE'23 gốc nhưng **không phải cấu hình
  Random Forest chính thức** (số cây, feature, tiền xử lý có thể khác) — không nên trích
  dẫn như "baseline chính thức của cuộc thi".

## 8. Kết luận

Trên cùng 511 câu test Python, **RoBERTa (E1 baseline) đạt micro-F1 0.6466, nhỉnh hơn
Logistic Regression + TF-IDF (0.6115) khoảng 3.5 điểm %**, với khoảng cách macro-F1 hẹp
hơn (~1 điểm %) vì ML bù lại được ở 2 nhãn hiếm nhất. Với chi phí tính toán chỉ bằng
~1/14 và thời gian triển khai nhanh hơn nhiều, TF-IDF + Logistic Regression là baseline ML
hợp lý và cạnh tranh được cho bài toán này — đúng tinh thần "chọn kỹ thuật phù hợp" mà đề
bài yêu cầu, không nhất thiết phải là kỹ thuật phức tạp nhất.

## Nguồn số liệu

- ML: [`results/exp_ml_baseline/metrics.json`](../results/exp_ml_baseline/metrics.json),
  [`src/models/ml/README.md`](../src/models/ml/README.md)
- RoBERTa: [`report.pdf`](../report.pdf) (Bảng 1, 2, 4, 7, 11; Hình 5), `docs/report/`

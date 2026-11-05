# Classical ML baselines — code comment classification (Python / Java / Pharo)

Ngày: 2026-11-02

## Cách chạy

```bash
# Giai đoạn 1 — 5-fold CV trên 100% dữ liệu, so sánh 6 phương pháp
python -m experiments.ml.train_baseline --stage kfold --language all

# Giai đoạn 2 — chỉ train phương pháp đã chọn, trên 80/10/10
python -m experiments.ml.train_baseline --stage final --language all --methods ovr_logreg ovr_sgd
```

`--language` nhận `python`/`java`/`pharo`/`all`. `--methods` mặc định cả 6 (xem
`src/models/ml/models.py:MODEL_KEYS`). `--seed` mặc định 42.

## Nguồn dữ liệu — dùng chung loader với RoBERTa/CodeBERT

`src/models/ml/data.py` gọi thẳng `src.data.two_stage.load_stage_data()` — **đúng hàm
nhánh DL đang dùng**, không tự viết lại logic đọc/validate CSV. Nhờ vậy:

- Stage `kfold` đọc `data/k_fold_data/<lang>/{all.csv, folds.json}` (100% dữ liệu, dùng
  để so sánh phương pháp).
- Stage `final` đọc `data/full_training_data/<lang>/{train.csv, val.csv, test.csv}`
  (80/10/10, train cuối + test 1 lần).
- Mọi kiểm tra toàn vẹn (group-leakage giữa train/val/test, fingerprint dân số khớp
  manifest, mỗi câu xuất hiện đúng 1 lần ở validation qua 5 fold...) đã được
  `load_stage_data()` tự làm — ML thừa hưởng nguyên vẹn, không có nguy cơ tự viết sai.
- `metrics.json` của ML ghi lại đúng `data_fingerprint`/`population_fingerprint` mà
  `load_stage_data()` tính — **khớp bit-for-bit** với số trong `metrics.json` của
  RoBERTa cùng ngôn ngữ/stage, nên 2 kết quả so sánh được mà không cần tin tưởng mù
  quáng, có thể đối chiếu trực tiếp 2 giá trị hash.

Lưu ý quan trọng được `load_stage_data()` tự gắn vào mọi kết quả qua `evaluation_note`:
*giai đoạn 1 (k-fold) đã dùng toàn bộ dữ liệu, kể cả các câu sau này nằm trong test của
giai đoạn 2 — nên test giai đoạn 2 là kết quả thăm dò, không độc lập hoàn toàn với việc
chọn phương pháp.* Áp dụng y hệt cho cả ML và DL.

## Output chuẩn chung (áp dụng cho cả ML và DL)

Mỗi lần chạy `run_staged(method, language, stage)` ghi ra:

```
results/training/<stage>/<timestamp>/<method>/<language>/
├── config.yaml          # phương pháp, loại model, siêu tham số (từ clf.get_params()), seed
├── data_manifest.json   # copy manifest.json của nguồn dữ liệu (fingerprint, seed, số mẫu...)
├── label_cols.json      # thứ tự nhãn
├── metrics.json         # toàn bộ metric, thời gian, info chọn model
├── predictions.csv      # ID câu, split/fold, nhãn thật, xác suất/score, nhãn dự đoán
├── training.log          # log quá trình (console + file)
├── training_curve.json   # (chỉ stage final) xem mục "Training curve" bên dưới
├── plots/                # biểu đồ CV / test / per-label / training curve
└── artifacts/            # vectorizer.joblib + model.joblib (chỉ stage final)
```

Ghi đè bị chặn: nếu `metrics.json` đã tồn tại ở path đích, `run_staged()` raise lỗi thay
vì âm thầm ghi đè — khớp hành vi `experiments/dl/staged_experiment.py`.

### Khác biệt theo giai đoạn

| | `kfold` | `final` |
|---|---|---|
| `predictions.csv` | **Out-of-fold**: mỗi câu trong `all.csv` dự đoán đúng 1 lần (ở fold nó làm validation) | Chỉ `validation` + `test` (không có `train`) |
| Metric | `folds` (từng fold) + `cross_validation` (mean/std, dùng `src.evaluation.summaries.summarize_metric_runs` — hàm chung với DL) + `out_of_fold` (metric gộp trên toàn bộ OOF) | `validation` (chẩn đoán) + `test` (số báo cáo cuối) |
| `artifacts/` | Không có — stage này chỉ để sàng lọc, không cần giữ model | Có — `vectorizer.joblib` + `model.joblib`, nạp lại được để inference |

**Lưu ý về `validation` ở stage `final`**: khác với DL (dùng validation để chọn
checkpoint/epoch tốt nhất), các model tuyến tính ở đây hội tụ 1 lần, không có khái niệm
"epoch" để chọn — nên `validation` chỉ là số chẩn đoán song song với `test`, **không được
dùng để chọn hyperparameter hay mô hình**. Ghi rõ trong `metrics.json.validation_note`
của từng run để không ai hiểu nhầm đây là bước tuning.

### Training curve — có thật, không phải giả lập

`sklearn` không expose loss theo từng bước lặp của `.fit()` như DL (gọi 1 lần là chạy
xong toàn bộ solver). `src/models/ml/curves.py` tái tạo lại đường cong bằng cách
**refit từ đầu ở nhiều mốc** `max_iter` (logreg/chain/SVC/SGD) hoặc `n_estimators`
(Random Forest), rồi đo `train_loss` (Hamming loss trên train) và `validation micro-F1`
ở mỗi mốc. Vì các solver này **tất định** khi cố định `random_state` (lbfgs, liblinear,
SGD không-shuffle-ngẫu-nhiên-khác-seed), refit ở `max_iter=k` cho đúng trạng thái mà 1
lần fit dài hơn sẽ đi qua ở bước thứ k — đây là tái dựng quỹ đạo tối ưu thật, không phải
nội suy giả. Cái giá phải trả là refit lại nhiều lần thay vì 1 lần fit duy nhất (chấp
nhận được — toàn bộ 18 run stage `final` kèm curve chạy trong ~2.5 phút).

`dummy_most_frequent` không có trong `CURVE_PARAM` — nó không tối ưu gì nên không có gì
để vẽ, `plots/` của nó chỉ có `test_summary.png` + `test_per_label.png`.

Ví dụ đọc kết quả (`ovr_logreg`/Python): train loss giảm 0.30 → 0.018 rồi đi ngang từ
epoch ~7, validation micro-F1 dao động rồi ổn định ~0.565 từ epoch ~8 — **hội tụ rõ
ràng**, `max_iter=2000` hiện tại dư thừa so với mức cần thiết (~150-200 là đủ).

## Metrics — khớp field-name với DL

`src/models/ml/metrics.py` tính đúng các field `src.evaluation.multilabel.masked_multilabel_metrics`
bên DL đã dùng: `micro_precision/recall/f1`, `macro_precision/recall/f1`,
`micro_average_precision`, `macro_average_precision`, `hamming_loss`, và `per_label`
(`precision/recall/f1/average_precision/support`). Một file `metrics.json` của ML và một
file của RoBERTa có thể diff trực tiếp theo tên field, không cần quy đổi.

- **micro-F1/macro-F1**: xem giải thích chi tiết ở lịch sử báo cáo cũ (`docs/`) — micro
  gộp theo tần suất nhãn, macro coi mọi nhãn ngang nhau.
- **Average precision (AP)**: diện tích dưới đường precision-recall theo thứ hạng điểm số
  — không cần điểm đã calibrate thành xác suất thật, chỉ cần đúng thứ tự xếp hạng. Vì vậy
  `ovr_linear_svc` (không có `predict_proba`) vẫn tính được AP từ `decision_function`
  (`src/models/ml/scoring.py`).
- **Hamming loss**: tỷ lệ ô (câu × nhãn) dự đoán sai/tổng số ô.

## Model

| Method key | Mô tả |
|---|---|
| `dummy_most_frequent` | Baseline sàn — luôn đoán nhãn phổ biến nhất mỗi cột |
| `ovr_logreg` | One-vs-Rest Logistic Regression, `class_weight='balanced'` |
| `classifier_chain_logreg` | Chuỗi classifier, mô hình hoá tương quan giữa nhãn |
| `ovr_linear_svc` | One-vs-Rest Linear SVM |
| `ovr_sgd` | One-vs-Rest SGD (log-loss) |
| `ovr_random_forest` | One-vs-Rest Random Forest (300 cây) — đối chứng phi tuyến, tái lập tinh thần baseline NLBSE'23 gốc |

Đặc trưng: TF-IDF word (1-2gram) + char (3-5gram, `char_wb`), ~20k chiều/loại, fit lại
**mỗi fold** (`kfold`) hoặc **chỉ trên train** (`final`) — không bao giờ fit trên phần dữ
liệu dùng để đánh giá.

### Vì sao chọn nhóm mô hình tuyến tính

1. **Đặc trưng TF-IDF thưa, nhiều chiều** (hàng chục nghìn chiều) — mô hình tuyến tính
   thường đạt hiệu quả ngang hoặc hơn mô hình phi tuyến phức tạp trên dữ liệu kiểu này.
2. **Số mẫu ít** (1.413–2.044 câu train tuỳ ngôn ngữ) — mô hình phức tạp dễ overfit.
3. **Hỗ trợ multi-label tự nhiên qua One-vs-Rest**, dễ `class_weight='balanced'` cho nhãn
   mất cân bằng.
4. **Baseline chính thức NLBSE'23** cũng dùng mô hình nhẹ (Random Forest + TF-IDF).
5. **Thực nghiệm xác nhận — cả 3 ngôn ngữ đều cho cùng 1 kết luận** (xem Mục "Kết quả"):
   Random Forest thắng CV ở Python nhưng thua rõ ở test cả 3 ngôn ngữ (Python: 0.542 vs
   0.567–0.570; Java: 0.636 — thấp nhất trong 5 model thật; chỉ hoà/nhỉnh ở Pharo). Mô
   hình tuyến tính ổn định hơn, generalize tốt hơn trên cả 3 ngôn ngữ — không phải chỉ
   đúng tình cờ ở 1 ngôn ngữ.

## Kết quả

### Giai đoạn 1 — CV micro-F1 trên 100% dữ liệu (sàng lọc phương pháp)

| Ngôn ngữ | Xếp hạng (CV micro-F1, mean ± std) |
|---|---|
| **Python** | ovr_random_forest **0.5882±0.0126** > ovr_logreg 0.5833±0.0126 > ovr_sgd 0.5811±0.0093 > ovr_linear_svc 0.5730±0.0095 > classifier_chain_logreg 0.5623±0.0193 |
| **Java** | ovr_logreg **0.5972±0.0566** > ovr_sgd 0.5932±0.0556 > ovr_linear_svc 0.5811±0.0577 > classifier_chain_logreg 0.5789±0.0675 > ovr_random_forest 0.5726±0.0405 |
| **Pharo** | ovr_sgd **0.6486±0.0188** > ovr_random_forest 0.6458±0.0226 > ovr_logreg 0.6438±0.0223 > ovr_linear_svc 0.6372±0.0210 > classifier_chain_logreg 0.6209±0.0247 |

(`dummy_most_frequent` không đưa vào bảng xếp hạng — luôn bằng 0.0000, xem lý do ở mục
"Model". Vẫn chạy và có trong `metrics.json`/`results/training/.../dummy_most_frequent/`,
chỉ ẩn khỏi leaderboard vì không có giá trị so sánh.)

### Giai đoạn 2 — test (held-out, 1 lần, 80/10/10)

| Ngôn ngữ | Test micro-F1 (xếp hạng) |
|---|---|
| **Python** | ovr_linear_svc **0.5704** > ovr_sgd 0.5690 > ovr_logreg 0.5675 > classifier_chain_logreg 0.5577 > ovr_random_forest 0.5421 |
| **Java** | classifier_chain_logreg **0.6964** > ovr_sgd 0.6950 > ovr_logreg 0.6871 > ovr_linear_svc 0.6838 > ovr_random_forest 0.6360 |
| **Pharo** | ovr_random_forest **0.6447** > ovr_sgd 0.6420 > ovr_logreg 0.6400 > ovr_linear_svc 0.6308 > classifier_chain_logreg 0.6161 |

Số liệu đầy đủ (per-label, AP, Hamming, thời gian...): `results/training/{kfold,final}/*/*/*/metrics.json`.

### CV không dự đoán đúng thứ hạng test — đúng ở cả 3 ngôn ngữ

| Ngôn ngữ | Model CV cao nhất | Thứ hạng test của model đó |
|---|---|---|
| Python | ovr_random_forest (hạng 1 CV) | **hạng 5/5 trên test** (thấp nhất) |
| Java | ovr_logreg (hạng 1 CV) | hạng 3/5 trên test (classifier_chain/sgd vượt lên) |
| Pharo | ovr_sgd (hạng 1 CV) | hạng 2/5 trên test (random_forest vượt lên) |

Không ngôn ngữ nào giữ nguyên thứ hạng CV sang test — nhất quán với hiện tượng tương tự
phía RoBERTa (`report.pdf`, Mục 6: E11 CV cao nhất Python nhưng test chỉ ngang E1). Bài
học chung cho cả nhóm: **không chọn model cuối cùng chỉ dựa vào CV**, luôn cần test độc
lập (dù biết test ở đây chưa hoàn toàn độc lập — xem `evaluation_note`).

## Hạn chế

- `validation` ở stage `final` chỉ chẩn đoán, không dùng để chọn hyperparameter — khác
  với DL (dùng validation chọn checkpoint/threshold). Nếu muốn so sánh chặt hơn, có thể
  thêm bước grid-search hyperparameter trên validation ở lần sau.
- 1 seed (42) — chưa lặp nhiều seed để kiểm định ý nghĩa thống kê của chênh lệch nhỏ.
- `ovr_random_forest` tái lập tinh thần baseline NLBSE'23 gốc nhưng không phải cấu hình
  chính thức (số cây, tiền xử lý có thể khác).
- Test giai đoạn 2 không độc lập hoàn toàn với việc chọn phương pháp (xem
  `evaluation_note` — giai đoạn 1 đã thấy toàn bộ dữ liệu, bao gồm cả câu sau này ở test).

## Cấu trúc module

| File | Vai trò |
|---|---|
| `data.py` | `load(language, stage)` — wrapper mỏng quanh `src.data.two_stage.load_stage_data` |
| `scoring.py` | `predict_scores()` — lấy điểm số liên tục cho AP, fallback `predict_proba` → `decision_function` |
| `metrics.py` | `compute_metrics()` — khớp schema `src.evaluation.multilabel` |
| `features.py` | `build_vectorizer()` — TF-IDF word+char |
| `models.py` | `build_models(seed)` — 6 model, `MODEL_KEYS` |
| `legacy/` | Script cũ dùng sai cột nhãn (`category` thay vì `instance_type`) — chỉ giữ tham khảo |

`experiments/ml/staged_experiment.py` chứa logic chạy 1 (method, language, stage); 
`experiments/ml/train_baseline.py` là CLI lặp qua nhiều method/language.

# E13–E17: nâng cấp E11 CodeBERT

Các số E13–E17 hiện chỉ các thử nghiệm CodeBERT dưới đây. Mỗi biến thể khởi tạo
mới từ `microsoft/codebert-base`; không tiếp tục từ model đã học full train của E11.
Đây là các giả thuyết cần kiểm chứng, chưa có kết quả pretrained training.

| Exp | Phương pháp | Đối chứng | Thay đổi |
|---|---|---|---|
| E11 | CodeBERT + BCE | Baseline | LR encoder/head cùng `2e-5` |
| E13 | CodeBERT + Head LR | E11 | Encoder `2e-5`, head `1e-4` |
| E14 | CodeBERT + Layer-wise LR Decay (LLRD) | E13 | Thêm hệ số giảm LR theo tầng `0.9` |
| E15 | CodeBERT + R-Drop | E11 | Hai forward dropout, BCE trung bình + KL Bernoulli đối xứng, alpha `1` |
| E16 | CodeBERT + LLRD + R-Drop | E14 và E15 | Kết hợp hai hướng; LR và alpha như E14/E15 |
| E17 | CodeBERT + Focal Loss | E11 | Gamma `2`, không alpha weighting, LR chung `2e-5` |

E13 kiểm tra head ngẫu nhiên có cần học nhanh hơn backbone pretrained. E14 kiểm
tra giảm mức cập nhật ở các tầng thấp có giúp bảo toàn biểu diễn. Đây là cách
áp dụng ý tưởng discriminative fine-tuning của
[ULMFiT](https://aclanthology.org/P18-1031/) vào encoder CodeBERT, không phải tái lập
nguyên kiến trúc hoặc lịch training của ULMFiT. Giá trị LR/decay là cấu hình thử,
không phải cấu hình đã được chứng minh tối ưu cho bộ dữ liệu hiện tại.

E15 kiểm tra tính nhất quán giữa các dropout mask theo
[R-Drop](https://arxiv.org/abs/2106.14448). Vì bài toán đa nhãn, KL tính trên các
Bernoulli độc lập, không softmax giữa các nhãn. Khi đánh giá chỉ có một forward.
E17 dùng cùng công thức Focal Loss đã kiểm tra ở E3 để tách ảnh hưởng của loss
trên backbone CodeBERT.

## Protocol giữ chung

Quy trình chạy hiện tại: [training hai giai đoạn](two_stage_training.md). Các kết quả cũ dùng `--stage legacy`.

- Giai đoạn 1 dùng `data/k_fold_data/<language>/`: 5-fold trên toàn bộ dữ liệu; mỗi ngôn ngữ train riêng.
- Tokenizer/backbone CodeBERT, max length 128, đầu CLS + Linear.
- Batch 16/32, AdamW, weight decay 0.01, warmup 10%, linear scheduler, clipping 1.
- Tối đa 10 epoch, early stopping patience 2 theo validation micro F1, threshold 0.5.
- Mỗi fold khởi tạo lại pretrained; seed fold bằng seed cơ sở cộng fold index.
- Giai đoạn 2 dùng `data/full_training_data/<language>/`: khởi tạo mới, train 80%, chọn best epoch bằng validation 10%, chấm test 10%.
- Xếp hạng bằng **CV micro F1** trong cùng ngôn ngữ/fingerprint/folds/seed. Test chỉ
  báo cáo sau train giai đoạn 2. Test này đã tham gia lựa chọn phương pháp ở giai đoạn 1 nên không độc lập.

Với CodeBERT 12 tầng, E14/E16 dùng:

```text
LR(head)        = 1e-4
LR(layer i)     = 2e-5 * 0.9**(11-i), i = 0..11
LR(embeddings)  = 2e-5 * 0.9**12
```

Tầng cuối có LR `2e-5`, embeddings khoảng `5.65e-6`. Scheduler nhân cùng một hệ số
warmup/decay vào mọi group nên giữ tỷ lệ LR. E13 dùng decay 1. Weight decay áp dụng
như E11, kể cả bias/LayerNorm, để tách riêng tác động LR. Code E1–E12 tiếp tục dùng
AdamW một group nếu không khai báo các tùy chọn LR mới.

LR thực tế ban đầu và số tham số mỗi group lưu ở `folds[*].optimizer` và
`training.optimizer` ở giai đoạn 2 (`final_training.optimizer` cho kết quả legacy) trong `metrics.json`; LR theo bước có trong TensorBoard.
Cấu hình và seed thực dùng được lưu ở `config.yaml`; backbone lưu ở `pretrained_backbone`.

## Chạy training

Từ thư mục gốc repo, dùng môi trường Python hiện có PyTorch và dependencies.
Không cần thêm thư viện hoặc tải một loại backbone mới so với E11.

```bash
# Xem cấu hình và kiểm tra lệnh trước.
python3 -m experiments.dl.model_zoo list
bash scripts/train_experiments.sh --experiments 13,14,15,16,17 \
  --language all --seed 42 --device cuda:0 --dry-run

# GPU 0: learning rate và focal loss, cả ba ngôn ngữ.
bash scripts/train_experiments.sh --experiments 13,14,17 \
  --language all --seed 42 --device cuda:0 \
  --output-dir results/training/codebert_lr_focal_seed42

# GPU 1: hai cấu hình R-Drop, cả ba ngôn ngữ.
bash scripts/train_experiments.sh --experiments 15,16 \
  --language all --seed 42 --device cuda:1 \
  --output-dir results/training/codebert_rdrop_seed42
```

Nếu có một GPU, chạy toàn bộ bằng một lệnh:

```bash
bash scripts/train_experiments.sh --experiments 13,14,15,16,17 \
  --language all --seed 42 --device cuda:0 \
  --output-dir results/training/codebert_upgrades_seed42
```

Mỗi lệnh gồm 15 pipeline exp/ngôn ngữ nếu chọn đủ 5 exp × 3 ngôn ngữ; mỗi pipeline
chạy 5 fold và một lượt full train. Hai nhóm GPU ở trên lần lượt là 9 và 6 pipeline.
R-Drop cần hai forward khi training nên dự kiến tốn thời gian hơn BCE; cần đo bằng
các run thực tế. Chưa tự động train hoặc sửa 36 kết quả E1–E12 đã giữ lại.
Dùng thư mục output mới cho mỗi lượt vì chạy lại cùng đường dẫn sẽ ghi đè checkpoint.

Chạy riêng một cấu hình hoặc dùng CLI chung:

```bash
python3 -m experiments.dl.train_exp14 --language python --device cuda:0 \
  --seed 42 --output-dir results/training/codebert_single_seed42/exp14/python
python3 -m experiments.dl.model_zoo train --exp E16 --language java --device cuda:0 \
  --seed 42 --output-dir results/training/codebert_single_seed42/exp16/java
```

## So sánh và xác nhận ứng viên

Mỗi script Bash tự sinh `model_zoo/report.md` và CSV trong thư mục output của nó.
Để so cùng E11 hiện tại và các lượt mới, tổng hợp các root đã chạy, ví dụ với hai GPU:

```bash
python3 -m src.evaluation.analyze_results \
  --results results/training/dl \
    results/training/codebert_lr_focal_seed42 \
    results/training/codebert_rdrop_seed42 \
  --output-dir results/codebert_comparison
```

Với lệnh một GPU, thay hai root mới bằng `results/training/codebert_upgrades_seed42`.
Báo cáo sắp theo CV micro F1 trong cùng cohort; `leaderboard.csv` giữ từng run riêng,
không tự chọn seed hoặc lấy test cao nhất. Báo cáo hiện có E1–E12 vẫn là snapshot cũ;
chỉ thêm số liệu mới sau khi các run hoàn tất.

Sau lượt seed 42, chọn 1–2 ứng viên theo CV rồi chạy lại **cả E11 và các ứng viên**
với seed 123 và 2026, dùng các thư mục mới và cùng folds. So sánh trung bình CV trên
cùng ba seed; không chỉ giữ seed cao nhất. Ví dụ nếu E14 là một ứng viên:

```bash
bash scripts/train_experiments.sh --experiments 11,14 --language all \
  --seed 123 --device cuda:0 --output-dir results/training/codebert_confirm_seed123
bash scripts/train_experiments.sh --experiments 11,14 --language all \
  --seed 2026 --device cuda:0 --output-dir results/training/codebert_confirm_seed2026
```

Vì test hiện tại đã được quan sát nhiều lần, kết luận cải thiện vẫn mang tính thăm dò.
Khi có thể, xác nhận ứng viên chốt bằng holdout mới theo project/class. Không tăng
`max_length` mặc định: EDA hiện chỉ có 5 câu Java vượt 128 token, các ngôn ngữ còn lại
không bị truncation ở mức này.

Nếu E13/E14 cho tín hiệu tốt, thử tiếp lưới nhỏ trên validation bằng bản sao YAML:
encoder LR `{1e-5, 2e-5, 3e-5}`, head LR `{5e-5, 1e-4}`, layer decay `{0.9, 0.95}`.
Đây là giai đoạn tinh chỉnh sau ablation; không cần chạy tất cả tổ hợp ngay từ đầu.

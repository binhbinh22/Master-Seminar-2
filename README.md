# Master-Seminar-2

## Model zoo RoBERTa / CodeBERT với dữ liệu processed

### Model Zoo: danh mục, training và so sánh kết quả

Registry dùng chung nằm trong `src/models/dl/zoo.py`; mỗi E1–E12 có tên kiến trúc,
variant và config. Có thể tiếp tục dùng `train_exp1`–`train_exp12` hoặc dùng CLI chung:

```bash
# Xem các model/config được hỗ trợ (không tải model)
python3 -m experiments.dl.model_zoo list

# Train một model
python3 -m experiments.dl.model_zoo train --exp E3 --language python --device cuda:0

# Tổng hợp các lần training đã hoàn tất, kể cả các thư mục chạy trên hai GPU
python3 -m src.evaluation.analyze_results --results results/training --output-dir results/model_zoo
```

Phân tích chỉ đọc `metrics.json`, không tải checkpoint và không train lại. Có thể
truyền nhiều thư mục sau `--results`, hoặc truyền trực tiếp các file metrics.
Script Bash tự chạy phân tích sau khi toàn bộ các lượt được chọn hoàn tất thành công,
lưu báo cáo vào `<output-root>/model_zoo/`. Lệnh phân tích riêng ở trên dùng để
tổng hợp thêm kết quả từ nhiều lần gọi script, ví dụ các nhóm exp chạy trên hai GPU.
Kết quả gồm:

- `report.md`: danh mục model, số run, bảng so sánh, liên kết tới metrics và biểu đồ.
- `leaderboard.csv`: CV mean/std, test metrics, seed, device, epoch, thời gian,
  đường dẫn checkpoint/threshold và trạng thái tồn tại checkpoint.
- `per_label.csv`: precision, recall, F1, AP và support theo nhãn cho mỗi run.
- `<language>-<cohort>/comparison.png`: CV micro/macro F1 ± SD và test F1.
- `<language>-<cohort>/per_label_f1.png`: heatmap test F1 của các model theo nhãn.

Các run chỉ được so sánh chung khi cùng ngôn ngữ, hash nội dung train/test/folds/labels,
thứ tự nhãn và split protocol. Run cũ thiếu hash được để riêng. Mỗi run/seed là một
dòng, không tự gộp hoặc chọn run có test cao nhất. Bảng sắp theo CV micro F1;
CV của E4 vẫn ở threshold ban đầu, test dùng threshold đã tuning từ OOF.
Nếu chưa có training hoàn tất, công cụ báo chưa tìm thấy kết quả, không sinh số liệu mẫu.

Trong terminal, training có `tqdm` theo batch với loss hiện tại, loss trung bình,
learning rate, tốc độ và thời gian ước tính; validation/test cũng có tiến độ.
Script Bash vẫn hiển thị tiến độ khi dùng `tee`. Khi chuyển hướng ra file không có
terminal, mặc định chỉ ghi log theo epoch. Dùng `MODEL_ZOO_PROGRESS=on` để luôn
hiện thanh tiến độ, `MODEL_ZOO_PROGRESS=off` để tắt; hoặc `training.progress: false`
trong config để tắt cho cả train và evaluation.

Sau mỗi exp/ngôn ngữ, đồ thị từng run tự lưu trong `plots/`:
`training_curves.png`, `cross_validation.png`, `test_summary.png`, `test_per_label.png`.
Hai biểu đồ test chỉ có khi bật đánh giá test. Metrics được lưu trước khi vẽ hình.

Chạy từ thư mục gốc repository. Cài dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Mọi exp đọc `data/processed/<language>/{train.csv,test.csv,folds.json,label_cols.json}`.
Không chia lại dữ liệu, không đọc partition raw. Mỗi ngôn ngữ được train riêng:
Python có 5 nhãn, Java và Pharo có 7 nhãn, theo đúng thứ tự `label_cols.json`.
Loader kiểm tra schema, nhãn nhị phân, ID, nhóm câu trùng, train/test và độ phủ CV.
Tạo dữ liệu trước bằng `notebooks/data_preprocessing.ipynb`; xem `data/README.md`.

| Exp | Mô hình | Config |
|---|---|---|
| E1 | RoBERTa CLS + BCE | `configs/exp1_roberta_baseline.yaml` |
| E2 | RoBERTa CLS + Weighted BCE | `configs/exp2_roberta_weighted_bce.yaml` |
| E3 | RoBERTa CLS + Focal Loss | `configs/exp3_roberta_focal.yaml` |
| E4 | RoBERTa CLS + BCE + threshold tuning | `configs/exp4_roberta_threshold_tuning.yaml` |
| E5 | RoBERTa + Label-Aware Attention + BCE | `configs/exp5_roberta_label_attention.yaml` |
| E6 | RoBERTa + Label-Aware Attention + BCE + Contrastive Loss | `configs/exp6_roberta_label_attention_contrastive.yaml` |
| E7 | RoBERTa + Asymmetric Loss | `configs/exp7_roberta_asl.yaml` |
| E8 | RoBERTa + R-Drop | `configs/exp8_roberta_rdrop.yaml` |
| E9 | RoBERTa + Label Correlation GNN theo từng ngôn ngữ (5/7/7 nhãn) | `configs/exp9_roberta_label_gnn.yaml` |
| E10 | RoBERTa + Adversarial Training FGM | `configs/exp10_roberta_fgm.yaml` |
| E11 | CodeBERT + BCE | `configs/exp11_codebert_bce.yaml` |
| E12 | RoBERTa + Label-Aware Attention + Label Correlation | `configs/exp12_roberta_label_attention_gnn.yaml` |


Chi tiết công thức, tham số và nguồn tham khảo cho E7–E12: [docs/experiments_e7_e12.md](docs/experiments_e7_e12.md).
E9 được chốt dùng graph theo từng ngôn ngữ, tự đọc số node/thứ tự nhãn từ
`label_cols.json`: Python 5, Java 7, Pharo 7. E12 dùng cùng hệ nhãn này.
Graph chỉ tính từ fold train; `label_graph.json` lưu graph full train và thông tin node/cạnh.

Ví dụ chạy E1 cho Python, E6 cho Java hoặc E2 cho cả ba ngôn ngữ:

```bash
python3 -m experiments.dl.train_exp1 --language python
python3 -m experiments.dl.train_exp6 --language java
python3 -m experiments.dl.train_exp2 --language all
```

Tương tự, dùng `train_exp3`, `train_exp4`, `train_exp5` cho các exp còn lại.
Có thể chạy tuần tự bằng Bash:

```bash
# Kiểm tra lệnh trước khi chạy
bash scripts/train_experiments.sh --dry-run

# E1–E12 trên Python
bash scripts/train_experiments.sh --language python

# Chọn exp và ngôn ngữ
bash scripts/train_experiments.sh --experiments 1,2,6 --language java

# E1–E12 trên cả 3 ngôn ngữ (36 lượt chạy)
bash scripts/train_experiments.sh
```

Script dùng Python của môi trường hiện tại; có thể chỉ định `--python /path/to/python`.
Kết quả mặc định nằm trong `results/training/<timestamp>-<pid>/exp<N>/<language>/`,
có `training.log` cho từng lượt. Dùng `--output-dir PATH` để chọn thư mục gốc khác.
Script dừng khi có lỗi; `--dry-run` không chạy training hoặc tạo output.

Mặc định chạy Python. Có thể dùng `--config PATH`, `--data-dir data/processed`
và `--output-dir PATH`. Với `--language all`, output tùy chỉnh được thêm thư mục ngôn ngữ.
Lần chạy đầu tải backbone/tokenizer `roberta-base` từ Hugging Face.

### Chọn CPU hoặc GPU

`--device auto` dùng GPU 0 nếu CUDA khả dụng, ngược lại dùng CPU.
`--device cpu`, `--device cuda:0`, `--device cuda:1` chọn thiết bị cụ thể.
Chọn GPU khi CUDA không khả dụng hoặc index không tồn tại sẽ báo lỗi trước khi
tải model. Thiết bị được in ra terminal và lưu trong config/metrics của run.
Với lệnh Python trực tiếp, mặc định lấy `training.device` trong YAML (`auto`);
script Bash mặc định truyền `auto`, hoặc giá trị bạn chỉ định qua `--device`.

Kiểm tra trong chính môi trường Python định dùng:

```bash
python3 -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPUs:", torch.cuda.device_count())'

# Một exp trên GPU 1
python3 -m experiments.dl.train_exp1 --language python --device cuda:1

# E1–E12 trên GPU 0
bash scripts/train_experiments.sh --language python --device cuda:0

# Dùng Python từ môi trường có PyTorch hỗ trợ CUDA
bash scripts/train_experiments.sh --python /path/to/environment/bin/python --device cuda:0
```

Nếu phiên bản PyTorch có hậu tố `+cpu` và CUDA là `False`, cần dùng môi trường
có PyTorch hỗ trợ CUDA; cờ `--device` không thể bổ sung CUDA cho bản CPU.

Để tận dụng hai GPU, mở hai terminal và phân exp khác nhau:

```bash
# Terminal 1: E1–E3, cả 3 ngôn ngữ, GPU 0
bash scripts/train_experiments.sh --experiments 1,2,3 --language all --device cuda:0

# Terminal 2: E4–E6, cả 3 ngôn ngữ, GPU 1
bash scripts/train_experiments.sh --experiments 4,5,6 --language all --device cuda:1
```

Mỗi terminal chạy tuần tự trên một GPU; hai terminal chạy song song. Mỗi model
chỉ dùng một GPU, chưa dùng DDP. Theo dõi bằng `watch -n 1 nvidia-smi`.
Index CUDA tính theo các GPU đang hiển thị cho process: nếu đặt
`CUDA_VISIBLE_DEVICES=1`, card vật lý 1 sẽ trở thành `cuda:0` trong process đó.

### Chạy các exp bổ sung E7–E12

```bash
# GPU 0: E7–E9
bash scripts/train_experiments.sh --experiments 7,8,9 --language all --device cuda:0

# GPU 1: E10–E12
bash scripts/train_experiments.sh --experiments 10,11,12 --language all --device cuda:1
```

Các exp mới có tối đa 10 epoch; mặc định patience 2, riêng E9 v2 patience 3,
chạy tối thiểu 5 epoch trước khi cho phép early stop và dùng AP phân xử khi F1 bằng nhau.
GNN ước lượng graph riêng từ
fold train; R-Drop dùng hai forward khi train; FGM nhiễu input embeddings; E11
nạp đúng tokenizer/backbone `microsoft/codebert-base`. Chi tiết tại tài liệu ở trên.

### Quy trình chung

Mỗi fold khởi tạo lại model từ pretrained backbone, tokenize (padding/truncation,
`max_length=128` mặc định), train và chọn epoch theo validation micro F1.
Giới hạn mặc định là **10 epoch/fold**, early stopping patience **2**;
E9 v2 dùng patience **3**, tối thiểu **5 epoch**, AP phân xử khi F1 bằng nhau.
Model được khôi phục về epoch tốt nhất. Sau CV, khởi tạo model mới và train trên
full train với số epoch bằng trung vị best epoch của các fold, rồi đánh giá test.

E9 v2 khởi tạo từ head CLS baseline, nhánh GCN khởi tạo hiệu chỉnh bằng 0 và
không chia logit cho `sqrt(768)`. Phải train lại E9 từ pretrained, không nạp checkpoint
head cũ. Có thể kiểm tra riêng epoch đầu, giữ nguyên LR schedule 10 epoch:

```bash
python3 -m experiments.dl.check_exp9_first_epoch --language all --fold 0 --device cuda:0
```

Script lưu diagnostic vào `results/diagnostics/`, không chạy test hoặc toàn bộ CV.
Seed, optimizer, lịch learning rate và folds giống nhau giữa các exp theo mặc định.
Một exp/ngôn ngữ chạy 5 model CV và 1 model cuối. Không dùng test để chọn cấu hình.

- **E2:** `pos_weight = negative / positive`, tính lại trên train của từng fold;
  model cuối tính từ full train. Nhãn không có positive hoặc negative dùng weight 1.
- **E3:** sigmoid focal loss `alpha_t * (1-p_t)^gamma * BCE`, mặc định `gamma=2`,
  `alpha=null` (không alpha weighting). Có thể chỉnh trong config.
- **E4:** chọn threshold riêng cho từng nhãn bằng F1 trên dự đoán out-of-fold của
  full train, grid 0.05–0.95 bước 0.05. Hòa điểm ưu tiên gần 0.5. Sau đó cố định
  threshold để đánh giá model cuối trên test. CV metrics vẫn dùng threshold 0.5;
  không coi F1 trên OOF đã tuning là ước lượng độc lập. Threshold là hậu xử lý,
  phải nạp `thresholds.json` khi inference; riêng state dict không chứa threshold.
- **E5:** mỗi nhãn có query học được, attention trên các token không padding,
  tạo context và logit riêng cho từng nhãn; tối ưu BCE.
- **E6:** cùng kiến trúc attention, thêm phép chiếu CLS. Contrastive loss dùng
  cosine similarity giữa CLS đã chiếu và query nhãn, chia temperature, rồi
  cross-entropy với target phân bố đều trên các nhãn dương của câu. Các nhãn còn
  lại là negative. `loss = BCE + 0.1 * contrastive`, temperature mặc định 0.1.
  Đây là biến thể sentence–label contrastive; không phải pairwise sentence contrastive.

### Kết quả và sử dụng lại model

Output: `results/<experiment>/<language>/`, gồm:

- `best_model.pt`: state dict model cuối train full train.
- `config.yaml`, `dataset_config.yaml`, `label_cols.json`, `folds.json`.
- `tokenizer/`, `thresholds.json` (thứ tự theo label_cols).
- `metrics.json`: chỉ số từng fold, mean/sample SD, best epoch, test,
  class weights cho E2 và thresholds cho E4.
- `oof_predictions.pt` riêng E4, chứa xác suất và nhãn theo thứ tự train.csv.
- `tensorboard/` và `plots/`: loss, validation/test metrics, biểu đồ tổng hợp.

Khởi tạo đúng kiến trúc theo `variant` trong config, nạp state dict, tokenizer và
thứ tự nhãn. Dự đoán bằng `sigmoid(logits) >= thresholds`; mọi exp trừ E4 mặc định 0.5.
Một thư mục output chứa một kết quả cuối; chạy lại cùng đường dẫn sẽ ghi đè model
và metrics (TensorBoard có thư mục theo thời gian). Dùng `--output-dir` cho run riêng.

```bash
tensorboard --logdir results
```

Metrics chung: micro/macro precision, recall, F1, average precision, Hamming loss
và kết quả theo nhãn. Average precision không xác định được ghi `null`.

### Báo cáo thực nghiệm

[report.pdf](report.pdf) trình bày bài toán, dữ liệu, tiền xử lý, E1–E12,
kết quả của 36 run, so sánh với RoBERTa, phân tích theo nhãn, hội tụ,
chi phí và giới hạn thực nghiệm. Báo cáo có bảng biểu, hình minh họa và
tài liệu tham khảo; không chèn source code. Mục 2 (trang 6–16) trình bày
EDA với 10 hình về độ dài/token, truncation, cardinality, từ vựng, quan hệ
nhãn, train/test, các fold và nguy cơ leakage, trước phần dữ liệu và tiền xử lý
ở Mục 3. Chỉ giữ bản PDF cuối cùng.

### Kiểm tra offline

```bash
USE_TF=0 python3 -m unittest discover -s tests -v
```

Các kiểm tra dùng backbone RoBERTa nhỏ khởi tạo ngẫu nhiên để kiểm tra đường chạy,
không tải pretrained weights và không đánh giá chất lượng mô hình.

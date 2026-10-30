# E7–E12: phương pháp và cấu hình

Các exp mới dùng `data/processed`, cùng folds, tokenizer length, optimizer,
giới hạn **10 epoch**, early stopping patience **2** như E1–E6, trừ E9 v2
dùng patience **3**, chạy tối thiểu **5 epoch** trước khi cho phép dừng sớm.
Ngưỡng dự đoán là 0.5. Không tự kết hợp các phương pháp mới với Weighted BCE,
contrastive loss hoặc threshold tuning của những exp khác.

## Phạm vi nhãn của E9/E12

Pipeline hiện train riêng từng ngôn ngữ: Python 5 nhãn, Java 7, Pharo 7.
Vì vậy E9/E12 tạo graph tương ứng 5/7/7 node để so sánh cùng protocol.
**E9 được chốt là GNN theo từng ngôn ngữ**, với `dataset.label_space: per_language`.
Số node và thứ tự nhãn lấy từ `label_cols.json`, không khai báo cứng 19 đầu ra.
Tổng 19 tên nhãn trên ba ngôn ngữ không đồng nghĩa với 19 nhãn cùng được quan sát
trên mỗi câu. Không cần sửa raw hoặc tạo bộ processed mới để chạy E9.

## E7 — RoBERTa + Asymmetric Loss

ASL dùng hệ số tập trung khác nhau cho positive/negative và clipping xác suất
negative. Ý tưởng và cấu hình gốc tham khảo [ASL chính thức](https://github.com/Alibaba-MIIL/ASL).

Với xác suất positive `p = sigmoid(z)`, đặt `q = min(1, 1-p+clip)`.
Loss theo một nhãn:

```text
L = - y * (1-p)^gamma_pos * log(p)
    - (1-y) * (1-q)^gamma_neg * log(q)
```

Mặc định `gamma_pos=1`, `gamma_neg=4`, `clip=0.05`. Negative rất dễ có
`q=1`, nên không đóng góp loss. `asl_detach_focal=true` dừng gradient qua
trọng số focusing, không thay đổi trạng thái autograd toàn cục. Loss lấy trung
bình trên các cặp câu–nhãn như các exp khác; đây là khác biệt reduction so với
implementation tham khảo dùng tổng. Dùng log-sigmoid để giữ ổn định với logit lớn.

## E8 — RoBERTa + R-Drop

[R-Drop](https://arxiv.org/abs/2106.14448) ràng buộc nhất quán giữa hai lần
forward có dropout. Với multi-label, dùng **KL Bernoulli cho từng nhãn**,
không softmax trên tập nhãn:

```text
L = 0.5 * (BCE(z1,y) + BCE(z2,y))
    + rdrop_alpha * 0.5 * [KL(Bern(p1)||Bern(p2)) + KL(Bern(p2)||Bern(p1))]
```

KL lấy trung bình trên batch/nhãn hợp lệ; mặc định `rdrop_alpha=1`.
Hai forward chỉ áp dụng khi training có nhãn; evaluation/inference chỉ một forward.
Dropout vẫn dùng các lớp trong backbone; không tắt dropout hoặc tái đặt seed
giữa hai forward. Hệ số KL là siêu tham số cần chọn bằng validation.

## E9 — RoBERTa + Label Correlation GNN

Đầu vào: `data/processed/<language>/train.csv`, `test.csv`, `folds.json`,
`label_cols.json`. E9 chạy ba model riêng: Python với graph 5×5, Java 7×7 và
Pharo 7×7. Giữ train/test/folds chung với các exp còn lại. Builder từ chối
graph trộn nhiều ngôn ngữ, hàng test hoặc nhãn chưa được quan sát đầy đủ.

Đây là biến thể GCN nhãn cho đề tài, lấy cảm hứng từ hướng mô hình hóa quan hệ
nhãn bằng graph trong [ML-GCN](https://arxiv.org/abs/1904.03582), không phải bản
tái lập nguyên kiến trúc cho ảnh hoặc dùng word embeddings tên nhãn như bài gốc.

Graph chỉ được ước lượng từ **fold train**. Với ma trận nhãn `Y`, tính
co-occurrence `M=Y.T@Y`, support `s_c=sum(Y[:,c])` và trọng số
`A_cd=M_cd/sqrt(s_c*s_d)`. Loại các cạnh ngoài đường chéo nhỏ hơn
`graph_threshold=0.1`, thêm self-loop rồi chuẩn hóa đối xứng:

```text
A_hat = D^(-1/2) * (A_offdiag + I) * D^(-1/2)
H1 = ReLU(Linear1(A_hat @ E))
H2 = E + Linear2(A_hat @ H1)
logits = CLS @ H2.T + bias
```

`E` là embedding nhãn học được, không cần embedding bên ngoài. GCN có hai lớp,
hidden size 128, residual path và tối ưu cùng backbone bằng BCE. Adjacency là
thống kê cố định trong một lượt train; tham số GCN/embedding được học end-to-end.
Nhãn chưa có positive vẫn có self-loop, tránh chia cho 0. Model cuối tính lại
graph trên full train. Graph lưu trong state dict và `metrics.json` theo fold.
Không xây graph từ toàn bộ train trước CV hoặc từ validation/test.

### E9 v2: tránh dự đoán toàn âm và dừng sớm khi F1 còn bằng 0

Khởi tạo embedding nhãn `E` và bias từ đầu `Linear(768, C)` như E1. Trọng số
và bias lớp cuối nhánh GCN khởi tạo bằng 0, vì vậy ở bước khởi tạo `H2=E` và
logit E9 bằng logit baseline CLS. Graph học dần phần hiệu chỉnh, thay vì có
shared bias ngẫu nhiên lớn lấn át khác biệt giữa nhãn. Không chia logit phân
loại cho `sqrt(768)`; phép chia đó trước đây làm giảm gradient tín hiệu đầu ra.
E12 vẫn dùng khởi tạo GCN riêng như trước; thay đổi này chỉ dành cho E9.

Cấu hình E9 v2:

```yaml
model:
  graph_head: residual_classifier_v2
training:
  epochs: 10
  min_epochs: 5
  early_stopping_patience: 3
  validation_tiebreaker: micro_average_precision
```

F1 vẫn là tiêu chí chính. Chỉ khi F1 bằng nhau, chọn AP cao hơn; AP tốt hơn
không được ghi đè checkpoint có F1 cao hơn. Điều này cho phép ghi nhận tiến bộ
xếp hạng khi các xác suất vẫn chưa vượt 0.5. Model cuối vẫn lấy trung vị best
epoch từ CV. Ngưỡng 0.5, BCE, LR, batch size, folds và cách tạo adjacency không đổi.
Metrics có thêm số dự đoán dương, TP/FP/FN để phát hiện collapse.

Checkpoint cũ dùng head v1 không tương thích thang logit mới. Head v2 lưu
`head_version=2` trong state dict; strict loading checkpoint cũ sẽ báo thiếu
version thay vì âm thầm dùng trọng số sai. Cần train lại E9 từ pretrained vào
thư mục output mới; không lấy model đã train full train để chạy tiếp CV.

Kiểm tra epoch đầu trên fold train/validation thật, giữ scheduler theo toàn bộ
10 epoch và không dùng test:

```bash
conda activate seminar
python -m experiments.dl.check_exp9_first_epoch \
  --language all --fold 0 --device cuda:0
```

Đây chỉ là diagnostic một fold/ngôn ngữ. Script lưu `first_epoch_metrics.json`,
config, TensorBoard và `summary.json`/`report.md`, không tạo `metrics.json` của
một CV hoàn tất. Mặc định dùng thư mục riêng trong `results/diagnostics/`.
Nó trả exit code lỗi nếu còn micro F1 bằng 0 ở bất kỳ ngôn ngữ được kiểm tra.

Các cạnh thể hiện quan hệ **đồng xuất hiện positive**, không phải mọi quan hệ
ngữ nghĩa giữa tên nhãn. Với threshold 0.1 hiện tại, graph từ full train có
3 cạnh Python, 3 cạnh Java và 6 cạnh Pharo (không đếm self-loop). Một số nhãn
không có cạnh sang nhãn khác; vẫn giữ node/self-loop và residual để học riêng,
không ép thêm cạnh hoặc loại bỏ nhãn. Threshold này là cấu hình thử nghiệm,
chưa được chứng minh tối ưu bằng validation.

`metrics.json` lưu graph của từng fold và full train. `label_graph.json` lưu
riêng graph full train với `language`, `labels`, `num_nodes`, `train_sentences`,
`threshold`, `num_edges`, `isolated_labels` và adjacency đã chuẩn hóa. Thứ tự
node khớp chính xác với đầu ra và thứ tự nhãn processed. Adjacency cũng nằm trong
state dict; không tạo lại graph từ nhãn test khi inference.

Chạy E9 trên cả ba ngôn ngữ:

```bash
bash scripts/train_experiments.sh --experiments 9 --language all --device cuda:0
```

## E10 — RoBERTa + FGM

Phương pháp thêm nhiễu embedding tham khảo hướng adversarial training cho văn
bản của [Miyato và cộng sự](https://arxiv.org/abs/1605.07725). Biến thể ở đây
là FGM trên **input embeddings theo từng câu**, không sửa trực tiếp embedding
parameter, không phải PGD hay virtual adversarial training.

```text
g = gradient(BCE(clean_embeddings), clean_embeddings)
r = fgm_epsilon * g / max(||g||_2, 1e-12)
L = [BCE(clean) + adversarial_weight * BCE(clean_embeddings + stop_gradient(r))]
    / (1 + adversarial_weight)
```

Gradient ở vị trí padding bị đặt bằng 0 trước chuẩn hóa. Mặc định
`fgm_epsilon=1`, `adversarial_weight=1`. `autograd.grad` chỉ lấy gradient
embedding của batch hiện tại, không trộn gradient tham số đã tích lũy của các
batch trước. Optimizer chỉ cập nhật sau outer backward theo vòng training chung.
Evaluation/inference không thêm nhiễu. Chi phí training tăng do thêm forward và
bước lấy gradient embedding; chưa có số đo tốc độ để kết luận mức tăng cụ thể.

## E11 — CodeBERT + BCE

Dùng backbone và tokenizer `microsoft/codebert-base`, giữ đầu CLS + Linear và BCE
như E1. [Model card chính thức](https://huggingface.co/microsoft/codebert-base)
mô tả checkpoint thuộc kiến trúc RoBERTa, khởi tạo từ RoBERTa-base và tiếp tục
pretrain trên dữ liệu document/code. Vì vậy có thể dùng lớp `RobertaModel` hiện
có với đúng tên checkpoint. Không thay bằng checkpoint chỉ có tên gần giống.

Đầu vào vẫn chỉ là **câu comment** trong processed; E11 không tự thêm đoạn code.
Thí nghiệm này so sánh pretrained backbone dưới cùng đầu vào và cách chia.
Không mặc định CodeBERT tốt hơn RoBERTa, đặc biệt với ngôn ngữ chưa được đại diện
trong dữ liệu pretrain của checkpoint.

## E12 — Label-Aware Attention + Label Correlation

E12 tạo context token riêng từng nhãn như E5, sau đó dùng GCN trên các context:

```text
R = LabelAwareAttention(token_hidden_states)  # batch x labels x 768
R' = R + Linear2(A_hat @ ReLU(Linear1(A_hat @ R)))
logit_c = dot(R'_c, classifier_c) + bias_c
L = BCE(logits, labels)
```

Graph được tạo như E9, nhưng node features ở đây phụ thuộc từng câu thay vì chỉ
là embedding nhãn học được. E12 không tự cộng contrastive loss của E6.
So sánh E5–E12 để đánh giá phần bổ sung graph; E9–E12 để khảo sát thiết kế kết
hợp attention/context theo nhãn.

## Chạy và kiểm tra

```bash
python3 -m experiments.dl.model_zoo list
python3 -m experiments.dl.train_exp11 --language python --device cuda:0

# Terminal 1
bash scripts/train_experiments.sh --experiments 7,8,9 --language all --device cuda:0

# Terminal 2
bash scripts/train_experiments.sh --experiments 10,11,12 --language all --device cuda:1

# Chỉ xem lệnh, không train
bash scripts/train_experiments.sh --experiments 7,8,9,10,11,12 --language python --dry-run

USE_TF=0 python3 -m unittest discover -s tests -v
```

Mặc định `--experiments all` hiện bao gồm E1–E12: 36 lượt exp/ngôn ngữ. Mỗi
lượt dùng CV và full train như trước. Các kiểm tra offline dùng backbone nhỏ,
không tải CodeBERT/RoBERTa thật và không phải kết quả đánh giá chất lượng.

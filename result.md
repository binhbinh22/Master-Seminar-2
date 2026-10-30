# Kết quả thực nghiệm DL — Code Comment Classification

Tài liệu tổng hợp **36 run hoàn chỉnh: 12 phương pháp × 3 ngôn ngữ**, sử dụng cùng dữ liệu processed và cùng folds trong từng ngôn ngữ. Các bảng được trích từ metric đã lưu, không phải số liệu minh họa. Thời điểm tổng hợp kết quả: 30/10/2026.

Báo cáo đầy đủ, EDA và các hình phân tích: [report.pdf](report.pdf).

## 1. Bài toán và dữ liệu

Mỗi câu comment có thể mang nhiều nhãn đồng thời. Mô hình nhận **văn bản một câu**, sinh xác suất sigmoid cho từng nhãn; không nhận toàn bộ class hay đoạn code đi kèm. Mỗi ngôn ngữ có một mô hình và hệ nhãn riêng.

Nguồn: [NLBSE’23 Tool Competition](https://nlbse2023.github.io/tools/). Nhánh **Code Comment Classification** có 6.738 câu và baseline gốc Random Forest; nhánh **Issue Report Classification** là bài toán khác. Trong dự án này, **E1 là baseline RoBERTa được huấn luyện lại theo cùng protocol với các biến thể**, đáp ứng yêu cầu so sánh Transformer của học phần.

| Ngôn ngữ | Tổng câu | Train | Test | Nhãn | CV folds |
| --- | --- | --- | --- | --- | --- |
| Python | 2555 | 2044 | 511 | 5 | 5 |
| Java | 2418 | 1934 | 484 | 7 | 5 |
| Pharo | 1765 | 1411 | 354 | 7 | 5 |

Dữ liệu nằm tại `data/processed/<language>/`, gồm `train.csv`, `test.csv`, `folds.json`, `label_cols.json`. Mỗi ID chỉ có một dòng sau pivot từ dữ liệu long; thứ tự nhãn lấy từ `label_cols.json`.

Text chỉ chuẩn hóa khoảng trắng. Các câu có text giống hệt được giữ cùng group khi chia train/test và train/validation, nhằm tránh rò rỉ nội dung trùng chính xác. Không tự động xóa các ID có cùng text nhưng khác nhãn.

EDA trong báo cáo cho thấy nhiều negative, nhãn hiếm, text lặp và sự tập trung theo class. Chỉ 5 câu Java vượt 128 token (3 train, 2 test); Python và Pharo không có câu vượt giới hạn này. Grouping chưa xử lý hết near-duplicate; class vẫn có thể xuất hiện ở cả train và test.

## 2. Protocol đánh giá

- **Bộ chia:** train/test xấp xỉ 80/20; 5-fold CV trên train, với ràng buộc nhóm text và cân bằng đa nhãn gần đúng. Không chia lại giữa các exp.
- **Khởi tạo:** mỗi fold dùng mô hình mới từ pretrained; seed cơ sở 42, seed fold bằng 42 cộng chỉ số fold.
- **Chọn checkpoint:** validation micro F1. CV báo trung bình và sample SD của năm fold tại best epoch từng fold.
- **Full train:** khởi tạo mô hình mới và huấn luyện toàn bộ train với trung vị best epoch của năm fold. Test không chọn epoch.
- **Chọn phương pháp:** CV micro F1 là tiêu chí chính. Các cột test dùng để báo cáo và phân tích, không dùng để chọn lại ứng viên.
- **Metric:** micro/macro F1, precision/recall, average precision (AP), Hamming loss (HL), và metric từng nhãn. F1/AP càng cao càng tốt; HL càng thấp càng tốt.

Ngưỡng mặc định là 0.5. E4 chọn ngưỡng từng nhãn bằng F1 trên dự đoán out-of-fold của train, lưới 0.05–0.95, bước 0.05. **CV E4 vẫn được báo ở ngưỡng 0.5; test E4 dùng ngưỡng đã tuning.** Vì vậy cột CV chưa phải đánh giá độc lập hiệu quả của toàn bộ bước tuning.

Cấu hình chung: RoBERTa-base hoặc CodeBERT-base, hidden size 768, max length 128, AdamW, learning rate `2e-5`, weight decay `0.01`, warmup 10%, linear scheduler, batch train/eval 16/32, gradient clipping 1, mixed precision tắt, tối đa 10 epoch. Mặc định early stopping patience 2. Mỗi run gồm 5 lượt CV và 1 lượt full train; tổng cộng 216 lượt fine-tune.

**Ngoại lệ E9 v2:** GNN residual với khởi tạo đầu phân loại tương thích baseline; patience 3, tối thiểu 5 epoch trước khi cho phép dừng, dùng micro AP phân xử khi validation F1 bằng nhau. Các kết quả E9 dưới đây đều là v2. Khác biệt này phải được tính đến khi diễn giải lợi ích của graph.

## 3. Các phương pháp và ý nghĩa thử nghiệm

| Exp | Mô hình | Mục tiêu/đối chứng |
| --- | --- | --- |
| E1 | RoBERTa + BCE | Baseline Transformer cùng protocol. |
| E2 | RoBERTa + Weighted BCE | Tăng chi phí bỏ sót positive; so với E1. |
| E3 | RoBERTa + Focal Loss | Tập trung cặp câu–nhãn khó; so với E1. |
| E4 | RoBERTa + Threshold Tuning | Đo tác động của quyết định ngưỡng; so với E1. |
| E5 | RoBERTa + Label-Aware Attention | Biểu diễn context riêng theo nhãn; so với E1. |
| E6 | RoBERTa + Attention + Contrastive Loss | Kiểm tra term contrastive bổ sung; so với E5. |
| E7 | RoBERTa + Asymmetric Loss (ASL) | Giảm easy negatives với focusing bất đối xứng; so với E1/E2/E3. |
| E8 | RoBERTa + R-Drop | Nhất quán giữa hai forward dropout bằng KL Bernoulli; so với E1. |
| E9 | RoBERTa + Label Correlation GNN v2 | Học phần hiệu chỉnh từ graph nhãn; so với E1. |
| E10 | RoBERTa + Adversarial Training (FGM) | Thêm nhiễu embedding một bước; so với E1. |
| E11 | CodeBERT + BCE | So sánh pretrained backbone chuyên về code; so với E1. |
| E12 | RoBERTa + Attention + Label Correlation GNN | Kiểm tra graph bổ sung trên context attention; so với E5. |

Graph và positive weight chỉ được tính từ training subset của từng fold; model cuối tính lại từ full train. Graph có 5/7/7 node theo ngôn ngữ, không sử dụng nhãn test. E9 và E12 thay cả biểu diễn node/khởi tạo nên không phải ablation chỉ thay một thành phần.

## 4. Kết quả đầy đủ

Thang điểm từ 0 đến 1, làm tròn bốn chữ số. CV là **mean ± SD**, AP trong bảng là **macro AP**. Chữ đậm chỉ giá trị tốt nhất từng cột trong cùng ngôn ngữ, không phải bằng chứng về ý nghĩa thống kê hoặc quy tắc chọn model bằng test.

Các biểu đồ bên dưới dùng cùng 36 run với bảng số liệu. Trục mô hình được sắp theo CV micro F1 giảm dần; `Rxxx` là mã run trong bản tổng hợp. Biểu đồ cột bên trái hiển thị CV mean ± SD giữa fold, bên phải hiển thị test ở ngưỡng đã lưu. Thanh sai số không phải khoảng tin cậy; riêng E4, CV ở ngưỡng 0.5 còn test ở ngưỡng OOF-tuned. Heatmap hiển thị F1 test từng nhãn, thang màu cố định 0–1.

### Python

| Exp | CV micro F1 ± SD | Test micro F1 | Test macro F1 | Test macro AP | Test HL ↓ |
| --- | --- | --- | --- | --- | --- |
| E1 | 0.5747 ± 0.0273 | 0.6466 | 0.5755 | 0.6624 | 0.1472 |
| E2 | 0.5841 ± 0.0257 | 0.6039 | 0.5633 | 0.6407 | 0.1910 |
| E3 | 0.5885 ± 0.0263 | 0.6149 | 0.5381 | 0.6352 | 0.1554 |
| E4 | 0.5747 ± 0.0273 | 0.6196 | 0.5838 | 0.6624 | 0.1793 |
| E5 | 0.5906 ± 0.0307 | 0.6217 | 0.5279 | 0.6476 | 0.1558 |
| E6 | 0.5894 ± 0.0213 | 0.6491 | 0.5781 | 0.6626 | 0.1456 |
| E7 | 0.5750 ± 0.0232 | 0.6187 | **0.5885** | 0.6755 | 0.2137 |
| E8 | 0.5841 ± 0.0315 | 0.6539 | 0.5746 | 0.6653 | **0.1413** |
| E9 | 0.5874 ± 0.0290 | **0.6591** | 0.5867 | 0.6688 | **0.1413** |
| E10 | 0.5901 ± 0.0256 | 0.6337 | 0.5537 | 0.6487 | 0.1479 |
| E11 | **0.6410 ± 0.0047** | 0.6467 | 0.5627 | 0.6655 | 0.1479 |
| E12 | 0.5957 ± 0.0340 | 0.6399 | 0.5701 | **0.6763** | 0.1542 |

![So sánh CV và test micro/macro F1 của E1–E12 trên Python](docs/images/results/python_comparison.png)

**Hình 1 — Python, so sánh tổng thể.** E11 dẫn CV nhưng test micro gần ngang E1; E9 có test micro cao nhất. Thứ hạng CV không chuyển nguyên vẹn sang test.

![Heatmap F1 test từng nhãn của E1–E12 trên Python](docs/images/results/python_per_label_f1.png)

**Hình 2 — Python, F1 theo nhãn.** DevelopmentNotes và Expand khó hơn Parameters/Usage. E2 tăng DevelopmentNotes so với E1 nhưng giảm Usage; cải thiện nhãn hiếm có thể đi cùng giảm micro F1 tổng thể.

### Java

| Exp | CV micro F1 ± SD | Test micro F1 | Test macro F1 | Test macro AP | Test HL ↓ |
| --- | --- | --- | --- | --- | --- |
| E1 | 0.5908 ± 0.0437 | 0.5679 | 0.5997 | 0.6550 | 0.1361 |
| E2 | 0.5997 ± 0.0565 | 0.5811 | 0.6339 | 0.6223 | 0.1744 |
| E3 | 0.6001 ± 0.0488 | 0.5143 | 0.5358 | 0.6242 | 0.1555 |
| E4 | 0.5908 ± 0.0437 | 0.5650 | 0.6354 | 0.6550 | 0.1936 |
| E5 | 0.5957 ± 0.0489 | 0.5800 | 0.6165 | 0.6441 | 0.1316 |
| E6 | 0.5977 ± 0.0459 | 0.5571 | 0.6022 | 0.6559 | 0.1432 |
| E7 | 0.5809 ± 0.0472 | 0.5565 | 0.5931 | 0.6345 | 0.2004 |
| E8 | **0.6029 ± 0.0680** | 0.5682 | 0.5647 | 0.6408 | 0.1346 |
| E9 | 0.5773 ± 0.0698 | 0.5698 | 0.5861 | 0.6640 | 0.1319 |
| E10 | 0.5830 ± 0.0638 | 0.5290 | 0.5355 | 0.6529 | 0.1367 |
| E11 | 0.6006 ± 0.0501 | **0.6110** | **0.6778** | **0.6937** | **0.1278** |
| E12 | 0.5742 ± 0.0549 | 0.5735 | 0.6024 | 0.6358 | 0.1414 |

![So sánh CV và test micro/macro F1 của E1–E12 trên Java](docs/images/results/java_comparison.png)

**Hình 3 — Java, so sánh tổng thể.** E8 dẫn CV nhẹ, còn E11 dẫn test micro/macro F1. SD fold tương đối lớn, nên không khẳng định ưu thế chỉ từ khoảng cách nhỏ giữa hai giá trị CV trung bình.

![Heatmap F1 test từng nhãn của E1–E12 trên Java](docs/images/results/java_per_label_f1.png)

**Hình 4 — Java, F1 theo nhãn.** Ownership gần hoàn hảo ở nhiều mô hình, trong khi Pointer và rational khó hơn. Điểm cao của một nhãn dễ không đại diện cho chất lượng đồng đều trên toàn hệ nhãn.

### Pharo

| Exp | CV micro F1 ± SD | Test micro F1 | Test macro F1 | Test macro AP | Test HL ↓ |
| --- | --- | --- | --- | --- | --- |
| E1 | 0.6832 ± 0.0403 | 0.6970 | 0.5777 | 0.6307 | 0.0940 |
| E2 | 0.6800 ± 0.0307 | 0.6754 | 0.6171 | 0.6579 | 0.1207 |
| E3 | **0.6923 ± 0.0347** | **0.7028** | 0.6197 | 0.6707 | **0.0928** |
| E4 | 0.6832 ± 0.0403 | 0.6895 | 0.6239 | 0.6307 | 0.1061 |
| E5 | 0.6787 ± 0.0331 | 0.6711 | 0.5876 | 0.6729 | 0.1009 |
| E6 | 0.6776 ± 0.0432 | 0.6813 | 0.5915 | 0.6622 | 0.0993 |
| E7 | 0.6547 ± 0.0300 | 0.6686 | **0.6250** | **0.6934** | 0.1360 |
| E8 | 0.6832 ± 0.0385 | 0.6829 | 0.5346 | 0.6562 | 0.0944 |
| E9 | 0.6878 ± 0.0438 | 0.6813 | 0.5728 | 0.6509 | 0.0997 |
| E10 | 0.6787 ± 0.0356 | 0.6844 | 0.5659 | 0.6535 | 0.0960 |
| E11 | 0.6656 ± 0.0552 | 0.6622 | 0.5431 | 0.6249 | 0.1013 |
| E12 | 0.6893 ± 0.0263 | 0.6879 | 0.6150 | 0.6762 | 0.0989 |

![So sánh CV và test micro/macro F1 của E1–E12 trên Pharo](docs/images/results/pharo_comparison.png)

**Hình 5 — Pharo, so sánh tổng thể.** E3 dẫn CV và test micro F1; E7 có macro F1 tốt nhất nhưng micro thấp hơn E1. Mục tiêu nhãn hiếm và mục tiêu tổng thể có thể dẫn tới lựa chọn khác nhau.

![Heatmap F1 test từng nhãn của E1–E12 trên Pharo](docs/images/results/pharo_per_label_f1.png)

**Hình 6 — Pharo, F1 theo nhãn.** E7 cải thiện Classreferences so với E1 nhưng giảm Example. Cần đánh giá đồng thời từng nhãn, micro và macro thay vì chỉ chọn một cột điểm cao nhất.

## 5. Lựa chọn theo CV và đánh giá trên test

Các ứng viên dưới đây được xác định bằng CV micro F1. “Chênh test” là chênh test micro F1 so với E1, đơn vị **điểm phần trăm**; không phải phần trăm tăng tương đối.

| Ngôn ngữ | Chọn theo CV | CV micro F1 ± SD | Test micro F1 | E1 test micro F1 | Chênh test (điểm %) |
| --- | --- | --- | --- | --- | --- |
| Python | E11 | 0.6410 ± 0.0047 | 0.6467 | 0.6466 | +0.01 |
| Java | E8 | 0.6029 ± 0.0680 | 0.5682 | 0.5679 | +0.02 |
| Pharo | E3 | 0.6923 ± 0.0347 | 0.7028 | 0.6970 | +0.58 |

Trung bình đều test micro F1 của tổ hợp chọn bằng CV là **0.6393**, so với **0.6372** của E1: tăng khoảng **0.21 điểm phần trăm**. Đây là trung bình ba bài toán theo ngôn ngữ, không phải pooled F1 của mô hình 19 nhãn.

Nếu cần một kỹ thuật dùng chung, E11 có trung bình đều CV micro F1 cao nhất, **0.6357**; E1 là **0.6162**. Lựa chọn này vẫn phải chấp nhận điểm yếu của CodeBERT trên Pharo.

## 6. Phân tích và insight

### 6.1. Không có mô hình thắng đồng nhất

- **Python:** E11 tăng CV rõ so với E1 (0.6410 so với 0.5747), nhưng test micro gần như ngang nhau (0.6467 so với 0.6466). E9 v2 có test micro cao nhất 0.6591, nhưng không đứng đầu CV. Không chuyển gain CV thành lời khẳng định gain test tương đương.
- **Java:** E8 dẫn CV nhẹ; E11 đứng gần phía sau và đạt test micro/macro/AP cao nhất. Khoảng cách CV giữa E8 và E11 nhỏ so với SD fold. Không đổi lựa chọn sang E11 chỉ vì đã thấy test cao hơn rồi gọi đó là đánh giá độc lập.
- **Pharo:** E3 dẫn CV và test micro; E7 dẫn test macro F1 và macro AP nhưng micro thấp hơn E1. Lựa chọn khác nhau tùy mục tiêu tổng thể hay nhãn hiếm.

### 6.2. Threshold tuning đổi đánh đổi FP/FN

E4 giữ macro AP như E1 ở cả ba ngôn ngữ, nhưng tăng macro F1, giảm micro F1 và tăng Hamming loss. Điều này phù hợp với việc giữ thứ hạng xác suất nhưng đổi quyết định từng nhãn. Tối ưu F1 riêng mỗi nhãn không trực tiếp tối ưu micro F1.

| Ngôn ngữ | Exp | Micro precision | Micro recall | FP | FN |
| --- | --- | --- | --- | --- | --- |
| Python | E1 | 0.7006 | 0.6003 | 147 | 229 |
| Python | E4 | 0.5911 | 0.6510 | 258 | 200 |
| Java | E1 | 0.6048 | 0.5353 | 198 | 263 |
| Java | E4 | 0.4522 | 0.7527 | 516 | 140 |
| Pharo | E1 | 0.7424 | 0.6569 | 93 | 140 |
| Pharo | E4 | 0.6651 | 0.7157 | 147 | 116 |

Java giảm FN từ 263 xuống 140 nhưng FP tăng từ 198 lên 516. Vì vậy không thể kết luận threshold tuning “tốt hơn” chỉ từ macro F1. Khi bỏ sót nhãn có giá cao, cần ưu tiên recall; khi FP gây phiền, phải đọc precision và HL cùng F1. Mục tiêu nên được chốt trước khi chọn bằng validation.

### 6.3. Loss và imbalance không có lời giải chung

E2 cải thiện một số nhãn hiếm nhưng có thể làm giảm nhãn phổ biến; tác động micro/macro khác nhau theo ngôn ngữ. ASL E7 tăng recall nhưng precision thấp hơn E1. Trên Pharo, macro AP cao của E7 cho thấy thứ hạng hữu ích, trong khi quyết định ở ngưỡng 0.5 chưa đạt micro F1 tốt nhất. Focal Loss tốt ở Pharo nhưng giảm test Java. Có mất cân bằng không đồng nghĩa class weight hoặc loss phức tạp luôn tốt hơn BCE.

### 6.4. Attention và graph cần lợi ích được kiểm chứng

E5 chỉ tăng test micro trên Java so với E1. E6 cải thiện E5 trên Python/Pharo nhưng giảm trên Java. E12 tăng micro/macro so với E5 ở Python/Pharo, giảm ở Java. Graph full train chỉ có 3/3/6 cạnh ngoài self-loop tại ngưỡng cosine 0.1; prior nhãn khá thưa. Đồng xuất hiện không phải quan hệ nhân quả hay luật gán nhãn chắc chắn. Chưa có ablation nhiều seed để quy lợi ích riêng cho graph.

### 6.5. Pretraining phải phù hợp miền

E11 tăng CV micro ở Python/Java nhưng giảm ở Pharo; test Java cải thiện rõ, Python gần ngang baseline, Pharo giảm. Python/Java có trong các ngôn ngữ code pretrain của CodeBERT, còn Pharo không có; đây là giả thuyết phù hợp miền, chưa chứng minh là nguyên nhân duy nhất. Quy mô mẫu và phân bố class cũng có thể đóng góp.

## 7. Hội tụ và chi phí

Số epoch model cuối là trung vị best epoch CV, không phải số epoch thực chạy của mọi fold. Thời gian dưới đây bao gồm 5-fold CV, full train và evaluation; không phải inference latency.

| Exp | Python: epoch / phút | Java: epoch / phút | Pharo: epoch / phút |
| --- | --- | --- | --- |
| E1 | 7 / 14.84 | 6 / 12.88 | 7 / 11.16 |
| E2 | 7 / 15.69 | 4 / 11.13 | 7 / 11.60 |
| E3 | 5 / 12.96 | 6 / 12.92 | 9 / 11.93 |
| E4 | 7 / 15.07 | 6 / 13.09 | 7 / 11.29 |
| E5 | 6 / 14.84 | 6 / 14.03 | 7 / 11.62 |
| E6 | 7 / 15.47 | 6 / 13.74 | 7 / 12.01 |
| E7 | 9 / 18.33 | 7 / 14.11 | 9 / 12.60 |
| E8 | 7 / 27.04 | 6 / 23.93 | 6 / 17.65 |
| E9 | 7 / 15.79 | 5 / 12.81 | 7 / 11.98 |
| E10 | 9 / 36.98 | 6 / 29.33 | 10 / 26.37 |
| E11 | 7 / 16.32 | 6 / 12.74 | 7 / 11.91 |
| E12 | 9 / 17.57 | 6 / 12.93 | 8 / 12.15 |

E8 có thời gian pipeline quan sát khoảng 1.58–1.86 lần E1; E10 khoảng 2.28–2.49 lần. E10 không cải thiện test micro/macro so với E1 trên cả ba ngôn ngữ trong cấu hình này. Epoch thực chạy, GPU và tải chạy đồng thời khác nhau nên không coi các tỷ lệ này là benchmark tốc độ được kiểm soát.

Một số fold còn cải thiện tới epoch 10, một số đạt đỉnh sớm. Không có căn cứ bắt mọi model dừng ở 5 epoch hoặc luôn chạy đủ 10. Cần early stopping, khôi phục best weights và xác nhận bằng validation nếu tăng ngân sách.

## 8. Giới hạn và phạm vi kết luận

1. **Một seed cơ sở:** SD giữa fold không thay thế đánh giá nhiều seed và không phải khoảng tin cậy. Các chênh lệch nhỏ chưa được chứng minh có ý nghĩa thống kê.
2. **CV dùng chọn epoch:** điểm best validation có thiên lệch lựa chọn; chưa dùng nested CV. OOF sau khi tuning threshold cũng không phải đánh giá độc lập.
3. **Test đã được quan sát:** kết quả mang tính thăm dò; không mô tả test hiện tại là benchmark mù. Cần holdout mới hoặc protocol xác nhận mới để kiểm chứng các thay đổi tiếp theo.
4. **Khác partition gốc:** bộ chia group chung phục vụ joint multi-label và k-fold; không tuyên bố vượt baseline Random Forest chính thức trên cùng protocol cuộc thi.
5. **Khác chính sách E9:** khởi tạo head và checkpoint/early stopping riêng; chưa thể quy gain chỉ cho GNN.
6. **Tổng quát hóa:** class giao giữa train/test, near-duplicate chưa được loại hết, input chỉ có một câu; chưa xác nhận khả năng chuyển sang project hoàn toàn mới.
7. **Phạm vi source:** kết quả này chỉ thuộc pipeline DL E1–E12. Các script/report ML cũ không phải bằng chứng đánh giá joint multi-label trong tài liệu này.

## 9. Nguồn số liệu và tái lập

Nguồn cục bộ dùng để tổng hợp là `results/model_zoo/leaderboard.csv` và `metrics.json` của 36 run đã giữ lại theo `results/training/CLEANUP_MANIFEST.json`. Các bảng được đối soát với metric từng run và fingerprint dữ liệu. Mỗi exp/ngôn ngữ giữ một run hoàn chỉnh; E9 chỉ sử dụng head v2, không chọn run bằng test cao nhất.

Các thư mục training/model zoo đang được `.gitignore` loại trừ. Vì vậy, sau khi clone repo, các đường dẫn kết quả cục bộ trên **không mặc nhiên tồn tại**. Sáu hình so sánh và heatmap được sao chép nguyên bản sang `docs/images/results/`, ngoài vùng ignore, và nhúng bằng đường dẫn tương đối để hiển thị trên GitHub. Cần commit thư mục hình cùng `result.md`. PDF chứa thêm các hình và phân tích đầy đủ; checkpoint lớn không được nhúng vào Markdown.

### Fingerprint SHA-256 của dữ liệu

Hash được tính từ tên và nội dung lần lượt `train.csv`, `test.csv`, `folds.json`, `label_cols.json`, theo đúng loader thực nghiệm.

| Ngôn ngữ | Fingerprint |
| --- | --- |
| Python | `1e77a6d33beb1547394723d2062e0cc94cae6fb39ba5c17071a31d01013b52bf` |
| Java | `0d2c2c9066d93d74db835f53e6fc949c21ba4a6cb14b7e5f5b4182ce1451875e` |
| Pharo | `23a6634e63a4a6e93d00399ddfe582583201a4328fdfe671e25805ac39eb1c7e` |

### Chạy lại

Chạy từ thư mục gốc repo, trong môi trường Python đã cài PyTorch phù hợp thiết bị:

```bash
python3 -m pip install -r requirements.txt

# Kiểm tra loader, các biến thể và đường training với mô hình nhỏ offline.
USE_TF=0 python3 -m unittest discover -s tests -v

# Ví dụ: E1 trên Python.
bash scripts/train_experiments.sh --experiments 1 --language python --device cuda:0

# Chạy toàn bộ 12 phương pháp, ba ngôn ngữ, tuần tự trên một GPU.
bash scripts/train_experiments.sh --experiments all --language all --device cuda:0

# Tổng hợp metric sau khi đã có các run hoàn chỉnh.
python3 -m src.evaluation.analyze_results --results results/training --output-dir results/model_zoo
```

Đổi sang `cuda:1` để chọn GPU thứ hai, hoặc `cpu` nếu cần. Fine-tune thực tế cần truy cập pretrained backbone hoặc cache tương ứng. Script tự sinh thư mục output theo từng lần gọi; không kỳ vọng chạy lại cho số liệu bit-for-bit giống hệt trên mọi môi trường/GPU.

Lần kiểm tra source gần nhất: **26 tests pass**. Đây là kiểm tra phần mềm, không thay thế việc chạy đánh giá chất lượng mô hình.

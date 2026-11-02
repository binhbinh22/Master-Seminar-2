# Dữ liệu training hai giai đoạn

Nguồn: `raw/nlbse2023/{python,java,pharo}.csv`, dạng long theo câu/nhãn.
Bỏ partition gốc theo từng nhãn; giữ toàn bộ sentence ID và target sau pivot.

| Thư mục | Nội dung mỗi ngôn ngữ | Mục đích |
|---|---|---|
| `k_fold_data/<language>/` | `all.csv`, `folds.json`, `label_cols.json`, `manifest.json` | 5-fold trên 100% dữ liệu để so sánh phương pháp |
| `full_training_data/<language>/` | `train.csv`, `val.csv`, `test.csv`, `label_cols.json`, `manifest.json` | Train 80%, validation 10%, test 10% |

| Ngôn ngữ | Tổng cho CV | Train | Val | Test | Nhãn |
|---|---:|---:|---:|---:|---:|
| Python | 2.555 | 2.044 | 255 | 256 | 5 |
| Java | 2.418 | 1.935 | 241 | 242 | 7 |
| Pharo | 1.765 | 1.413 | 176 | 176 | 7 |

CSV gồm `comment_sentence_id`, `class`, `group_id`, `text`, sau đó mỗi nhãn là
một cột 0/1 theo thứ tự `label_cols.json`. Text chỉ chuẩn hóa khoảng trắng.
Các câu trùng text luôn ở cùng fold/split. Phân chia seed 42, cân bằng gần đúng
nhãn và số câu; tỷ lệ có thể lệch do số nguyên và kích thước nhóm.
Hai bộ dùng cùng quần thể; loader kiểm tra fingerprint, độ phủ và chồng lấn nhóm.

Tạo lại bằng:

```bash
python3 -m src.data.prepare_two_stage --raw-dir data/raw --output-dir data --folds 5 --seed 42 --force
```

Không có `--force` sẽ từ chối ghi đè dữ liệu hiện có.
Chi tiết lệnh train và chọn top 3–4: [hướng dẫn](../docs/two_stage_training.md).
Test giai đoạn 2 đã tham gia lựa chọn phương pháp qua CV nên không độc lập.

`processed/` đã được xóa. Các CSV ở gốc `data/` vẫn phục vụ notebook EDA;
không phải đầu vào của hai giai đoạn mới. Các kiểm thử legacy dùng fixture tạm,
không cần bộ dữ liệu `processed/` trên đĩa.

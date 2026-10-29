# Data

Dữ liệu cho bài toán NLBSE'23 Code Comment Classification (Python, Java, Pharo).

## Raw

`python.csv`, `java.csv`, `pharo.csv`: dữ liệu NLBSE'23 gốc ở dạng long one-vs-rest. Mỗi câu có một dòng cho từng category (`instance_type` = 1 nếu câu thuộc category đó). Cột `partition` là phân vùng train/test theo từng category nên không dùng để chia dữ liệu.

## Processed

`processed/<ngôn ngữ>/` là bộ dữ liệu chung cho mọi mô hình, được tạo bởi [notebooks/data_preprocessing.ipynb](../notebooks/data_preprocessing.ipynb):

| File | Nội dung |
|---|---|
| `train.csv`, `test.csv` | `comment_sentence_id`, `class`, `group_id`, `text`, rồi mỗi nhãn một cột 0/1 |
| `folds.json` | 5 fold trên train: `[{"fold", "train_indices", "val_indices"}, ...]`, chỉ số là vị trí dòng (từ 0) trong `train.csv` |
| `label_cols.json` | thứ tự cột nhãn |

| Ngôn ngữ | Train | Test | Nhãn |
|---|---|---|---|
| python | 2.044 | 511 | 5 |
| java | 1.934 | 484 | 7 |
| pharo | 1.411 | 354 | 7 |

- `text` chỉ được chuẩn hóa khoảng trắng; giữ nguyên chữ hoa/thường và ký hiệu code.
- Các câu có text giống hệt nhau cùng một `group_id` và luôn nằm cùng phía (train/test, và train/validation của mỗi fold).
- Test (≈ 20%) và 5 fold được chia cân bằng nhãn, seed 42. Test chỉ dùng để đánh giá cuối.
- Một nhóm trùng rất lớn có nhãn hiếm không thể tách ra nên có thể làm lệch nhãn đó, ví dụ `Ownership` của Java (nhóm 32 câu `@author kevin bourrillion`): fold validation chứa nhóm này có tỷ lệ nhãn ≈ 1,7× mức lý tưởng.
- Chạy lại notebook sẽ ghi đè `processed/`.

TF-IDF, tokenizer... không nằm trong `processed/`; từng notebook huấn luyện tự tạo, và TF-IDF phải fit trên phần train của từng fold.

## Wide multi-label cũ (`*_preprocessed.csv`)

`python_preprocessed.csv`, `java_preprocessed.csv`, `pharo_preprocessed.csv` là bản wide multi-label chưa chia train/test, dùng cho [notebooks/EDA_all_datasets.ipynb](../notebooks/EDA_all_datasets.ipynb) (đọc `python_preprocessed.csv`). Java/Pharo được tạo bằng `python src/data/make_wide.py --language java pharo`. Các file này không phải đầu vào của `processed/`.

# Data

Thư mục dữ liệu cho bài toán NLBSE'23 Code Comment Classification (Python).

## Files

- `python.csv`: dữ liệu NLBSE'23 gốc ở dạng long one-vs-rest. Mỗi câu có một dòng cho từng category.
- `python_preprocessed.csv`: dữ liệu đã chuyển sang wide multi-label, một dòng cho mỗi `comment_sentence_id`.

## Preprocessed schema

`python_preprocessed.csv` gồm:

- `comment_sentence_id`: định danh câu;
- `class`: class/source chứa comment;
- `comment_sentence`: text sau khi chuẩn hóa whitespace;
- `DevelopmentNotes`, `Expand`, `Parameters`, `Summary`, `Usage`: năm nhãn nhị phân;
- `label_count`: số nhãn dương của câu.

File processed có 2.555 dòng, không có missing value và không trùng `comment_sentence_id`. Tổng positive của từng nhãn được đối soát với file gốc.

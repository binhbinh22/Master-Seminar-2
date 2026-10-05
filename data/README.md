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

## Java / Pharo

Tạo wide multi-label bằng `python src/data/make_wide.py --language java pharo`:

- `java_preprocessed.csv`: 2.418 câu, 7 nhãn (`Expand, Ownership, Pointer, deprecation, rational, summary, usage`);
- `pharo_preprocessed.csv`: 1.765 câu, 7 nhãn (`Classreferences, Collaborators, Example, Intent, Keyimplementationpoints, Keymessages, Responsibilities`).

Thêm cột `partition_official`: partition theo đa số phiếu giữa các category (0 = train, 1 = test). Cột này **chỉ để đối chiếu**, không dùng để chia dữ liệu vì partition gốc nằm trên từng cặp (câu, category).

## Artifacts cho mô hình

Chạy `LANGUAGE=<python|java|pharo>` với `notebooks/data_preprocessing_roberta.ipynb` và `notebooks/data_preprocessing_tfidf.ipynb`.
Cùng cách chia ở cả 3 ngôn ngữ: `MultilabelStratifiedKFold` 80/20 (seed 42) + 5-fold CV trên train.

- Python: `data/roberta/`, `data/tfidf/`
- Java/Pharo: `data/{roberta,tfidf}/{java,pharo}/`

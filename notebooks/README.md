# Research EDA

Ba notebook dùng dữ liệu chuẩn trong `data/raw/nlbse2023/` và phong cách vẽ chung tại `eda_utils.py`:

- `EDA_all_datasets.ipynb`: toàn bộ dữ liệu, nhãn, nhóm text/class, partition gốc và kiểm tra bộ chia thực nghiệm.
- `EDA_python_code_comment_classification.ipynb`: support và quan hệ nhãn Python, tổ hợp nhãn, đặc trưng từ vựng và ràng buộc nhóm.
- `EDA_visualizations.ipynb`: hình chọn lọc cho báo cáo, tokenizer RoBERTa cục bộ, độ lệch fold và giao class train/test.

Chạy từ gốc repo; cài dependencies notebook nếu chưa có:

```bash
python3 -m pip install -r notebooks/requirements.txt
python3 scripts/export_eda_figures.py --output-dir results/EDA --dpi 300 --save-notebooks
```

Hình PNG được chia theo notebook; `manifest.json` ghi cell nguồn, hash nguồn và số hình.
Exporter chỉ thay bộ hình cũ sau khi tất cả notebook thành công. `--save-notebooks` lưu bảng/hình đã chạy vào notebook.

Màu phân biệt ngôn ngữ hoặc partition theo chú giải. Thang xanh biểu thị độ lớn; thang nâu/xanh có tâm 0 biểu thị độ lệch prevalence theo điểm phần trăm. Màu không biểu thị độ tốt của mô hình.

Ma trận nhãn dùng `int64`, nhóm text chỉ chuẩn hóa khoảng trắng và giữ nguyên hoa/thường. Loader kiểm tra target với raw, độ phủ CV và tách nhóm thực tế. Không tạo KFold minh họa thay cho folds đã dùng train. Không suy ra lỗi model hay quan hệ nhân quả từ EDA.

Phần tokenizer cần artifact E1 trong `results/training/kfold*/exp1/python/tokenizer`; chỉ đọc cục bộ, không tự tải model. `data_preprocessing.ipynb` thuộc protocol cũ và không được chạy bởi exporter EDA.

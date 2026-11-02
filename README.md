# Code Comment Classification

Phân loại đa nhãn comment Python, Java và Pharo bằng 17 cấu hình RoBERTa/CodeBERT.
Chạy các lệnh từ thư mục gốc repo.

## Cài đặt

```bash
python3 -m pip install -r requirements.txt
```

## Dữ liệu

- `data/raw/nlbse2023/`: dữ liệu gốc.
- `data/k_fold_data/`: 5-fold trên 100% dữ liệu để so sánh phương pháp.
- `data/full_training_data/`: train/validation/test khoảng 80/10/10.

Hai bộ đã được tạo sẵn. Nếu cần tạo lại:

```bash
python3 -m src.data.prepare_two_stage --folds 5 --seed 42 --force
```

## Giai đoạn 1: K-fold và chọn phương pháp

```bash
bash scripts/train_kfold.sh --device cuda:0 \
  --output-dir results/training/kfold_seed42

python3 -m src.evaluation.two_stage_results select \
  --results results/training/kfold_seed42 --top-k 4 \
  --output-dir results/training/selection
```

Mặc định chạy E1–E17 trên cả ba ngôn ngữ. Chọn top 4 riêng cho từng ngôn ngữ
bằng CV micro F1; dùng `--top-k 3` nếu muốn chọn 3 phương pháp.

## Giai đoạn 2: train các phương pháp đã chọn

```bash
bash scripts/train_full.sh \
  --selection results/training/selection/selection.json \
  --device cuda:0 --output-dir results/training/final_seed42
```

Khởi tạo lại model, train trên 80%, dùng validation 10% chọn checkpoint rồi
chấm test 10%. Kết quả nằm trong thư mục output, báo cáo tại `report/`.

Cả hai script hỗ trợ `--language python`, `--device cpu`, `--seed 42`,
`--python /path/to/python`, `--dry-run` và `--help`.
Để chạy một nhóm phương pháp ở giai đoạn 1, thêm `--experiments 1,2,11`.

**Lưu ý:** test giai đoạn 2 đã tham gia lựa chọn phương pháp qua K-fold,
nên kết quả test không độc lập với quá trình lựa chọn.

## Kiểm thử

```bash
USE_TF=0 MODEL_ZOO_PROGRESS=off python3 -m unittest discover -s tests -v
```

Chi tiết: [quy trình hai giai đoạn](docs/two_stage_training.md).
Danh sách model/config: `python3 -m experiments.dl.model_zoo list`.

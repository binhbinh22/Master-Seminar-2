                                                                   # Report — Code Comment Classification (NLBSE'23)

Báo cáo LaTeX cho Seminar 2, viết theo yêu cầu Topic 2.

## Cấu trúc

- `report.tex` — file báo cáo chính (LaTeX).
- `imgs/` — thư mục chứa hình ảnh (tạo khi cần chèn hình).

## Cách build

### Cách 1: Overleaf (khuyến nghị)
1. Nén thư mục `docs/report/` thành file `.zip`.
2. Upload lên [Overleaf](https://www.overleaf.com) → New Project → Upload Project.
3. Chọn compiler **pdfLaTeX** (hoặc **XeLaTeX** nếu lỗi font tiếng Việt).

### Cách 2: Local (TeX Live / MiKTeX)
```bash
cd docs/report
pdflatex report.tex
pdflatex report.tex   # chạy 2 lần để cập nhật mục lục
```

Nếu gặp lỗi font tiếng Việt với `pdflatex`, dùng `xelatex`:
```bash
xelatex report.tex
xelatex report.tex
```

## Nội dung báo cáo (theo Topic 2)

| Mục | Section trong report.tex |
|-----|--------------------------|
| Dataset & Data Understanding | `\section{Dataset \& Data Understanding}` |
| Data Preprocessing & Architecture | `\section{Data Preprocessing \& Architecture}` |
| Model Development | `\section{Model Development}` |
| RoBERTa Baseline | `\subsection{RoBERTa Baseline}` |
| Cross-validation & Evaluation | `\section{Cross-validation \& Evaluation}` |
| Result Analysis | `\section{Result Analysis}` |
| Project Packaging & Report | `\section{Project Packaging \& Reproducibility}` |

## Việc cần hoàn thiện

- [ ] Chạy RoBERTa baseline và điền kết quả vào `\subsection{RoBERTa Baseline}`.
- [ ] Hoàn thiện `\subsection{Error Analysis}` (confusion Summary/Expand, threshold tuning).
- [ ] Thêm hình minh họa (kiến trúc, biểu đồ kết quả) vào `imgs/`.
- [ ] Điền tên + MSSV thành viên nhóm ở trang bìa.
- [ ] Kiểm tra lại số liệu khớp với `results/`.

## Nguồn số liệu

Số liệu trong báo cáo lấy từ:
- `results/ml_pipeline_report.md` — kết quả 4 model classical.
- `results/ANALYSIS.md` — phân tích per-label và nhận xét.
- `data/README.md` — mô tả schema dữ liệu.

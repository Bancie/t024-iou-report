# T024 IoU report

Hai notebook marimo báo cáo kết quả IoU của SAM 3 trên bộ review nuImages (class `car`): baseline với prompt `passenger car`, và lần chạy full live demo (vòng lặp trên opt, chọn prompt trên validation, test chấm một lần). Notebook chỉ đọc file kết quả có sẵn trong repo, không gọi Modal, không tải ảnh, không chạy lại model; khi chạy trên molab nó tải kết quả từ repo này qua mạng.

Metric là binary IoU của mask `car` đã gộp, trung bình theo ảnh (mean IoU). Đây không phải mIoU trên 19 class.

## Số chính

| Split | Prompt | Số ảnh | Mean IoU |
| --- | --- | --- | --- |
| test | prompt được chọn | 87 | 0.7263 |
| test | baseline `passenger car` | 87 | 0.5745 |
| validation | prompt được chọn | 30 | 0.8235 |
| validation | baseline `passenger car` | 30 | 0.7323 |

Prompt được chọn trên validation, test chấm một lần. Chi tiết nằm trong `full_demo_report.py`.

## Xem trên molab

- Báo cáo full live demo (có các số trên): https://molab.marimo.io/github/Bancie/t024-iou-report/blob/main/full_demo_report.py
- Báo cáo IoU baseline (`passenger car`, 147 ảnh scored): https://molab.marimo.io/github/Bancie/t024-iou-report/blob/main/iou_eval_report.py

## Chạy trên macOS (Python 3.11+)

```bash
git clone https://github.com/Bancie/t024-iou-report.git
cd t024-iou-report
python3 -m venv .venv && source .venv/bin/activate
pip install "marimo>=0.16" matplotlib
marimo edit full_demo_report.py
```

Xuất HTML tĩnh:

```bash
marimo export html full_demo_report.py -o full_demo_report.html
```

## Cấu trúc

- `full_demo_report.py` — notebook full live demo (đọc `results/full_demo/` và `results/edge_loop/`)
- `iou_eval_report.py` — notebook baseline (đọc các file `results/*.json` và `results/*.txt`)
- `results/` — chỉ file JSON và text nhỏ: `per_image.json`, `summary.json`, `review_report.json`, `splits_summary.json`, `examples.json`, `sam3_health.json`, `check_gold.txt`, `split_check.txt`, `full_demo/*.json`, `edge_loop/*.json`

## Không có trong repo

- Ảnh nuImages, mask gold và ảnh overlay (nuImages dùng license CC BY-NC-SA). Phần "Overlay" trong notebook vì vậy chỉ hiện tên và IoU, không hiện ảnh.
- Bộ review và script tạo overlay (`make_overlays.py`) cần dataset gốc nên không đưa vào.
- Ba khóa cấu hình (URL endpoint Modal và đường dẫn dataset local) đã được bỏ khỏi `results/full_demo/run_summary.json`; notebook vốn đã ẩn chúng khi hiển thị.

## Đóng góp

Fork repo, sửa notebook, mở PR. Nguồn gốc: `source-tool/gdp/report/` trong repo nội bộ của team T024; bản này thêm bước tải `results/` qua mạng cho molab và thay vài dòng chữ nói về overlay và repo private.

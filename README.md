# paddlehuman-fight-detection
Dự án demo mang tính nghiên cứu, thử nghiệm phát hiện hành vi đánh nhau của con người bằng PaddleHuman. Dự án chỉ phục vụ mục đích thử nghiệm và học tập, không nhằm mục đích thương mại.

Đọc 1 video và model phát hiện đánh nhau, xuất ra **chính video đó (giữ âm thanh)**. Ở các đoạn có đánh nhau, video có banner đỏ **"CẢNH BÁO: PHÁT HIỆN ĐÁNH NHAU"** và viền đỏ nhấp nháy. Không cần Docker hay Redis.

- Model: PaddleHuman **ppTSM** (`models/ppTSM.onnx`), chạy bằng `onnxruntime` (GPU nếu có).
- Chạy trên Google Colab: mở `fight_warning_colab.ipynb` (hướng dẫn nằm ngay trong notebook). Ở bước 2 của notebook, đặt `PROJECT_DIR` trỏ tới thư mục chứa repo này trên Drive.

## Cấu trúc

```
├── run.py                  # CLI: python run.py --input in.mp4 --output out.mp4
├── models/ppTSM.onnx       # model
├── fight_warning/
│   ├── detector.py         # tiền xử lý + suy luận ONNX
│   ├── pipeline.py         # chấm điểm video, làm mượt, tìm đoạn đánh nhau, ghi video ra
│   └── overlay.py          # vẽ banner/viền cảnh báo
├── fight_warning_colab.ipynb
└── requirements.txt
```

## Chạy bằng dòng lệnh

```bash
pip install -r requirements.txt
python run.py --input input.mp4 --output output/input_warning.mp4
```

Kết quả gồm:
- `output/input_warning.mp4`: video có cảnh báo (H.264, giữ audio gốc nếu máy có `ffmpeg`).
- `output/input_warning_report.json`: các đoạn đánh nhau (giây bắt đầu/kết thúc, điểm) và điểm của từng cửa sổ.

Tham số hay dùng (`python run.py --help` để xem hết):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--threshold` | 0.65 | Điểm fight (đã làm mượt) >= ngưỡng thì tính là đánh nhau |
| `--smoothing-window` | 3 | Làm mượt bằng trung bình N lần chấm điểm liên tiếp |
| `--min-event-sec` | 1.0 | Bỏ các đoạn ngắn hơn N giây |
| `--merge-gap-sec` | 1.0 | Gộp 2 đoạn cách nhau ít hơn N giây |
| `--stride` | 8 | Cứ N frame chấm điểm 1 lần (tính ở 25 fps, tự co giãn theo fps video) |
| `--device` | auto | `auto` / `cuda` / `cpu` |
| `--show-hud` | False | In điểm fight nhỏ ở góc cả khi không đánh nhau (để debug) |

## Cách hoạt động

1. **Chấm điểm** (`fight_warning/pipeline.py: score_video`): đọc video, cứ mỗi `stride` frame lấy 1 clip 8 frame cách nhau `sample_freq` = 7 frame (khoảng 2.2 giây ở 25 fps). Tiền xử lý: resize 341x256, cắt giữa 224x224, chuẩn hoá ImageNet, rồi model trả về xác suất fight.
2. **Tìm đoạn đánh nhau** (`build_timeline`): làm mượt điểm, gán điểm mỗi clip cho frame ở giữa clip, nội suy ra từng frame, so ngưỡng, gộp các đoạn gần nhau và bỏ đoạn quá ngắn.
3. **Ghi video** (`render_video` + `fight_warning/overlay.py`): đọc lại video, vẽ cảnh báo lên frame thuộc đoạn đánh nhau, đẩy frame sang `ffmpeg` (libx264 + audio gốc). Nếu không có ffmpeg thì dùng `cv2.VideoWriter` (mp4v, không có audio).

Vì xem trước được toàn bộ video nên cảnh báo bật sát lúc bắt đầu đánh nhau, không bị trễ như khi chạy realtime.

## Tuyên bố miễn trừ trách nhiệm

Dự án này chỉ dành cho mục đích nghiên cứu, giáo dục và thử nghiệm.

Dự án không được thiết kế và không nhằm mục đích sử dụng thương mại. Phần cài đặt được cung cấp như một bản demo phát hiện hành vi đánh nhau của con người bằng PaddleHuman.

Người dùng có trách nhiệm đảm bảo việc sử dụng dự án này tuân thủ pháp luật hiện hành, các quy định về quyền riêng tư và giấy phép của mọi model, thư viện bên thứ ba liên quan.

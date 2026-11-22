"""CLI: video đầu vào -> video đầu ra có cảnh báo ở đoạn đánh nhau.

Ví dụ:
    python run.py --input input.mp4 --output output/input_warning.mp4
    python run.py --input input.mp4 --output out.mp4 --threshold 0.7 --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fight_warning import Settings, process_video  # noqa: E402

DEFAULT_MODEL = os.path.join(HERE, "models", "ppTSM.onnx")


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def main() -> None:
    d = Settings()
    p = argparse.ArgumentParser(description="Phát hiện đánh nhau và chèn cảnh báo lên video.")
    p.add_argument("--input", "-i", required=True, help="Đường dẫn video đầu vào")
    p.add_argument("--output", "-o", default=None,
                   help="Đường dẫn video đầu ra (mặc định: output/<tên>_warning.mp4)")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help="File model ONNX (ppTSM)")
    p.add_argument("--report", default=None, help="File JSON kết quả (mặc định: <output>_report.json)")
    p.add_argument("--threshold", type=float, default=d.threshold)
    p.add_argument("--smoothing-window", type=int, default=d.smoothing_window)
    p.add_argument("--min-event-sec", type=float, default=d.min_event_sec)
    p.add_argument("--merge-gap-sec", type=float, default=d.merge_gap_sec)
    p.add_argument("--sample-freq", type=int, default=d.sample_freq)
    p.add_argument("--stride", type=int, default=d.stride)
    p.add_argument("--no-adapt-fps", action="store_true", help="Không co giãn sample_freq/stride theo fps video")
    p.add_argument("--device", default=d.device, choices=["auto", "cuda", "gpu", "cpu"])
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--warning-text", default=d.warning_text)
    p.add_argument("--font", default=None, help="File font .ttf (mặc định tự tìm DejaVuSans-Bold)")
    p.add_argument("--show-score", type=_bool, default=d.show_score)
    p.add_argument("--show-hud", type=_bool, default=d.show_hud)
    p.add_argument("--blink", type=_bool, default=d.blink)
    p.add_argument("--keep-audio", type=_bool, default=d.keep_audio)
    p.add_argument("--crf", type=int, default=d.crf)
    a = p.parse_args()

    if not os.path.isfile(a.input):
        p.error(f"Không thấy video đầu vào: {a.input}")
    if not os.path.isfile(a.model):
        p.error(f"Không thấy model: {a.model}")
    output = a.output or os.path.join(
        HERE, "output", os.path.splitext(os.path.basename(a.input))[0] + "_warning.mp4")

    s = Settings(
        threshold=a.threshold, smoothing_window=a.smoothing_window,
        min_event_sec=a.min_event_sec, merge_gap_sec=a.merge_gap_sec,
        sample_freq=a.sample_freq, stride=a.stride, adapt_to_fps=not a.no_adapt_fps,
        device=a.device, batch_size=a.batch_size,
        warning_text=a.warning_text, font_path=a.font,
        show_score=a.show_score, show_hud=a.show_hud, blink=a.blink,
        keep_audio=a.keep_audio, crf=a.crf,
    )
    process_video(a.input, output, a.model, s, report_path=a.report)


if __name__ == "__main__":
    main()

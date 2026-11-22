"""Video vào -> video ra có cảnh báo ở đoạn đánh nhau.

Chạy 2 lượt trên cùng 1 file video (không cần Redis/Docker):
  1. score_video   : đọc toàn bộ video, cứ mỗi `stride` frame lấy 1 cửa sổ (8 frame, cách nhau
                     `sample_freq` frame ~ 2 giây) cho model chấm điểm fight.
  2. build_timeline: làm mượt điểm, so ngưỡng, gộp/loại đoạn ngắn -> danh sách đoạn đánh nhau.
  3. render_video  : đọc lại video, vẽ cảnh báo lên các frame thuộc đoạn đánh nhau, ghi ra file
                     (H.264 + giữ audio gốc nếu có ffmpeg, nếu không thì cv2.VideoWriter mp4v).

Vì chạy offline nên điểm của mỗi cửa sổ được gán cho frame ở GIỮA cửa sổ (không bị trễ ~2 giây như
luồng realtime gốc), cảnh báo xuất hiện sát thời điểm bắt đầu đánh nhau hơn.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detector import FightDetector
from .overlay import WarningOverlay


@dataclass
class Settings:
    # --- Ngưỡng & làm mượt (mặc định giống violence-paddlehuman-business-service) ---
    threshold: float = 0.65          # điểm fight (sau làm mượt) >= ngưỡng -> đánh nhau
    smoothing_window: int = 3        # trung bình trượt trên N cửa sổ liên tiếp
    min_event_sec: float = 1.0       # bỏ các đoạn đánh nhau ngắn hơn (lọc báo động giả)
    merge_gap_sec: float = 1.0       # gộp 2 đoạn nếu khoảng trống giữa chúng ngắn hơn
    # --- Lấy mẫu clip cho model (tính theo frame ở fps chuẩn 25, tự co giãn theo fps thật) ---
    sample_freq: int = 7             # khoảng cách giữa 8 frame trong 1 clip (8 x 7 ~ 2.24 s @25fps)
    stride: int = 8                  # cứ mỗi N frame chấm điểm 1 lần (~0.32 s @25fps)
    reference_fps: float = 25.0
    adapt_to_fps: bool = True
    # --- Suy luận ---
    device: str = "auto"             # auto | cuda | cpu
    batch_size: int = 8
    # --- Hiển thị ---
    warning_text: str = "CẢNH BÁO: PHÁT HIỆN ĐÁNH NHAU"
    font_path: Optional[str] = None
    show_score: bool = True          # in điểm fight trên banner cảnh báo
    show_hud: bool = False           # in điểm fight nhỏ ở góc cả khi KHÔNG đánh nhau (để debug)
    blink: bool = True               # viền đỏ nhấp nháy
    # --- Ghi video ---
    keep_audio: bool = True
    crf: int = 20                    # chất lượng H.264 (nhỏ hơn = đẹp hơn, file nặng hơn)


@dataclass
class ScoreResult:
    fps: float
    n_frames: int
    width: int
    height: int
    sample_freq: int
    stride: int
    span: int
    centers: List[float] = field(default_factory=list)   # vị trí (frame) giữa mỗi cửa sổ
    scores: List[float] = field(default_factory=list)    # điểm fight thô của mỗi cửa sổ


def _log(msg: str) -> None:
    print(msg, flush=True)


def _video_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 1 or fps > 240:
        return 25.0
    return float(fps)


# ---------------------------------------------------------------------------- lượt 1: chấm điểm
def score_video(video_path: str, detector: FightDetector, s: Settings) -> ScoreResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    fps = _video_fps(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ratio = fps / s.reference_fps if s.adapt_to_fps else 1.0
    sf = max(1, int(round(s.sample_freq * ratio)))
    stride = max(1, int(round(s.stride * ratio)))
    T = detector.num_seg
    span = (T - 1) * sf + 1

    res = ScoreResult(fps=fps, n_frames=0, width=width, height=height, sample_freq=sf, stride=stride, span=span)
    _log(f"[1/2] Chấm điểm: {video_path} | {width}x{height} @ {fps:.2f} fps, ~{total} frame | "
         f"clip {T} frame x cách {sf} (= {span / fps:.2f}s), chấm mỗi {stride} frame")

    buf: deque = deque(maxlen=span)
    pend_clips: List[np.ndarray] = []
    pend_centers: List[float] = []

    def flush() -> None:
        if pend_clips:
            probs = detector.predict(pend_clips)
            res.centers.extend(pend_centers)
            res.scores.extend(float(p) for p in probs)
            pend_clips.clear()
            pend_centers.clear()

    t0 = time.time()
    last_log = t0
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        buf.append(detector.preprocess_frame(frame))
        n = idx + 1
        if n >= span and (n - span) % stride == 0:
            pend_clips.append(np.stack([buf[i * sf] for i in range(T)]))
            pend_centers.append(idx - (span - 1) / 2.0)
            if len(pend_clips) >= s.batch_size:
                flush()
        now = time.time()
        if now - last_log >= 10:
            pct = f"{100.0 * n / total:.1f}%" if total > 0 else "?"
            _log(f"      {n}/{total or '?'} frame ({pct}) - {n / (now - t0):.1f} frame/s")
            last_log = now
    cap.release()

    n_frames = idx + 1
    if n_frames <= 0:
        raise ValueError(f"Video không có frame nào đọc được: {video_path}")
    res.n_frames = n_frames

    if n_frames < span:
        # Video ngắn hơn 1 clip: lấy đều 8 frame trên toàn bộ video
        ids = np.linspace(0, len(buf) - 1, T).round().astype(int)
        pend_clips.append(np.stack([buf[i] for i in ids]))
        pend_centers.append((n_frames - 1) / 2.0)
    elif (n_frames - span) % stride != 0:
        # Thêm 1 cửa sổ kết thúc đúng frame cuối để phủ hết đuôi video
        pend_clips.append(np.stack([buf[i * sf] for i in range(T)]))
        pend_centers.append(n_frames - 1 - (span - 1) / 2.0)
    flush()

    el = time.time() - t0
    _log(f"      Xong {n_frames} frame, {len(res.scores)} lần chấm điểm, {el:.1f}s ({n_frames / max(el, 1e-6):.1f} frame/s)")
    return res


# ---------------------------------------------------------------------------- hậu xử lý
def _centered_moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) <= 1:
        return x.copy()
    kernel = np.ones(k, dtype=np.float64)
    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones_like(x), kernel, mode="same")  # chia đúng số phần tử ở 2 đầu
    return num / den


def _runs(flags: np.ndarray) -> List[List[int]]:
    """Các đoạn liên tiếp True -> [[start, end], ...] (end tính cả)."""
    if not flags.any():
        return []
    d = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def build_timeline(sr: ScoreResult, s: Settings) -> Dict:
    raw = np.asarray(sr.scores, dtype=np.float64)
    centers = np.asarray(sr.centers, dtype=np.float64)
    smoothed = _centered_moving_average(raw, int(s.smoothing_window))

    frames = np.arange(sr.n_frames, dtype=np.float64)
    frame_score = np.interp(frames, centers, smoothed)
    flags = frame_score >= s.threshold

    segs = _runs(flags)
    gap = int(round(s.merge_gap_sec * sr.fps))
    merged: List[List[int]] = []
    for a, b in segs:
        if merged and a - merged[-1][1] - 1 <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    min_len = int(round(s.min_event_sec * sr.fps))
    final = [seg for seg in merged if seg[1] - seg[0] + 1 >= min_len]

    fight = np.zeros(sr.n_frames, dtype=bool)
    segments = []
    for a, b in final:
        fight[a:b + 1] = True
        in_seg = (centers >= a) & (centers <= b)
        seg_raw = raw[in_seg] if in_seg.any() else frame_score[a:b + 1]
        segments.append({
            "start_frame": a,
            "end_frame": b,
            "start_sec": round(a / sr.fps, 2),
            "end_sec": round((b + 1) / sr.fps, 2),
            "duration_sec": round((b - a + 1) / sr.fps, 2),
            "max_score": round(float(np.max(seg_raw)), 4),
            "mean_score": round(float(np.mean(seg_raw)), 4),
        })

    return {
        "frame_score": frame_score.astype(np.float32),
        "fight": fight,
        "segments": segments,
        "windows": [
            {"frame": round(float(c), 1), "time_sec": round(float(c) / sr.fps, 3),
             "score": round(float(r), 4), "smoothed": round(float(m), 4)}
            for c, r, m in zip(centers, raw, smoothed)
        ],
    }


# ---------------------------------------------------------------------------- lượt 2: ghi video
class _FfmpegWriter:
    def __init__(self, ffmpeg: str, out_path: str, w: int, h: int, fps: float,
                 audio_src: Optional[str], crf: int) -> None:
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
        if audio_src:
            cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k"]
        cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # yuv420p cần kích thước chẵn
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self.close()  # ffmpeg đã chết -> close() sẽ raise kèm thông báo lỗi của ffmpeg
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        err = self.proc.stderr.read().decode("utf-8", errors="ignore")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg lỗi: {err.strip()}")


class _CvWriter:
    def __init__(self, out_path: str, w: int, h: int, fps: float) -> None:
        self.vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not self.vw.isOpened():
            raise RuntimeError(f"Không tạo được file output: {out_path}")

    def write(self, frame: np.ndarray) -> None:
        self.vw.write(frame)

    def close(self) -> None:
        self.vw.release()


def _ffmpeg_with_x264() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
        return ffmpeg if "libx264" in out else None
    except Exception:
        return None


def render_video(video_path: str, out_path: str, sr: ScoreResult, tl: Dict, s: Settings) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    w, h, fps = sr.width, sr.height, sr.fps
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    ffmpeg = _ffmpeg_with_x264()
    if ffmpeg:
        writer = _FfmpegWriter(ffmpeg, out_path, w, h, fps, video_path if s.keep_audio else None, s.crf)
        mode = "ffmpeg/H.264" + (" + audio gốc" if s.keep_audio else "")
    else:
        writer = _CvWriter(out_path, w, h, fps)
        mode = "cv2/mp4v (không có ffmpeg+libx264 -> không giữ audio)"
    _log(f"[2/2] Ghi video: {out_path} ({mode})")

    overlay = WarningOverlay(w, h, fps, warning_text=s.warning_text, font_path=s.font_path,
                             show_score=s.show_score, show_hud=s.show_hud, blink=s.blink)
    fight, score = tl["fight"], tl["frame_score"]
    last = sr.n_frames - 1
    t0 = last_log = time.time()
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            j = min(idx, last)
            writer.write(overlay.draw(frame, idx, bool(fight[j]), float(score[j])))
            now = time.time()
            if now - last_log >= 10:
                _log(f"      {idx + 1}/{sr.n_frames} frame ({100.0 * (idx + 1) / sr.n_frames:.1f}%)")
                last_log = now
    finally:
        cap.release()
        writer.close()
    _log(f"      Xong {idx + 1} frame, {time.time() - t0:.1f}s")
    return out_path


# ---------------------------------------------------------------------------- hàm chính
def process_video(input_path: str, output_path: str, model_path: str,
                  settings: Optional[Settings] = None, report_path: Optional[str] = None) -> Dict:
    s = settings or Settings()
    t0 = time.time()
    detector = FightDetector(model_path, device=s.device)
    sr = score_video(input_path, detector, s)
    tl = build_timeline(sr, s)
    render_video(input_path, output_path, sr, tl, s)

    report = {
        "input": os.path.abspath(input_path),
        "output": os.path.abspath(output_path),
        "model": os.path.abspath(model_path),
        "providers": detector.providers,
        "fps": sr.fps,
        "n_frames": sr.n_frames,
        "duration_sec": round(sr.n_frames / sr.fps, 2),
        "resolution": [sr.width, sr.height],
        "clip": {"num_seg": detector.num_seg, "sample_freq": sr.sample_freq, "stride": sr.stride,
                 "span_frames": sr.span, "span_sec": round(sr.span / sr.fps, 2)},
        "settings": asdict(s),
        "fight_segments": tl["segments"],
        "fight_total_sec": round(float(tl["fight"].sum()) / sr.fps, 2),
        "windows": tl["windows"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    report_path = report_path or os.path.splitext(output_path)[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = os.path.abspath(report_path)

    _log("")
    if tl["segments"]:
        _log(f"Phát hiện {len(tl['segments'])} đoạn đánh nhau (tổng {report['fight_total_sec']}s):")
        for i, seg in enumerate(tl["segments"], 1):
            _log(f"  #{i}: {seg['start_sec']:.2f}s -> {seg['end_sec']:.2f}s "
                 f"({seg['duration_sec']:.2f}s, max score {seg['max_score']:.2f})")
    else:
        _log("Không phát hiện đoạn đánh nhau nào.")
    _log(f"Video output : {report['output']}")
    _log(f"Report JSON  : {report['report_path']}")
    _log(f"Tổng thời gian: {report['elapsed_sec']}s")
    return report

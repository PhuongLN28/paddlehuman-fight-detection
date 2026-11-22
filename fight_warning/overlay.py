"""Vẽ cảnh báo đánh nhau lên frame.

Banner (nền đỏ bán trong suốt + icon tam giác + chữ tiếng Việt) được render 1 lần bằng PIL rồi chỉ
alpha-blend mỗi frame -> nhanh, và hiển thị đúng dấu tiếng Việt (cv2.putText không hỗ trợ Unicode).
Nếu không có PIL/font thì tự rơi về chữ ASCII bằng cv2.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

_FALLBACK_TEXT = "WARNING: FIGHT DETECTED"


def find_font(font_path: Optional[str] = None) -> Optional[str]:
    candidates = []
    if font_path:
        candidates.append(font_path)
    try:
        import matplotlib  # Colab luôn có sẵn, kèm font DejaVu (hỗ trợ tiếng Việt)

        candidates.append(os.path.join(os.path.dirname(matplotlib.__file__),
                                       "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf"))
    except Exception:
        pass
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


class WarningOverlay:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        warning_text: str = "CẢNH BÁO: PHÁT HIỆN ĐÁNH NHAU",
        font_path: Optional[str] = None,
        show_score: bool = True,
        show_hud: bool = False,
        blink: bool = True,
    ) -> None:
        self.w, self.h = int(width), int(height)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.show_score = show_score
        self.show_hud = show_hud
        self.blink = blink

        base = min(self.w, self.h)
        self.border = max(4, int(round(base * 0.016)))
        self.banner_h = max(36, int(round(base * 0.11)))
        self.score_scale = max(0.4, self.banner_h / 90.0)
        self.score_thick = max(1, int(round(self.banner_h / 40.0)))
        # Chừa chỗ bên phải banner cho dòng "score 0.93"
        (sw, _), _ = cv2.getTextSize("score 0.00", cv2.FONT_HERSHEY_SIMPLEX, self.score_scale, self.score_thick)
        self.score_area = (sw + self.banner_h // 2) if show_score else 0

        self._color, self._alpha = self._render_banner(warning_text, find_font(font_path))

    # ------------------------------------------------------------------ render 1 lần
    def _render_banner(self, text: str, font: Optional[str]):
        bh, w = self.banner_h, self.w
        # Lớp màu (BGR) + alpha theo từng pixel. Nền đỏ alpha 0.72.
        color = np.zeros((bh, w, 3), dtype=np.float32)
        color[:] = (0, 0, 200)
        alpha = np.full((bh, w, 1), 0.72, dtype=np.float32)

        # Icon tam giác cảnh báo màu vàng + dấu "!"
        pad = int(bh * 0.15)
        tri_h = bh - 2 * pad
        x0 = int(bh * 0.3)
        pts = np.array([[x0, pad + tri_h], [x0 + tri_h, pad + tri_h], [x0 + tri_h // 2, pad]], np.int32)
        icon = np.zeros((bh, w), dtype=np.uint8)
        cv2.fillPoly(icon, [pts], 255, lineType=cv2.LINE_AA)
        m = icon[..., None].astype(np.float32) / 255.0
        color = color * (1 - m) + np.array((0, 215, 255), np.float32) * m
        alpha = np.maximum(alpha, m)
        cx = x0 + tri_h // 2
        ex = np.zeros((bh, w), dtype=np.uint8)
        t = max(2, tri_h // 9)
        cv2.line(ex, (cx, pad + int(tri_h * 0.35)), (cx, pad + int(tri_h * 0.68)), 255, t, cv2.LINE_AA)
        cv2.circle(ex, (cx, pad + int(tri_h * 0.82)), max(1, t // 2 + 1), 255, -1, cv2.LINE_AA)
        m = ex[..., None].astype(np.float32) / 255.0
        color = color * (1 - m)  # dấu "!" màu đen
        alpha = np.maximum(alpha, m)

        # Chữ cảnh báo
        text_x = x0 + tri_h + int(bh * 0.3)
        max_text_w = max(10, w - text_x - self.score_area - int(bh * 0.2))
        mask = self._text_mask(text, font, bh, w, text_x, max_text_w)
        m = mask[..., None].astype(np.float32) / 255.0
        color = color * (1 - m) + np.array((255, 255, 255), np.float32) * m
        alpha = np.maximum(alpha, m)
        return color, alpha

    def _text_mask(self, text, font, bh, w, text_x, max_text_w) -> np.ndarray:
        mask = np.zeros((bh, w), dtype=np.uint8)
        if font:
            try:
                from PIL import Image, ImageDraw, ImageFont

                size = int(bh * 0.5)
                while size > 8:
                    f = ImageFont.truetype(font, size)
                    l, t, r, b = f.getbbox(text)
                    if r - l <= max_text_w:
                        break
                    size -= 1
                img = Image.new("L", (w, bh), 0)
                ImageDraw.Draw(img).text((text_x - l, (bh - (b - t)) // 2 - t), text, font=f, fill=255)
                return np.array(img, dtype=np.uint8)
            except Exception as exc:  # PIL thiếu / font lỗi -> dùng chữ ASCII
                print(f"[overlay] Không render được font ({exc!r}), dùng chữ ASCII.", flush=True)
        scale = bh / 45.0
        thick = max(1, int(round(bh / 30.0)))
        while scale > 0.3:
            (tw, th), _ = cv2.getTextSize(_FALLBACK_TEXT, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            if tw <= max_text_w:
                break
            scale -= 0.05
        cv2.putText(mask, _FALLBACK_TEXT, (text_x, (bh + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, 255, thick, cv2.LINE_AA)
        return mask

    # ------------------------------------------------------------------ mỗi frame
    def draw(self, frame: np.ndarray, frame_idx: int, is_fight: bool, score: float) -> np.ndarray:
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            return frame
        if is_fight:
            # Viền đỏ nhấp nháy (~1.25 Hz) để thu hút chú ý, banner đứng yên cho dễ đọc
            t = frame_idx / self.fps
            if not self.blink or (t % 0.8) < 0.5:
                cv2.rectangle(frame, (0, 0), (self.w - 1, self.h - 1), (0, 0, 255), self.border * 2)
            roi = frame[: self.banner_h].astype(np.float32)
            frame[: self.banner_h] = (roi * (1 - self._alpha) + self._color * self._alpha).astype(np.uint8)
            if self.show_score:
                txt = f"score {score:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, self.score_scale, self.score_thick)
                org = (self.w - tw - self.banner_h // 4, (self.banner_h + th) // 2)
                cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, self.score_scale,
                            (255, 255, 255), self.score_thick, cv2.LINE_AA)
        elif self.show_hud:
            txt = f"fight score {score:.2f}"
            s = self.score_scale * 0.7
            (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, s, self.score_thick)
            p = 8
            cv2.rectangle(frame, (p, p), (p + tw + 2 * p, p + th + base + 2 * p), (40, 40, 40), -1)
            cv2.putText(frame, txt, (2 * p, p + th + p), cv2.FONT_HERSHEY_SIMPLEX, s,
                        (0, 220, 0), self.score_thick, cv2.LINE_AA)
        return frame

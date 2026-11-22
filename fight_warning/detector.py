"""Wrapper ONNX cho model PaddleHuman ppTSM (fight / non-fight).

Model lấy từ serving/model_engine/violence_paddlehuman/weights/ppTSM.onnx của repo gốc.
  - Input : [N, 8, 3, 224, 224] float32, RGB, đã chuẩn hoá mean/std ImageNet.
  - Output: [N, 2] logits, class 0 = non-fight, class 1 = fight -> cần softmax.
Tiền xử lý giữ đúng như serving gốc: resize cạnh ngắn = 256 với tỉ lệ cố định 4:3, rồi center crop 224.
"""
from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3, 1, 1)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _preload_cuda_libs(ort) -> None:
    """Giúp onnxruntime-gpu tìm thấy CUDA/cuDNN trên Colab (lib đi kèm torch cài qua pip)."""
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
            return
        except Exception:
            pass
    try:  # onnxruntime-gpu bản cũ: import torch để nạp sẵn libcudnn/libcublas vào process
        import torch  # noqa: F401
    except Exception:
        pass


class FightDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        short_size: int = 256,
        target_size: int = 224,
        positive_class_id: int = 1,
        cpu_threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        self.short_size = int(short_size)
        self.target_size = int(target_size)
        self.positive_class_id = int(positive_class_id)

        device = (device or "auto").strip().lower()
        available = ort.get_available_providers()
        want_cuda = device in ("auto", "cuda", "gpu") and "CUDAExecutionProvider" in available
        if want_cuda:
            _preload_cuda_libs(ort)
            providers: List = [
                ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        if cpu_threads and cpu_threads > 0:
            so.intra_op_num_threads = int(cpu_threads)
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.providers = self.session.get_providers()
        self.on_gpu = "CUDAExecutionProvider" in self.providers

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name
        # Số frame model nhận trong 1 clip (ppTSM = 8). Đọc từ shape để không hard-code.
        self.num_seg = inp.shape[1] if isinstance(inp.shape[1], int) else 8

        if device in ("cuda", "gpu") and not self.on_gpu:
            print("[detector] CẢNH BÁO: yêu cầu GPU nhưng onnxruntime đang chạy CPU "
                  f"(available={available}). Kiểm tra lại onnxruntime-gpu / runtime GPU.", flush=True)
        print(f"[detector] model={model_path} providers={self.providers} num_seg={self.num_seg}", flush=True)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame gốc -> crop RGB uint8 [224, 224, 3] (chưa chuẩn hoá, để tiết kiệm RAM)."""
        h, w = frame_bgr.shape[:2]
        if w <= h:
            ow, oh = self.short_size, int(self.short_size * 4.0 / 3.0)
        else:
            oh, ow = self.short_size, int(self.short_size * 4.0 / 3.0)
        # INTER_AREA khi thu nhỏ: gần với PIL BILINEAR (có khử răng cưa) mà PaddleVideo dùng lúc train.
        interp = cv2.INTER_AREA if (ow < w and oh < h) else cv2.INTER_LINEAR
        img = cv2.resize(frame_bgr, (ow, oh), interpolation=interp)
        left = int(round((ow - self.target_size) / 2.0))
        top = int(round((oh - self.target_size) / 2.0))
        img = img[top:top + self.target_size, left:left + self.target_size]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def predict(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        """clips: list các mảng uint8 [T, 224, 224, 3] -> mảng xác suất fight, shape [N]."""
        if len(clips) == 0:
            return np.zeros((0,), dtype=np.float32)
        x = np.stack(clips).astype(np.float32) / 255.0          # [N, T, H, W, C]
        x = x.transpose(0, 1, 4, 2, 3)                           # [N, T, C, H, W]
        x = np.ascontiguousarray((x - _MEAN) / _STD, dtype=np.float32)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        probs = _softmax(logits.reshape(len(clips), -1))
        return probs[:, self.positive_class_id].astype(np.float32)

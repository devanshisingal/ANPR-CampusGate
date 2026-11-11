"""Lightweight pure ONNX Runtime runner for YOLOv8 detection models.

Replaces the Ultralytics + PyTorch runtime wrapper when loading `.onnx`
weights on CPU or GPU, avoiding the ~15-20s startup import overhead and ~300 MB
PyTorch memory footprint while preserving Ultralytics letterbox and class-aware
NMS geometry.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("detection.onnx_yolo")

_NMS_CLASS_OFFSET = 7680.0


@dataclass(frozen=True)
class RawYoloBox:
    """Single decoded ONNX detection in original image coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_label: str


def _parse_names(metadata: dict[str, str]) -> dict[int, str]:
    raw = metadata.get("names")
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return {int(k): str(v) for k, v in parsed.items()}
        if isinstance(parsed, (list, tuple)):
            return {idx: str(name) for idx, name in enumerate(parsed)}
    except Exception as exc:
        _logger.debug("Could not parse ONNX class names metadata: %s", exc)
    return {}


def _letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    *,
    auto: bool,
    stride: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize and pad an image while keeping aspect ratio, matching Ultralytics."""
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = new_shape
    ratio = min(target_h / max(orig_h, 1), target_w / max(orig_w, 1))

    new_unpad_w = max(1, int(round(orig_w * ratio)))
    new_unpad_h = max(1, int(round(orig_h * ratio)))

    dw = float(target_w - new_unpad_w)
    dh = float(target_h - new_unpad_h)
    if auto and stride > 0:
        dw = float(dw % stride)
        dh = float(dh % stride)

    dw *= 0.5
    dh *= 0.5

    if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
        resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    top = max(0, int(round(dh - 0.1)))
    bottom = max(0, int(round(dh + 0.1)))
    left = max(0, int(round(dw - 0.1)))
    right = max(0, int(round(dw + 0.1)))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, (dw, dh)


class OnnxYoloRunner:
    """Direct ``onnxruntime.InferenceSession`` runner for exported YOLOv8 ONNX models."""

    def __init__(
        self,
        model_path: str,
        *,
        imgsz: int = 640,
        device: str | int = "cpu",
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: int = 1,
    ) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.device = device

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = (
            max(1, int(intra_op_num_threads))
            if intra_op_num_threads is not None
            else min(4, os.cpu_count() or 4)
        )
        sess_opts.intra_op_num_threads = threads
        sess_opts.inter_op_num_threads = max(1, int(inter_op_num_threads))

        avail = ort.get_available_providers()
        want_cuda = str(device).strip().lower() != "cpu" and "CUDAExecutionProvider" in avail
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if want_cuda
            else ["CPUExecutionProvider"]
        )

        self._session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self._input_meta = self._session.get_inputs()[0]
        self._input_name = self._input_meta.name
        self._output_names = [o.name for o in self._session.get_outputs()]

        meta = self._session.get_modelmeta().custom_metadata_map or {}
        self.names: dict[int, str] = _parse_names(meta)
        try:
            self.stride = max(1, int(meta.get("stride", 32)))
        except (TypeError, ValueError):
            self.stride = 32
        self._end2end = str(meta.get("end2end", "false")).strip().lower() == "true"

        in_shape = self._input_meta.shape
        if len(in_shape) == 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
            self._target_hw = (int(in_shape[2]), int(in_shape[3]))
            self._auto_rect = False
        else:
            self._target_hw = (self.imgsz, self.imgsz)
            self._auto_rect = True

        self._is_fp16 = "float16" in str(self._input_meta.type).lower()
        self.active_provider = (
            self._session.get_providers()[0] if self._session.get_providers() else "CPUExecutionProvider"
        )

    def warmup(self) -> None:
        """Run one synthetic pass so first-frame ORT setup completes during startup."""
        dummy_h, dummy_w = self._target_hw
        dummy = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        self.predict(dummy, conf_threshold=0.9, iou_threshold=0.5)

    def close(self) -> None:
        self._session = None

    def predict(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float,
        iou_threshold: float = 0.5,
        allowed_classes: Optional[Iterable[str]] = None,
    ) -> list[RawYoloBox]:
        """Run detection on one BGR image and return unclipped/clipped xyxy boxes."""
        if self._session is None or image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]
        padded, ratio, (dw, dh) = _letterbox(
            image,
            self._target_hw,
            auto=self._auto_rect,
            stride=self.stride,
        )

        # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        chw = padded[:, :, ::-1].transpose(2, 0, 1)
        dtype = np.float16 if self._is_fp16 else np.float32
        blob = np.ascontiguousarray(chw, dtype=dtype) * dtype(1.0 / 255.0)
        batch = blob[None, ...]

        raw_out = self._session.run(self._output_names, {self._input_name: batch})[0]
        arr = np.asarray(raw_out, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]

        allowed_lower = {c.lower() for c in allowed_classes} if allowed_classes is not None else None

        if self._end2end or (arr.ndim == 2 and arr.shape[1] == 6 and arr.shape[0] != 6):
            return self._decode_end2end(arr, orig_w, orig_h, ratio, dw, dh, conf_threshold, allowed_lower)

        # Ultralytics YOLOv8 output is (4 + num_classes, anchors)
        expected_rows = (4 + max(self.names) + 1) if self.names else 0
        if expected_rows > 4 and arr.shape[0] == expected_rows and arr.shape[1] != expected_rows:
            pred = arr.T
        elif expected_rows > 4 and arr.shape[1] == expected_rows and arr.shape[0] != expected_rows:
            pred = arr
        elif arr.shape[0] < arr.shape[1]:
            pred = arr.T
        else:
            pred = arr

        if pred.shape[1] < 5:
            return []

        cls_scores = pred[:, 4:]
        if cls_scores.shape[1] == 1:
            confs = cls_scores[:, 0]
            keep = confs >= conf_threshold
            if not np.any(keep):
                return []
            pred_kept = pred[keep]
            confs_kept = confs[keep]
            cls_ids = np.zeros(len(confs_kept), dtype=np.int64)
        else:
            max_scores = cls_scores.max(axis=1)
            keep = max_scores >= conf_threshold
            if not np.any(keep):
                return []
            pred_kept = pred[keep]
            scores_kept = pred_kept[:, 4:]
            cls_ids = np.argmax(scores_kept, axis=1)
            confs_kept = scores_kept[np.arange(len(pred_kept)), cls_ids]

        if allowed_lower is not None and self.names:
            allowed_mask = np.array(
                [self.names.get(int(cid), "").lower() in allowed_lower for cid in cls_ids],
                dtype=bool,
            )
            if not np.any(allowed_mask):
                return []
            pred_kept = pred_kept[allowed_mask]
            confs_kept = confs_kept[allowed_mask]
            cls_ids = cls_ids[allowed_mask]

        cx = pred_kept[:, 0]
        cy = pred_kept[:, 1]
        bw = pred_kept[:, 2]
        bh = pred_kept[:, 3]
        half_w = bw * 0.5
        half_h = bh * 0.5

        inv_ratio = 1.0 / ratio if ratio > 0 else 1.0
        x1 = np.clip((cx - half_w - dw) * inv_ratio, 0.0, float(orig_w))
        y1 = np.clip((cy - half_h - dh) * inv_ratio, 0.0, float(orig_h))
        x2 = np.clip((cx + half_w - dw) * inv_ratio, 0.0, float(orig_w))
        y2 = np.clip((cy + half_h - dh) * inv_ratio, 0.0, float(orig_h))

        # Offset per class so cv2.dnn.NMSBoxes performs class-aware NMS matching Ultralytics.
        offsets = cls_ids.astype(np.float32) * _NMS_CLASS_OFFSET
        nms_boxes = np.column_stack([x1 + offsets, y1 + offsets, x2 - x1, y2 - y1]).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, confs_kept.tolist(), float(conf_threshold), float(iou_threshold)
        )
        if len(indices) == 0:
            return []

        flat_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        results: list[RawYoloBox] = []
        for idx in flat_indices:
            if x2[idx] <= x1[idx] or y2[idx] <= y1[idx]:
                continue
            cid = int(cls_ids[idx])
            label = self.names.get(cid, str(cid)).lower()
            results.append(
                RawYoloBox(
                    bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
                    confidence=float(confs_kept[idx]),
                    class_id=cid,
                    class_label=label,
                )
            )
        return results

    def _decode_end2end(
        self,
        arr: np.ndarray,
        orig_w: int,
        orig_h: int,
        ratio: float,
        dw: float,
        dh: float,
        conf_threshold: float,
        allowed_lower: Optional[set[str]],
    ) -> list[RawYoloBox]:
        inv_ratio = 1.0 / ratio if ratio > 0 else 1.0
        results: list[RawYoloBox] = []
        for row in arr:
            conf = float(row[4])
            if conf < conf_threshold:
                continue
            cid = int(row[5])
            label = self.names.get(cid, str(cid)).lower()
            if allowed_lower is not None and label not in allowed_lower:
                continue
            x1 = float(np.clip((row[0] - dw) * inv_ratio, 0.0, float(orig_w)))
            y1 = float(np.clip((row[1] - dh) * inv_ratio, 0.0, float(orig_h)))
            x2 = float(np.clip((row[2] - dw) * inv_ratio, 0.0, float(orig_w)))
            y2 = float(np.clip((row[3] - dh) * inv_ratio, 0.0, float(orig_h)))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                RawYoloBox(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cid,
                    class_label=label,
                )
            )
        return results


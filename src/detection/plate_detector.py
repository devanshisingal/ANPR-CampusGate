"""
License plate detector for the ALPR University Gate system.

Supports single-model detection and multi-model Ensemble Fusion
(Weighted Box Fusion / Max-Confidence NMS) to merge predictions across
baseline and fine-tuned models for optimal accuracy and recall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("detection.plate_detector")


def _box_iou(box1: np.ndarray | list, box2: np.ndarray | list) -> float:
    """Compute Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _fuse_detections(
    detections: list[dict],
    iou_threshold: float = 0.45,
    method: str = "wbf",
    agreement_boost: float = 0.05,
) -> list[dict]:
    """Fuse multi-model bounding boxes using Weighted Box Fusion (WBF) or Max-Confidence.

    Args:
        detections: List of dicts, each with keys: 'box' (x1, y1, x2, y2), 'conf', 'model_idx'
        iou_threshold: IoU overlap threshold to consider two detections the same plate.
        method: 'wbf' (Weighted Box Fusion) or 'max_conf' (take box of most confident model).
        agreement_boost: Small confidence boost when multiple models independently detect the same plate.

    Returns:
        List of fused detection dicts: {'box': [x1, y1, x2, y2], 'conf': float, 'agreeing_models': int}
    """
    if not detections:
        return []

    # Sort descending by confidence
    dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    clusters: list[list[dict]] = []

    for d in dets:
        matched_cluster = None
        for cluster in clusters:
            # Check IoU with the cluster's leader or cluster average
            leader_box = cluster[0]["box"]
            if _box_iou(d["box"], leader_box) >= iou_threshold:
                matched_cluster = cluster
                break
        if matched_cluster is not None:
            matched_cluster.append(d)
        else:
            clusters.append([d])

    fused_results: list[dict] = []
    for cluster in clusters:
        if len(cluster) == 1:
            # Single-model detection (no conflict)
            fused_results.append({
                "box": [round(float(v), 2) for v in cluster[0]["box"]],
                "conf": float(cluster[0]["conf"]),
                "agreeing_models": 1,
            })
            continue

        confs = np.array([d["conf"] for d in cluster], dtype=np.float32)
        boxes = np.array([d["box"] for d in cluster], dtype=np.float32)

        if method == "max_conf":
            best_idx = int(np.argmax(confs))
            fused_box = boxes[best_idx].tolist()
        else:
            # Weighted Box Fusion (WBF): average coordinates weighted by model confidences
            weights = confs / (np.sum(confs) + 1e-6)
            fused_box = np.sum(boxes * weights[:, None], axis=0).tolist()

        # When multiple independent models agree, reward with small agreement boost (capped at 1.0)
        fused_conf = min(1.0, float(np.max(confs)) + agreement_boost)

        fused_results.append({
            "box": [round(float(v), 2) for v in fused_box],
            "conf": fused_conf,
            "agreeing_models": len(cluster),
        })

    # Sort fused results by confidence descending
    fused_results.sort(key=lambda d: d["conf"], reverse=True)
    return fused_results


class PlateDetector:
    """Detect license plates within a vehicle crop using Single or Ensemble YOLO models.

    Args:
        model_paths:          Path or list of paths to trained plate detection weights.
        confidence_threshold: Minimum confidence to keep a plate detection.
        imgsz:                Inference image size (320 standard for vehicle crops).
        device:               Execution device ('cpu' or CUDA device index).
        half:                 Use FP16 half precision (CUDA only).
        iou_threshold:        Ensemble IoU threshold for matching boxes.
        fusion_method:        'wbf' (Weighted Box Fusion) or 'max_conf'.
    """

    def __init__(
        self,
        model_paths: Union[str, Path, List[Union[str, Path]]],
        confidence_threshold: float = 0.4,
        imgsz: int = 320,
        device: str | int = 0,
        half: bool = False,
        iou_threshold: float = 0.45,
        fusion_method: str = "wbf",
    ) -> None:
        if isinstance(model_paths, (str, Path)):
            self.model_paths = [str(model_paths)]
        else:
            self.model_paths = [str(p) for p in model_paths]

        # Backward compatibility attribute
        self.model_path = self.model_paths[0] if self.model_paths else ""
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.iou_threshold = iou_threshold
        self.fusion_method = fusion_method
        self._models: list[Any] = []
        self._predict_kwargs_list: list[dict] = []

        for mp in self.model_paths:
            kwargs = (
                {"imgsz": self.imgsz}
                if Path(mp).suffix.lower() in {".engine", ".onnx"}
                else {"imgsz": self.imgsz, "device": self.device, "half": self.half}
            )
            self._predict_kwargs_list.append(kwargs)

    @classmethod
    def from_config(cls, config: dict) -> "PlateDetector":
        det_cfg = config.get("detection", {})
        
        # Check ensemble configuration
        ens_cfg = det_cfg.get("plate_ensemble", {})
        if isinstance(ens_cfg, dict):
            ensemble_enabled = bool(ens_cfg.get("enabled", False))
            configured_paths = ens_cfg.get("models") or det_cfg.get("plate_model_paths")
            iou_thresh = float(ens_cfg.get("iou_threshold", 0.45))
            fusion_method = str(ens_cfg.get("fusion_strategy", "wbf"))
        else:
            ensemble_enabled = bool(ens_cfg or det_cfg.get("ensemble", False))
            configured_paths = det_cfg.get("plate_model_paths")
            iou_thresh = float(det_cfg.get("plate_iou_threshold", 0.45))
            fusion_method = str(det_cfg.get("plate_fusion_method", "wbf"))

        # Normalize fusion method name
        if "high" in fusion_method.lower():
            fusion_method = "max_conf"
        elif "weight" in fusion_method.lower():
            fusion_method = "wbf"

        if ensemble_enabled and configured_paths and isinstance(configured_paths, list):
            valid_paths = [p for p in configured_paths if Path(p).exists()]
            model_paths = valid_paths if valid_paths else [det_cfg["plate_model_path"]]
        else:
            model_paths = [det_cfg["plate_model_path"]]

        return cls(
            model_paths=model_paths,
            confidence_threshold=float(det_cfg.get("plate_confidence", 0.4)),
            imgsz=int(det_cfg.get("plate_imgsz", 320)),
            device=det_cfg.get("device", "cpu"),
            half=bool(det_cfg.get("half", False)),
            iou_threshold=iou_thresh,
            fusion_method=fusion_method,
        )

    def _load_model(self) -> None:
        """Lazy-load all models in the ensemble on first use and warm them up."""
        if self._models:
            return

        from ultralytics import YOLO

        loaded_models = []
        for idx, (m_path, kwargs) in enumerate(zip(self.model_paths, self._predict_kwargs_list)):
            try:
                m = YOLO(m_path)
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                m(dummy, verbose=False, **kwargs)
                loaded_models.append(m)
                _logger.info(
                    "Plate detector [%d/%d] loaded from '%s' (imgsz=%d device=%s half=%s)",
                    idx + 1, len(self.model_paths), m_path, self.imgsz, self.device, self.half,
                )
            except Exception as exc:
                _logger.error("Failed to load plate detector from '%s': %s", m_path, exc)
                raise

        self._models = loaded_models
        if len(self._models) > 1:
            _logger.info(
                "🚀 Ensemble Fusion active with %d models (method='%s', iou_thresh=%.2f)",
                len(self._models), self.fusion_method, self.iou_threshold,
            )

    @property
    def _model(self):
        """Backward compatibility for external access to the primary model."""
        if not self._models:
            self._load_model()
        return self._models[0] if self._models else None

    def close(self) -> None:
        """Release loaded models and free memory."""
        models = self._models
        self._models = []
        for model in models:
            try:
                inner = getattr(model, "model", None)
                if inner is not None and hasattr(inner, "cpu"):
                    inner.cpu()
                del inner, model
            except Exception as exc:
                _logger.warning("Failed to release detector component: %s", exc)

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def detect_boxes(self, vehicle_crop: np.ndarray) -> list[dict]:
        """Detect license plates within a vehicle crop and return fused box metadata.

        Returns:
            List of dicts: {'box': [x1, y1, x2, y2], 'conf': float, 'agreeing_models': int}
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        self._load_model()
        h, w = vehicle_crop.shape[:2]
        raw_detections: list[dict] = []

        for model_idx, (model, kwargs) in enumerate(zip(self._models, self._predict_kwargs_list)):
            try:
                results = model(vehicle_crop, verbose=False, **kwargs)
            except Exception as exc:
                _logger.warning("Inference failed for model %d (%s): %s", model_idx, self.model_paths[model_idx], exc)
                continue

            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf < self.confidence_threshold:
                        continue
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1 = max(0.0, float(xyxy[0]))
                    y1 = max(0.0, float(xyxy[1]))
                    x2 = min(float(w), float(xyxy[2]))
                    y2 = min(float(h), float(xyxy[3]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    raw_detections.append({
                        "box": [x1, y1, x2, y2],
                        "conf": conf,
                        "model_idx": model_idx,
                    })

        # Apply multi-model Weighted Box Fusion or NMS
        if len(self._models) > 1:
            fused = _fuse_detections(
                raw_detections,
                iou_threshold=self.iou_threshold,
                method=self.fusion_method,
            )
        else:
            fused = [{"box": d["box"], "conf": d["conf"], "agreeing_models": 1} for d in raw_detections]

        return fused

    def detect(self, vehicle_crop: np.ndarray) -> list[np.ndarray]:
        """Detect license plates within a vehicle crop.

        Args:
            vehicle_crop: BGR image crop of a detected vehicle (H, W, 3).

        Returns:
            List of plate crop images (NumPy arrays). Empty list if no plate
            detected above the confidence threshold.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        fused_boxes = self.detect_boxes(vehicle_crop)
        h, w = vehicle_crop.shape[:2]
        plate_crops: list[np.ndarray] = []

        for item in fused_boxes:
            b = item["box"]
            x1 = max(0, int(b[0]))
            y1 = max(0, int(b[1]))
            x2 = min(w, int(b[2]))
            y2 = min(h, int(b[3]))

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = vehicle_crop[y1:y2, x1:x2]
            if plate_crop.size > 0:
                plate_crops.append(plate_crop)

        return plate_crops

"""
Model Comparison and Ensemble Benchmark Script for ALPR University Gate.

Compares:
  1. Baseline Model (best.pt)
  2. Fine-tuned Campus Model (best_finetuned.pt)
  3. Ensemble Fusion (Weighted Box Fusion)

Usage:
  python scripts/compare_models.py --source mycarplate.mp4 --frames 100
  python scripts/compare_models.py --source mycarplate.mp4 --output output/comparison.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.utils.config import load_config
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.ocr import create_ocr_engine


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Fine-tuned vs Ensemble Detection")
    parser.add_argument("--source", type=str, default="mycarplate.mp4", help="Video source path")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file")
    parser.add_argument("--frames", type=int, default=200, help="Max frames to process (0 = all)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Optional output video path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 65)
    print("🚗 ALPR DUAL-MODEL & ENSEMBLE BENCHMARK")
    print("=" * 65)

    # 1. Initialize Detectors
    print("Initializing Vehicle Detector...")
    vd = VehicleDetector.from_config(cfg)

    print("Initializing Model 1: Baseline (best.pt)...")
    d_base = PlateDetector(
        model_paths="models/plate_detector/best.pt",
        confidence_threshold=args.conf,
        device=cfg.get("detection", {}).get("device", "cpu"),
        imgsz=320,
    )

    print("Initializing Model 2: Fine-Tuned (best_finetuned.pt)...")
    d_fine = PlateDetector(
        model_paths="models/plate_detector/best_finetuned.pt",
        confidence_threshold=args.conf,
        device=cfg.get("detection", {}).get("device", "cpu"),
        imgsz=320,
    )

    print("Initializing Ensemble Fusion (WBF: Baseline + Fine-Tuned)...")
    d_ens = PlateDetector(
        model_paths=[
            "models/plate_detector/best.pt",
            "models/plate_detector/best_finetuned.pt",
        ],
        confidence_threshold=args.conf,
        device=cfg.get("detection", {}).get("device", "cpu"),
        imgsz=320,
        fusion_method="wbf",
    )

    ocr = create_ocr_engine(cfg.get("ocr", {}).get("backend", "rapidocr"), cfg)

    # 2. Open Video Source
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video source '{args.source}'")
        return

    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    stats = {
        "frames_processed": 0,
        "base_detections": 0,
        "fine_detections": 0,
        "ens_detections": 0,
        "base_confs": [],
        "fine_confs": [],
        "ens_confs": [],
        "ocr_reads": set(),
    }

    print(f"\nProcessing video: {args.source} ...")
    start_t = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if args.frames > 0 and frame_idx > args.frames:
            break

        vehicles = vd.detect(frame)
        for v in vehicles:
            crop = v.crop if hasattr(v, "crop") and v.crop.size > 0 else frame
            
            # Run all 3 detection setups
            b_base = d_base.detect_boxes(crop)
            b_fine = d_fine.detect_boxes(crop)
            b_ens = d_ens.detect_boxes(crop)

            stats["base_detections"] += len(b_base)
            stats["fine_detections"] += len(b_fine)
            stats["ens_detections"] += len(b_ens)

            for b in b_base: stats["base_confs"].append(b["conf"])
            for b in b_fine: stats["fine_confs"].append(b["conf"])
            for b in b_ens: stats["ens_confs"].append(b["conf"])

            # OCR on Ensemble Fused crops
            for item in b_ens:
                box = item["box"]
                x1, y1, x2, y2 = map(int, box)
                pcrop = crop[max(0, y1):min(crop.shape[0], y2), max(0, x1):min(crop.shape[1], x2)]
                if pcrop.size > 0:
                    text, conf = ocr.recognize(pcrop)
                    if text and len(text) >= 4:
                        stats["ocr_reads"].add(text)

        if writer:
            writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx} | Base: {stats['base_detections']} | Fine: {stats['fine_detections']} | Fused: {stats['ens_detections']}")

    cap.release()
    if writer:
        writer.release()
        print(f"✅ Comparison video saved to: {args.output}")

    total_time = time.time() - start_t
    stats["frames_processed"] = frame_idx

    print("\n" + "=" * 65)
    print("📊 BENCHMARK & ENSEMBLE SUMMARY REPORT")
    print("=" * 65)
    print(f"Total Frames Processed     : {stats['frames_processed']} in {total_time:.1f}s ({stats['frames_processed']/(total_time or 1):.1f} fps)")
    print("-" * 65)
    print(f"1. Baseline Model (best.pt):")
    print(f"   - Plates Detected       : {stats['base_detections']}")
    print(f"   - Average Confidence    : {np.mean(stats['base_confs']):.3f}" if stats['base_confs'] else "   - Average Confidence    : N/A")
    print(f"2. Fine-Tuned (best_finetuned.pt):")
    print(f"   - Plates Detected       : {stats['fine_detections']}")
    print(f"   - Average Confidence    : {np.mean(stats['fine_confs']):.3f}" if stats['fine_confs'] else "   - Average Confidence    : N/A")
    print(f"3. Ensemble Fusion (WBF):")
    print(f"   - Fused Plates Captured : {stats['ens_detections']}")
    print(f"   - Average Fused Conf    : {np.mean(stats['ens_confs']):.3f}" if stats['ens_confs'] else "   - Average Fused Conf    : N/A")
    print("-" * 65)
    print(f"Unique Plates Recognized   : {list(stats['ocr_reads'])}")
    print("=" * 65)


if __name__ == "__main__":
    main()

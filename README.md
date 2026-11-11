# 🚗 ANPR - Campus Gate & Multi-Camera Trajectory System

A production-quality **Automatic Number Plate Recognition (ANPR)**, access control, and **Multi-Camera Vehicle Trajectory Reconstruction** system engineered for campus security and gate monitoring.

The system detects vehicles in live RTSP video feeds or recorded surveillance clips, localizes Indian license plates using a fine-tuned YOLOv8 model accelerated via **ONNX Runtime / TensorRT**, extracts registration numbers with a high-throughput **RapidOCR (PP-OCRv4 ONNX)** engine, classifies vehicle types and body colors, checks campus **Whitelist/Blacklist** status with instant alerts, and reconstructs multi-camera journeys on an interactive **Next.js operations dashboard** backed by a **FastAPI** REST API.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Multi-Camera Trajectory Reconstruction](#-multi-camera-trajectory-reconstruction)
- [Project Structure](#-project-structure)
- [What Each File Does](#-what-each-file-does)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Model Setup & Weights](#-model-setup--weights)
- [Configuration Guide](#-configuration-guide)
- [Usage — How to Run](#-usage--how-to-run)
- [Next.js Operations Dashboard](#-nextjs-operations-dashboard)
- [Training & Model Evaluation](#-training--model-evaluation)
- [API Reference](#-api-reference)
- [Running Tests](#-running-tests)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
- [Tech Stack](#-tech-stack)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)

---

## ⚡ Features

### 🎯 High-Accuracy Detection & Tracking
- **Vehicle Detection**: YOLOv8n (COCO pretrained) with direct ONNX Runtime / TensorRT acceleration.
- **License Plate Detection**: Custom fine-tuned YOLOv8 (`best_finetuned.onnx` / `best.pt`) with **99.47% mAP50** on Indian plates (~6.3× faster than dual-model PyTorch ensembles).
- **Dimension Filtering**: Automatic aspect and dimension gating (`min_plate_width_px: 36`, `min_plate_height_px: 12`) to eliminate false-positive OCR processing on tiny background artifacts.
- **ByteTrack Tracking**: Consistent multi-vehicle tracking across frames with heavy-track throttling per frame (`live_max_heavy_tracks_per_frame: 1`, `max_heavy_tracks_per_frame: 2`) to sustain target frame rates under heavy traffic.

### 🔍 Optimized OCR & Validation Pipeline
- **RapidOCR Default Engine**: Powered by PP-OCRv4 ONNX via `onnxruntime` with DBNet crop size clamping (`det_limit_side_len: 320`) and angle-classifier bypass — achieves up to **~29× faster CPU OCR** without heavy PaddlePaddle dependencies.
- **Flexible OCR Alternatives**: Switchable in `config.yaml` to **TensorRT PARSeq** (ultra-low latency on Jetson / GPU), PaddleOCR, TrOCR, EasyOCR, or multi-engine ensemble voting.
- **Adaptive Lighting & Preprocessing**: Dynamic gamma correction for daytime glare and nighttime darkness, CLAHE contrast tuning, unsharp masking, and clean 64px height upscaling before OCR.
- **Real-ESRGAN ×4 Super-Resolution**: Upscales small plate crops on demand when configured or running on high-power GPUs.
- **Multi-Frame OCR Fusion with Early Exit**: Majority-voting fusion over rolling frame windows, with instant **early-exit flush** once 3 uncontested high-confidence reads (≥ 0.82) match.
- **Indian Plate Validation**: Regex validation and positional alphanumeric repair (correcting `O`↔`0`, `I`↔`1`, `Z`↔`2`, `S`↔`5`) supporting standard state registrations and Bharat (BH) series plates.

### 🛡️ Campus Security & Access Control
- **Whitelist & Blacklist Access Control**: Live status evaluation (`Authorized`, `Blacklisted`, `Unknown`) with YAML/DB persistence and instant visual dashboard alerts for unauthorized or blacklisted vehicles.
- **Plate Color & Vehicle Classification**: HSV color segmentation maps plate colors to legal vehicle categories (White → Private, Yellow → Commercial, Green → EV, Blue → Diplomatic, Black → Rental/Self-drive).
- **Vehicle Body Color Profiling**: Extracts vehicle body paint colors for visual profile matching.
- **Campus Direction Profiles**: Tailored directional tracking for gate cameras (e.g. `ch01` Main Gate 2, `ch02` Gate 2) using directional movement vectors, virtual crossing lines, and multi-frame trajectory confirmation.

### 🗺️ Multi-Camera Trajectories & Modern Operations UI
- **Journey Reconstruction**: Aggregates timestamps and sightings across multiple campus gates into sequential vehicle trajectories with leg travel times and path history.
- **Next.js 15+ Operations Dashboard**: Built with Next.js 15, React 19, Tailwind CSS, and Leaflet — featuring a live multi-camera wall, feed health monitors, vehicle trajectory route maps, event filters, plate crop galleries, access control manager, and CSV export.
- **FastAPI REST API**: Comprehensive endpoints for event querying, vehicle profiles, camera MJPEG streaming, live snapshots, and whitelist/blacklist management.

---

## 🏗️ System Architecture & Pipeline

```
                       Video File / Live RTSP Stream / Multi-Camera Feeds
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │    src/capture/frame_capture.py  │
                              │ (Threaded capture & reconnect)   │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ src/detection/vehicle_detector.py│ (YOLOv8n ONNX / TRT)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  src/tracking/vehicle_tracker.py │ (ByteTrack)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │    src/utils/motion_filter.py    │ (Skip stationary vehicles)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  src/detection/plate_detector.py │ (Fine-tuned YOLOv8 ONNX)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │src/preprocessing/plate_preproc.py│ (Adaptive lighting + CLAHE)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │src/enhancement/super_resolution.py (Real-ESRGAN / 64px scale)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │    src/ocr/rapidocr_engine.py    │ (RapidOCR PP-OCRv4 ONNX
                              │  (or PARSeq TRT / PaddleOCR)     │  tuned for CPU / GPU)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  src/ocr/ocr_postprocessor.py    │ (Positional fix: O↔0, I↔1)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │src/validation/plate_validator.py │ (Indian plate regex + BH)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │   src/validation/ocr_fusion.py   │ (Majority vote + Early exit)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  src/alerts/whitelist.py         │ (Whitelist / Blacklist check)
                              │  src/utils/direction_detector.py │ (Campus gate IN / OUT vector)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │        src/database/db.py        │ (SQLite / PostgreSQL)
                              └────────────────┬─────────────────┘
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
       ┌─────────────────────────────────┐           ┌─────────────────────────────────┐
       │       src/api/server.py         │           │        frontend/ (Next.js)      │
       │ FastAPI Endpoints, Trajectories │◀─────────▶│ Camera Wall, Leaflet Route Map, │
       │     & MJPEG Stream Servers      │           │    Access Lists & Analytics     │
       └─────────────────────────────────┘           └─────────────────────────────────┘
```

---

## 🌐 Multi-Camera Trajectory Reconstruction

When deployed across multiple checkpoints or campus gates, sightings from individual cameras are merged into chronological journey records:

```
CAM001 (Main Gate 2) ──┐
CAM002 (Gate 2) ───────┼──▶ SQLite / PostgreSQL ──▶ Trajectory Engine ──▶ Leaflet Map & Timeline
CAM003 (Academic) ─────┤    (camera attribution)     (trip legs & timing)   (interactive visualization)
CAM004 (Hostel Gate) ──┘
```

- **Camera Registry (`config/camera_config.yaml`)**: Stores IDs, names, geographic coordinates (WGS84 lat/long), and feed sources.
- **Directional Mapping**: Evaluates gate-specific vectors (`ch01` moves towards lower-right for IN; `ch02` moves towards lower-left for IN).
- **Trajectory Engine (`src/trajectory/engine.py`)**: Groups detections by plate and reconstructs paths, measuring interval speeds and transit durations between gates.
- For complete multi-camera specifications, see [`docs/MULTI_CAMERA_TRAJECTORY_SYSTEM.md`](docs/MULTI_CAMERA_TRAJECTORY_SYSTEM.md).

---

## 📁 Project Structure

```
anpr_project/
├── main.py                          # Pipeline entry point — runs full ANPR
├── requirements.txt                 # Pinned dependencies (RapidOCR, ONNX Runtime, etc.)
├── pytest.ini                       # Pytest test suite configuration
├── conftest.py                      # Root pytest fixtures
│
├── frontend/                        # Next.js 15 Operations Dashboard
│   ├── app/                         # App Router (Dashboard pages, live stream, styles)
│   │   ├── page.tsx                 # Main monitoring UI & multi-camera wall
│   │   ├── layout.tsx               # Root layout & theme wrapper
│   │   └── globals.css              # Dashboard styling
│   ├── next.config.ts               # Proxy rewrites routing to FastAPI backend
│   ├── package.json                 # Next.js dependencies (React 19, Lucide, Tailwind)
│   └── tsconfig.json                # TypeScript compiler configuration
│
├── config/
│   ├── config.yaml                  # Core pipeline configuration (models, OCR, fusion, gates)
│   ├── camera_config.yaml           # Multi-camera registry and GPS siting
│   └── whitelist.yaml               # Authorized & blacklisted plate registry
│
├── src/
│   ├── alerts/
│   │   ├── whitelist.py             # Whitelist & blacklist manager with disk persistence
│   │   └── blacklist.py             # Blacklist alert query helpers
│   ├── api/
│   │   ├── server.py                # FastAPI REST API (logs, search, stats, entry)
│   │   ├── trajectory_routes.py     # Trajectory, camera feed, and MJPEG stream endpoints
│   │   └── schemas.py               # Pydantic request and response schemas
│   ├── cameras/
│   │   ├── manager.py               # Multi-camera queue & concurrent ingestion coordinator
│   │   ├── registry.py              # Camera metadata loader from YAML
│   │   ├── sources.py               # Video file and RTSP source abstractions
│   │   ├── status_store.py          # Real-time pipeline processing state persistence
│   │   └── preview_server.py        # MJPEG live preview HTTP server
│   ├── capture/
│   │   └── frame_capture.py         # Threaded RTSP & video reader with auto-reconnection
│   ├── classification/
│   │   ├── color_classifier.py      # HSV plate color classification
│   │   ├── vehicle_classifier.py    # Plate color to vehicle type mapping
│   │   └── vehicle_color.py         # Vehicle body paint color extractor
│   ├── dashboard/
│   │   ├── app.py                   # Legacy Streamlit single-gate dashboard
│   │   └── trajectory_app.py        # Legacy Streamlit trajectory visualizer
│   ├── dashboard_web/               # Standalone vanilla HTML/JS web dashboard
│   ├── database/
│   │   ├── db.py                    # SQLAlchemy session management, CRUD, crop saving
│   │   ├── models.py                # Database models (VehicleEvent, Camera, etc.)
│   │   ├── duplicate_filter.py      # Duplicate event suppressor (Levenshtein edit distance)
│   │   ├── vehicle_profiles.py      # Aggregated vehicle sighting profiles
│   │   └── postgres_schema.py       # PostgreSQL schema compatibility
│   ├── detection/
│   │   ├── _onnx_yolo.py            # High-speed ONNX Runtime YOLO runner with OpenCV fallback
│   │   ├── plate_detector.py        # Fine-tuned YOLOv8 plate detector (ONNX / PyTorch)
│   │   └── vehicle_detector.py      # YOLOv8n vehicle detector (ONNX / PyTorch)
│   ├── enhancement/
│   │   └── super_resolution.py      # Real-ESRGAN ×4 super-resolution model
│   ├── mapping/
│   │   ├── leaflet.py               # Leaflet map HTML & GeoJSON builder
│   │   └── heatmap.py               # Campus density heatmap generation
│   ├── ocr/
│   │   ├── base.py                  # Abstract base class for OCR engines
│   │   ├── rapidocr_engine.py       # RapidOCR PP-OCRv4 ONNX engine (default, CPU-tuned)
│   │   ├── parseq_tensorrt_engine.py# NVIDIA TensorRT PARSeq OCR engine
│   │   ├── paddle_ocr_engine.py     # PaddleOCR backend
│   │   ├── trocr_engine.py          # Microsoft TrOCR transformer backend
│   │   ├── easyocr_engine.py        # EasyOCR backend
│   │   ├── ensemble_engine.py       # Multi-engine voting ensemble
│   │   └── ocr_postprocessor.py     # Strips IND/INDIA, fixes O↔0, normalizes plates
│   ├── preprocessing/
│   │   └── plate_preprocessor.py    # Adaptive lighting, CLAHE, and sharpening
│   ├── tracking/
│   │   └── vehicle_tracker.py       # ByteTrack multi-object vehicle tracker
│   ├── trajectory/
│   │   ├── engine.py                # Multi-camera trajectory builder & leg computation
│   │   └── geojson.py               # Path GeoJSON serializer
│   ├── utils/
│   │   ├── config.py                # YAML configuration loader & validator
│   │   ├── direction_detector.py    # Campus-specific virtual line & directional vectors
│   │   ├── motion_filter.py         # Filters out stationary/parked vehicles
│   │   ├── live_frame.py            # Annotated live frame writer for streams
│   │   └── logger.py                # Rotating file and console logger
│   └── validation/
│       ├── plate_validator.py       # Indian license plate regex & positional corrector
│       └── ocr_fusion.py            # Multi-frame majority voting with early-exit flush
│
├── scripts/
│   ├── run_pipeline.py              # Full pipeline with camera management & live streaming
│   ├── run_alpr.py                  # Lightweight standalone CSV mode (no DB)
│   ├── video_output.py              # Annotated video generator with bounding boxes
│   ├── test_model.py                # Batch model validation on image directories
│   ├── compare_models.py            # Benchmark comparison between baseline & fine-tuned models
│   └── benchmark_ocr_backends.py    # Benchmarks RapidOCR, PARSeq, PaddleOCR, and TrOCR
│
├── tests/
│   ├── unit/                        # Unit tests for OCR, ONNX, tracking, and direction
│   └── integration/                 # Integration tests for API, pipeline, and database
│
└── docs/                            # In-depth architectural & deployment documentation
```

---

## 🔍 What Each File Does

| Component / File | Purpose & Responsibilities |
|---|---|
| `main.py` | Top-level execution wrapper invoking `scripts/run_pipeline.py`. |
| `src/detection/_onnx_yolo.py` | Direct ONNX Runtime wrapper providing fast YOLOv8 execution without PyTorch overhead. |
| `src/detection/plate_detector.py` | Detects plate regions using `best_finetuned.onnx` with size-filtering safeguards. |
| `src/detection/vehicle_detector.py` | Detects cars, buses, trucks, and motorcycles using `yolov8n.onnx`. |
| `src/ocr/rapidocr_engine.py` | High-speed PP-OCRv4 ONNX engine with DBNet limits tuned for minimal CPU latency. |
| `src/ocr/parseq_tensorrt_engine.py` | TensorRT 10/8 engine for PARSeq OCR on NVIDIA GPU / Jetson platforms. |
| `src/validation/ocr_fusion.py` | Accumulates OCR reads across frames with early exit on 3 matching high-confidence votes. |
| `src/validation/plate_validator.py` | Positional corrector and regex validator for standard Indian and Bharat (BH) plates. |
| `src/alerts/whitelist.py` | Loads, stores, and validates plates against Authorized and Blacklisted access lists. |
| `src/utils/direction_detector.py` | Campus directional vectors (`ch01` Main Gate 2, `ch02` Gate 2) for accurate IN/OUT detection. |
| `src/trajectory/engine.py` | Correlates sightings across campus cameras into chronologically ordered vehicle routes. |
| `src/mapping/leaflet.py` | Produces interactive Leaflet maps with route poly-lines, markers, and transit times. |
| `src/api/server.py` | Core FastAPI application serving query endpoints, statistics, and access control APIs. |
| `src/api/trajectory_routes.py` | Trajectory reconstruction, camera feed health, and MJPEG streaming routes. |
| `frontend/app/page.tsx` | Comprehensive Next.js operations dashboard for live camera walls, access logs, and alerts. |

---

## 📦 Prerequisites

- **Python**: `3.10`, `3.11`, or `3.12`
- **Node.js**: `v18.x` or higher (for the Next.js operations dashboard)
- **Hardware**:
  - **CPU**: Standard multi-core x86 or ARM CPU (RapidOCR + ONNX Runtime runs efficiently on CPU).
  - **GPU (Optional)**: NVIDIA GPU with CUDA 11.8 / 12.x or Jetson Orin Nano for TensorRT / Real-ESRGAN acceleration.

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/devanshisingal/ANPR-CampusGate.git
cd ANPR-CampusGate
```

### 2. Set Up Python Environment
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU Acceleration Note**:
> By default, `requirements.txt` installs CPU `onnxruntime`. To enable NVIDIA GPU acceleration on desktop or server:
> ```bash
> pip install onnxruntime-gpu==1.18.0
> ```
> For NVIDIA Jetson Orin (JetPack 6.x), refer to the onnxruntime index detailed inside `requirements.txt`.

### 4. Install Next.js Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 🧠 Model Setup & Weights

The repository utilizes optimized ONNX and PyTorch weights stored in `models/`:

```
models/
├── plate_detector/
│   ├── best_finetuned.onnx          # ← Fine-tuned plate detector (ONNX, recommended)
│   ├── best_finetuned.pt            # ← Fine-tuned plate detector (PyTorch)
│   └── best.pt                      # ← Baseline YOLOv8m plate detector
├── vehicle_detector/
│   ├── yolov8n.onnx                 # ← ONNX vehicle detector (COCO pretrained)
│   └── yolov8n.pt                   # ← Auto-downloaded by ultralytics if missing
└── realesrgan/
    └── RealESRGAN_x4plus.pth        # ← Super-resolution weights (optional)
```

### Downloading Real-ESRGAN Weights (Optional)
If you wish to use Real-ESRGAN super-resolution for very small or distant plates:
```bash
python -c "import urllib.request, os; os.makedirs('models/realesrgan', exist_ok=True); urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 'models/realesrgan/RealESRGAN_x4plus.pth'); print('Downloaded RealESRGAN weights.')"
```

---

## ⚙️ Configuration Guide

All core operational settings reside in `config/config.yaml`:

```yaml
video:
  source: "mycarplate.mp4"              # Video file, RTSP URL, or 0 for webcam
  frame_skip: 1                         # Process every frame (preserves tracking continuity)
  max_retries: 5                        # Reconnect attempts on stream disconnect

detection:
  device: "cpu"                         # "cpu" or GPU index (0)
  vehicle_model_path: "models/vehicle_detector/yolov8n.onnx"
  vehicle_confidence: 0.5
  plate_model_path: "models/plate_detector/best_finetuned.onnx"
  plate_confidence: 0.28
  plate_imgsz: 320
  min_plate_width_px: 36                # Skip specks smaller than 36x12 px
  min_plate_height_px: 12

tracking:
  lost_track_timeout: 60                # Frame timeout for lost tracks
  live_max_heavy_tracks_per_frame: 1    # Cap plate-detection calls per frame

preprocessing:
  adaptive_lighting: true               # Auto gamma correction for day/night glare
  target_height_px: 64                  # Upscale crop cleanly prior to CLAHE
  clahe_clip_limit: 2.0

enhancement:
  sr_threshold_px: 0                    # 0 to bypass SR on CPU; 80-120 on GPU with Real-ESRGAN

ocr:
  backend: "rapidocr"                   # "rapidocr" (default), "parseq_tensorrt", "paddleocr"
  rapidocr:
    det_limit_type: "max"
    det_limit_side_len: 320             # Clamps DBNet side length (~29x CPU speedup)
    use_cls: false                      # Skip angle classifier on upright gate cameras
    det_border_pad: 16

fusion:
  window_size: 5                        # Majority-vote window
  min_confidence: 0.60
  early_exit_votes: 3                   # Early flush when 3 uncontested reads agree
  early_exit_min_confidence: 0.82

direction:
  active_camera: "ch01"                 # Select campus profile: "ch01", "ch02", or "default"

database:
  path: "data/alpr.db"
  image_save_path: "data/plate_crops/"
  vehicle_image_save_path: "data/vehicle_crops/"
```

---

## 💻 Usage — How to Run

### Option 1: Full System (API + Next.js Dashboard)

Start the system in two terminals:

#### Terminal 1 — Start the FastAPI Backend
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```
- API Documentation (Swagger UI): `http://localhost:8000/docs`

#### Terminal 2 — Start the Next.js Operations Dashboard
```bash
cd frontend
npm run dev
```
- Open `http://localhost:3000` in your browser.

---

### Option 2: Run the Video Pipeline

Run the end-to-end ANPR pipeline directly on a recorded video or RTSP feed:

```bash
# Process video configured in config.yaml
python main.py

# Specify an alternate video or RTSP stream
python main.py --source "path/to/gate_feed.mp4"
python main.py --source "rtsp://admin:pass@192.168.1.50:554/stream1"
```

Events are saved to `data/alpr.db`, with plate crops saved in `data/plate_crops/` and live annotated preview frames in `data/live_frame.jpg`.

---

### Option 3: Multi-Camera Processing Queue

Process all registered campus cameras (from `config/camera_config.yaml`):

```bash
python scripts/run_pipeline.py
```

Runs sequential processing for recorded files and initiates live feeds concurrently, streaming status to the Next.js multi-camera wall.

---

### Option 4: Quick Batch CSV Mode (No Database)

Extract unique license plates from a video directly into a CSV file without database setup:

```bash
python scripts/run_alpr.py --source ALPR.mp4 --output output/results.csv --min-reads 3
```

---

### Option 5: Export Annotated Video

Generate an MP4 video overlaid with green vehicle bounding boxes, orange license plate boxes, and identified plate numbers:

```bash
python scripts/video_output.py --source ALPR.mp4 --output output/annotated.mp4
```

---

### Option 6: Legacy Streamlit Dashboards

For lightweight local inspection, the legacy Streamlit applications remain available:

```bash
# Gate log dashboard
streamlit run src/dashboard/app.py

# Trajectory visualization dashboard
streamlit run src/dashboard/trajectory_app.py
```

---

## 🖥️ Next.js Operations Dashboard

The dashboard at `frontend/` provides:

- **Live Multi-Camera Wall**: Real-time video/MJPEG feeds from all registered campus gates with live FPS, frame counters, and feed status indicators (`HEALTHY`, `STALLED`, `OFFLINE`).
- **Vehicle Access History & Filtering**: Filter entry/exit logs by date range, gate location, vehicle type, and IN/OUT direction.
- **Whitelist & Blacklist Management**: Manage campus access lists directly from the UI with real-time blacklist violation alerts.
- **Trajectory Journey Map**: Displays Leaflet-powered route reconstruction showing travel paths, transit speed estimates, and timestamps between cameras.
- **Vehicle Profile Popups**: View full-size vehicle captures, plate crops, body color classifications, and visit frequencies.
- **Data Export**: One-click CSV and Excel export of filtered security logs.

---

## 🔬 Training & Model Evaluation

To retrain the plate detection model on your own dataset:

```bash
# 1. Convert JSON annotation labels to YOLO format
python -m training.dataset_converter

# 2. (Optional) Run image augmentations (blur, glare, noise)
python -m training.augmentation

# 3. Train YOLOv8m (100 epochs)
python -m training.trainer

# 4. Evaluate mAP, precision, and recall on test split
python -m training.evaluator
```

Training and validation curves are saved to `models/plate_detector/train/`.

---

## 📡 API Reference

Start server: `uvicorn src.api.server:app --host 0.0.0.0 --port 8000`

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/logs` | Retrieve vehicle events with `start_date`, `end_date`, `direction`, and `camera_id` filters. |
| `GET` | `/search?plate=XX` | Search exact plate history with direction and date constraints. |
| `GET` | `/stats` | Entry and exit counters for specified date ranges or all-time totals. |
| `POST` | `/entry` | Programmatically register an entry/exit event. |

### Access Control (Whitelist & Blacklist)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/access-control/whitelist` | List all authorized license plates. |
| `POST` | `/access-control/whitelist` | Add a license plate to the authorized list. |
| `DELETE` | `/access-control/whitelist/{plate}` | Remove an authorized plate. |
| `GET` | `/access-control/blacklist` | List all blacklisted license plates. |
| `POST` | `/access-control/blacklist` | Add a license plate to the blacklist. |
| `DELETE` | `/access-control/blacklist/{plate}` | Remove a plate from the blacklist. |

### Multi-Camera & Trajectory Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/trajectory-api/cameras/feeds` | Feed health, active status, and stream URLs for all cameras. |
| `GET` | `/trajectory-api/cameras/{id}/snapshot` | Fetch the latest annotated JPEG frame for a camera. |
| `GET` | `/trajectory-api/cameras/{id}/stream` | Continuous MJPEG stream for a camera. |
| `GET` | `/trajectory-api/trajectories/{plate}` | Reconstruct ordered multi-camera trajectory for a plate. |
| `GET` | `/trajectory-api/trajectories/map/{plate}` | Generate interactive Leaflet HTML route map for a vehicle. |
| `GET` | `/trajectory-api/vehicles/{plate}` | Aggregated vehicle profile (sighting count, body color, first/last seen). |

---

## 🧪 Running Tests

The test suite covers unit tests, ONNX inference, camera reconnects, early-exit OCR fusion, and API endpoints:

```bash
# Run all tests
pytest tests/

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run targeted unit tests
pytest tests/unit/test_ocr_fusion_early_exit.py
pytest tests/unit/test_onnx_yolo_runner.py
pytest tests/unit/test_campus_direction.py
pytest tests/unit/test_frame_capture_reconnect.py

# Run with test coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📊 Model Performance & Benchmarks

### YOLOv8 Plate Detector Evaluation (Test Set)

| Metric | Baseline YOLOv8m (`best.pt`) | Fine-Tuned Model (`best_finetuned.onnx`) |
|---|---|---|
| **mAP50** | 98.97% | **99.47%** |
| **mAP50-95** | 88.26% | **93.00%** |
| **Precision** | 98.97% | **99.29%** |
| **Recall** | 99.00% | **99.91%** |
| **Inference Runtime** | ~29.5 ms/image | **~4.7 ms/image (ONNX Runtime)** |

### OCR Engine Benchmarks

| OCR Engine | Hardware | Latency / Crop | Notes |
|---|---|---|---|
| **RapidOCR (PP-OCRv4 ONNX)** | CPU | **~40–70 ms** | Default engine; DBNet clamped, no PaddlePaddle needed |
| **PARSeq TensorRT** | GPU (RTX / Orin) | **~3–6 ms** | Ultra-low latency GPU transformer backend |
| **PaddleOCR** | CPU / GPU | ~180–300 ms | Baseline OCR engine |
| **TrOCR (Transformers)** | GPU | ~150–220 ms | High accuracy on degraded/unusual fonts |

---

## 🛠️ Tech Stack

| Domain | Technologies |
|---|---|
| **Detection & Tracking** | YOLOv8 (Ultralytics), ONNX Runtime, ByteTrack (Supervision) |
| **Optical Character Recognition** | RapidOCR (PP-OCRv4 ONNX), TensorRT PARSeq, PaddleOCR, TrOCR |
| **Super-Resolution & Preprocessing** | OpenCV, NumPy, CLAHE, Real-ESRGAN |
| **Backend & APIs** | FastAPI, Uvicorn, Pydantic, SQLAlchemy |
| **Databases** | SQLite (default), PostgreSQL |
| **Frontend & UI** | Next.js 15, React 19, TypeScript, Tailwind CSS, Lucide React, Leaflet |
| **Legacy Dashboards** | Streamlit |
| **Testing** | Pytest, Hypothesis, Pytest-Cov |

---

## 🔧 Troubleshooting

### `ModuleNotFoundError: No module named 'src'`
Run all scripts from the repository root directory:
```bash
# Correct
python scripts/run_pipeline.py

# Incorrect
cd scripts && python run_pipeline.py
```

### RapidOCR / ONNX Runtime Execution
RapidOCR executes via `onnxruntime`. If running on CPU, verify `config/config.yaml` has `ocr.rapidocr.det_limit_side_len: 320` and `use_cls: false` for maximum throughput.

### RTSP Stream Reconnection
If an RTSP feed disconnects or drops frames, `src/capture/frame_capture.py` automatically initiates background retries up to `video.max_retries` with exponential backoff and frame stall timeouts (`video.stall_timeout: 10.0`).

### Camera Attribution Showing `—`
If an existing database is used from an older version, previously recorded entries will display `—` for camera ID and GPS coordinates until new events with camera attribution are logged.

---

## 📖 Documentation

- [Multi-Camera Trajectory System Architecture](docs/MULTI_CAMERA_TRAJECTORY_SYSTEM.md)
- [PARSeq Deployment & TensorRT Export Guide](docs/PARSEQ_DEPLOYMENT.md)
- [ALPR Technology Guide](docs/ALPR_Technology_Guide.html)
- [Pipeline In-Depth Overview](docs/pipeline.md)

# 🚗 Road Lane Line Detection & ADAS Telemetry System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-20%2F20%20Passing-success.svg)](#)

A production-grade, portfolio-level Computer Vision and Advanced Driver Assistance System (ADAS) engineered with **Python**, **OpenCV**, and **Streamlit**. It processes real-time front-facing road video feeds to detect lane lines, track driveable corridors, evaluate lateral vehicle departure warnings (LDW), compute road curvature, optionally identify surrounding vehicles using **Ultralytics YOLO**, and stream analytical telemetry to an automotive dashboard.

---

## 📸 Application & Pipeline Showcase

### 1. Interactive Streamlit ADAS Cockpit & Real-Time Telemetry
![ADAS Streamlit Dashboard](output/dashboard_sample_road.jpg)

### 2. Autonomous Lane Tracking with Telemetry HUD Overlay
![Detected Highway Overlay](output/detected_sample_road.jpg)

---

## 📑 Table of Contents
1. [1. Project Introduction](#1-project-introduction)
2. [2. Problem Statement](#2-problem-statement)
3. [3. Project Objectives](#3-project-objectives)
4. [4. Proposed Solution](#4-proposed-solution)
5. [5. Key Features](#5-key-features)
6. [6. Technology Stack](#6-technology-stack)
7. [7. System Architecture](#7-system-architecture)
8. [8. Image Processing Methodology](#8-image-processing-methodology)
9. [9. Lane Detection Methodology](#9-lane-detection-methodology)
10. [10. ADAS Warning Logic & Lane Departure](#10-adas-warning-logic--lane-departure)
11. [11. Vehicle Detection Methodology (YOLO)](#11-vehicle-detection-methodology-yolo)
12. [12. Telemetry Methodology](#12-telemetry-methodology)
13. [13. Installation Instructions](#13-installation-instructions)
14. [14. Project Execution Instructions](#14-project-execution-instructions)
15. [15. Folder Structure](#15-folder-structure)
16. [16. Dataset References](#16-dataset-references)
17. [17. Testing Instructions (pytest)](#17-testing-instructions-pytest)
18. [18. Streamlit Community Cloud Deployment](#18-streamlit-community-cloud-deployment)
19. [19. Limitations](#19-limitations)
20. [20. Future Scope](#20-future-scope)

---

## 1. Project Introduction
Driver distraction and inadvertent lane drifting remain among the primary causes of highway collisions globally. Modern autonomous and semi-autonomous vehicles employ Advanced Driver Assistance Systems (ADAS) to perceive their surroundings, track road corridors, and issue timely alerts.

This project delivers a functional, end-to-end software prototype of an intelligent driving assistant that ingests forward dashcam video feeds, performs deterministic Computer Vision edge and line geometry transformations, evaluates vehicle centering, and streams interactive telemetry into a dark-themed automotive dashboard.

---

## 2. Problem Statement
Detecting highway lanes reliably in varying conditions presents several computer vision challenges:
- High-frequency asphalt noise, shadows, and texture irregularities.
- Occluded or broken lane dashes.
- Processing latency constraints for real-time video streaming.
- False positive lines originating from roadside barriers, sky, or vehicle hoods.
- Video frame flickering due to single-frame detection drops.

---

## 3. Project Objectives
1. **Deterministic Edge Pipeline:** Isolate road lane boundaries without hardcoded values using Gaussian filtering, Canny gradient computation, and Region of Interest (ROI) masking.
2. **Mathematical Line Fitting:** Convert collinear edge pixels into continuous left and right boundaries using Probabilistic Hough Transform and 1st-degree polynomial regression ($x = my + c$).
3. **Lateral Deviation & LDW:** Estimate vehicle position relative to the lane center and trigger categorized alerts (`SAFE`, `CAUTION`, `WARNING`, `LANE DEPARTURE`).
4. **Road Curvature Estimation:** Analyze the convergence of lane boundaries to detect straight paths versus left/right bends.
5. **Surrounding Vehicle Identification:** Integrate lightweight Ultralytics YOLO to track surrounding traffic without degrading lane processing throughput.
6. **Telemetry & Dashboard Analytics:** Store frame-by-frame metrics in Pandas, generate interactive Plotly charts, and export auditable CSV/Markdown reports.

---

## 4. Proposed Solution
A modular, clean architecture separating:
- **Low-level image transformations** (`modules/image_processing.py`)
- **Lane geometry and extrapolation** (`modules/lane_detection.py`)
- **Safety warning state machines** (`modules/lane_departure.py`)
- **Road curvature analysis** (`modules/road_curvature.py`)
- **Deep learning object detection** (`modules/vehicle_detection.py`)
- **Data logging and statistics** (`modules/telemetry.py`)
- **Streaming video orchestration** (`modules/video_processor.py`)
- **Automotive cockpit UI** (`app.py`)

---

## 5. Key Features

- **Automotive-Grade Dark Theme:** Built with Streamlit, Plotly, and custom CSS styling for high-contrast visibility.
- **Dynamic Region of Interest:** Automatically scales trapezoidal masking proportionally across any video resolution.
- **Slope-Partitioned Regression:** Robust separation of left ($\text{slope} < 0$) and right ($\text{slope} > 0$) lines, discarding horizontal noise.
- **Exponential Temporal Smoothing (EMA):** Prevents frame jitter and maintains stable lane overlays across momentary dropouts.
- **Lane Departure Hysteresis:** Cooldown logic prevents duplicate counting of departure incidents across continuous video frames.
- **Real-Time Telemetry HUD:** In-frame head-up display showing active warning level, deviation in meters/pixels, curvature, and FPS throughput.
- **Interactive Analytics:** Plotly visualizations of lateral deviation trajectories, warning breakdowns, vehicle counts, and system FPS.
- **Downloadable Deliverables:** Export processed videos (MP4), full telemetry logs (CSV), and comprehensive markdown audit reports.

---

## 6. Technology Stack

| Domain | Technology / Library | Role |
|---|---|---|
| **Language** | Python 3.9 - 3.14 | Core execution engine |
| **Computer Vision** | OpenCV (`opencv-python`) | Image filtering, Canny gradients, Hough transforms, Video I/O |
| **Data Processing** | NumPy & Pandas | Numerical matrix manipulation and telemetry DataFrame logging |
| **Deep Learning** | Ultralytics YOLO (YOLO11 / YOLOv8) | Nearby vehicle classification and bounding box localization |
| **Frontend UI** | Streamlit | Responsive automotive web application |
| **Data Visualization** | Plotly Express & Graph Objects | Dynamic time-series and distribution charts |
| **Quality & Testing** | Pytest | 20-point automated test suite |

---

## 7. System Architecture

```text
[Road Dashcam Video / Upload]
               │
               ▼
┌───────────────────────────────┐
│     VideoProcessor Engine     │ ◄── Sequential frame stream
└──────────────┬────────────────┘
               │
       ┌───────┴───────────────────────────────┐
       ▼                                       ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│       LaneDetector           │       │    VehicleDetector (YOLO)    │
│  - Grayscale & Gaussian Blur │       │  - Optional Object Inference │
│  - Canny Edge Detection      │       │  - Cars, Buses, Trucks       │
│  - Trapezoidal ROI Mask      │       │  - Bounding Box Annotations  │
│  - HoughLinesP Transform     │       └──────────────┬───────────────┘
│  - Polyfit & EMA Smoothing   │                      │
└──────────────┬───────────────┘                      │
               │                                      │
       ┌───────┴───────────────────────┐              │
       ▼                               ▼              │
┌──────────────────────────────┐ ┌───────────────────────────┐│
│   LaneDepartureWarningSystem │ │  RoadCurvatureEstimator   ││
│  - Lateral Deviation (px/m)  │ │  - Midpoint Trajectory    ││
│  - Warning Levels & Hysteresis│ │  - Straight / Bend State ││
└──────────────┬───────────────┘ └─────────────┬─────────────┘│
               │                               │              │
               └───────────────┬───────────────┘              │
                               ▼                              │
┌─────────────────────────────────────────────────────────────┴─┐
│                    TelemetryCollector                         │
│  - Frame metrics logging (Pandas DataFrame)                   │
│  - Real-time HUD banner rendering                             │
│  - Statistical aggregation & CSV export                       │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│            Streamlit Interactive Dashboard (app.py)           │
│  - Video player, ADAS status badges, Plotly graphs, Reports   │
└───────────────────────────────────────────────────────────────┘
```

---

## 8. Image Processing Methodology

1. **Grayscale Conversion:** Reduces three-channel BGR images to one-channel luminance, cutting computation overhead by 66% while preserving gradient edges.
2. **Gaussian Smoothing:** Convolves pixels with a 2D Gaussian kernel ($5 \times 5$):
   $$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
3. **Canny Edge Detection:** Determines spatial intensity gradients using Sobel operators:
   $$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$
   Applies non-maximum suppression and hysteresis thresholding ($T_{\text{low}} = 50$, $T_{\text{high}} = 150$).
4. **Trapezoidal ROI Masking:** Eliminates non-road artifacts by masking all pixels outside the driver's forward horizon.

---

## 9. Lane Detection Methodology

1. **Probabilistic Hough Line Transform (`cv2.HoughLinesP`):** Converts edge pixels into discrete segments $[x_1, y_1, x_2, y_2]$ in polar space ($\rho = x\cos\theta + y\sin\theta$).
2. **Slope Filtering:** Segments are partitioned:
   - **Left Lane:** $-2.5 \le \text{slope} \le -0.4$ within the left spatial quadrant.
   - **Right Lane:** $+0.4 \le \text{slope} \le +2.5$ within the right spatial quadrant.
3. **Polynomial Regression:** Fits $x = f(y) = my + c$ to prevent mathematical singularity on vertical lines.
4. **Exponential Moving Average (EMA):**
   $$L_t = \alpha \cdot L_{\text{curr}} + (1 - \alpha) \cdot L_{t-1}$$
   Provides flicker-free lane tracking across continuous video frames.

---

## 10. ADAS Warning Logic & Lane Departure

Assuming the forward camera is centered along the vehicle's longitudinal axis ($x_{\text{vehicle}} = \text{width} / 2$):

$$\text{Lane Center} = \frac{x_{\text{left\_bottom}} + x_{\text{right\_bottom}}}{2}$$
$$\Delta_{\text{lateral}} = x_{\text{vehicle}} - \text{Lane Center}$$

### Metric Conversion
Using a standard highway lane width of $3.7\text{ meters}$:
$$\text{Deviation (meters)} = \Delta_{\text{lateral}} \times \left(\frac{3.7}{x_{\text{right\_bottom}} - x_{\text{left\_bottom}}}\right)$$

### Warning States & Hysteresis
- **`SAFE`** ($|\Delta_{\text{lateral}}| < 35\text{px}$): Vehicle safely centered within driveable corridor.
- **`CAUTION`** ($35\text{px} \le |\Delta_{\text{lateral}}| < 75\text{px}$): Minor lateral drift.
- **`WARNING`** ($75\text{px} \le |\Delta_{\text{lateral}}| < 125\text{px}$): Significant departure trajectory.
- **`LANE DEPARTURE`** ($|\Delta_{\text{lateral}}| \ge 125\text{px}$): Vehicle boundary breaches lane limits.
- **Cooldown Logic:** Departure incidents are registered with a 30-frame refractory cooldown to prevent duplicate counting during a single drift event.

---

## 11. Vehicle Detection Methodology (YOLO)

- Uses lightweight **Ultralytics YOLO** (`yolo11n.pt` / `yolov8n.pt`) with automatic Apple Silicon (MPS) and CPU acceleration.
- Filters for COCO classes: `car` (2), `motorcycle` (3), `bus` (5), `truck` (7).
- Draws color-coded bounding boxes and confidence badges.
- **Fault-Tolerant:** Lazily imported; operates gracefully if weights or network are unavailable without interrupting lane tracking.

---

## 12. Telemetry Methodology

Maintains a comprehensive Pandas DataFrame containing:
- `frame_idx`: Sequential frame index
- `timestamp_sec`: Elapsed video timestamp
- `deviation_px`: Pixel displacement from lane center
- `deviation_m`: Estimated metric offset
- `position_label`: `CENTERED`, `LEFT_OFFSET`, `RIGHT_OFFSET`
- `detection_status`: `BOTH_LANES_DETECTED`, `LEFT_LANE_ONLY`, etc.
- `curvature_direction`: `STRAIGHT`, `CURVING_LEFT`, `CURVING_RIGHT`
- `warning_level`: Active ADAS safety state
- `vehicle_count`: Detected nearby vehicles
- `processing_fps`: Real-time pipeline throughput

---

## 13. Installation Instructions

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/kdev76979-mac/road-lane-detection.git
cd road-lane-detection

# 2. Create virtual environment
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 14. Project Execution Instructions

### A. Run Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### B. Run CLI Quick Demo
```bash
python main.py
```

### C. Process a Custom Video or Live Webcam via CLI
```bash
# Custom video
python main.py --video path/to/your_video.mp4

# Live webcam (press 'q' to quit)
python main.py --webcam 0
```

---

## 15. Folder Structure

```text
road-lane-detection/
├── app.py                      # Main Streamlit web dashboard
├── main.py                     # CLI execution runner
├── requirements.txt            # Package dependencies
├── README.md                   # Complete documentation & study guide
├── LICENSE                     # MIT License
├── .gitignore                  # Git exclusion rules
│
├── modules/                    # Modular ADAS perception package
│   ├── __init__.py
│   ├── image_processing.py     # Grayscale, Canny, Gaussian, ROI, Hough transforms
│   ├── lane_detection.py       # Polyfit regression, slope filtering, temporal EMA
│   ├── lane_departure.py       # Lateral deviation, warning states, cooldown logic
│   ├── road_curvature.py       # Geometric curve & direction estimator
│   ├── vehicle_detection.py    # Lightweight YOLO vehicle tracking wrapper
│   ├── telemetry.py            # Pandas telemetry logger & statistical aggregator
│   ├── video_processor.py      # Frame streaming engine, HUD renderer, video export
│   └── report_generator.py     # Markdown audit report compiler
│
├── dataset/
│   └── road_videos/            # Sample highway clips
│       └── sample_highway.mp4
│
├── output/
│   ├── processed_videos/       # Exported annotated videos with HUD
│   ├── reports/                # Exported markdown and CSV safety audits
│   ├── dashboard_sample_road.jpg
│   └── detected_sample_road.jpg
│
└── tests/
    ├── __init__.py
    ├── test_adas_system.py     # 20-point comprehensive pytest test suite
    └── test_lane_detector.py   # Unit test verification
```

---

## 16. Dataset References
- Includes procedural highway generator (`generate_sample_road_image` & `generate_sample_road_video`) for zero-download local testing.
- Compatible with dashcam datasets including:
  - **TuSimple Lane Detection Benchmark**
  - **CULane Dataset**
  - **KITTI Vision Benchmark Suite**
  - Standard `.mp4` dashcam footage (720p / 1080p).

---

## 17. Testing Instructions (pytest)

Run the automated test suite verifying all 20 modular pipeline stages:

```bash
pytest tests/test_adas_system.py -v
```

Expected output:
```text
============================= test session starts ==============================
collected 20 items

tests/test_adas_system.py::test_to_grayscale PASSED                      [  5%]
tests/test_adas_system.py::test_gaussian_blur PASSED                     [ 10%]
tests/test_adas_system.py::test_canny_edges PASSED                       [ 15%]
tests/test_adas_system.py::test_roi_mask PASSED                          [ 20%]
tests/test_adas_system.py::test_hough_lines PASSED                       [ 25%]
tests/test_adas_system.py::test_lane_detector_on_sample PASSED           [ 30%]
tests/test_adas_system.py::test_lane_detector_missing_lanes PASSED       [ 35%]
tests/test_adas_system.py::test_lane_temporal_smoothing PASSED           [ 40%]
tests/test_adas_system.py::test_ldw_centered PASSED                      [ 45%]
tests/test_adas_system.py::test_ldw_right_departure PASSED               [ 50%]
tests/test_adas_system.py::test_ldw_left_departure PASSED                [ 55%]
tests/test_adas_system.py::test_ldw_cooldown_logic PASSED                [ 60%]
tests/test_adas_system.py::test_road_curvature_straight PASSED           [ 65%]
tests/test_adas_system.py::test_road_curvature_right PASSED              [ 70%]
tests/test_adas_system.py::test_road_curvature_left PASSED               [ 75%]
tests/test_adas_system.py::test_vehicle_detector_initialization PASSED    [ 80%]
tests/test_adas_system.py::test_telemetry_collector_dataframe_and_csv PASSED [ 85%]
tests/test_adas_system.py::test_markdown_report_generation PASSED        [ 90%]
tests/test_adas_system.py::test_video_metadata_and_processing PASSED     [ 95%]
tests/test_adas_system.py::test_invalid_video_path PASSED                [100%]

============================= 20 passed in 22.24s ==============================
```

---

## 18. Streamlit Community Cloud Deployment

1. **Push to GitHub:** Push your repository to GitHub (`main` branch).
2. **Link to Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io).
3. **Repository Settings:**
   - **Repository:** `kdev76979-mac/road-lane-detection`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Deploy:** Click **Deploy**. Streamlit Cloud will read `requirements.txt` and launch your dashboard with a shareable public URL.

---

## 19. Limitations
- **Geometric Approximation:** Road curvature is estimated from perspective line convergence rather than a fully calibrated intrinsic camera matrix.
- **Lighting Extremes:** Classical edge detection performance degrades under intense glare, snow, or unlit nighttime roads without deep learning priors.
- **Prototype Scope:** Designed as an educational, portfolio-grade ADAS research prototype; not certified for life-critical automotive actuation or automatic steering.

---

## 20. Future Scope
- [ ] **Inverse Perspective Mapping (IPM):** Bird's-eye view warp with 2nd-degree parabolic spline fitting ($x = ay^2 + by + c$).
- [ ] **Hybrid Deep Learning:** Integration with LaneNet or Ultra-Fast-Lane-Detection for adverse weather and night conditions.
- [ ] **Forward Collision Warning (FCW):** Bounding-box scale rate estimation ($\tau = s / \dot{s}$) for time-to-collision (TTC) calculations.
- [ ] **Embedded Hardware Optimization:** Compilation to TensorRT / ONNX Runtime for deployment on NVIDIA Jetson or Raspberry Pi 5.

---

## 👤 Author & License
- **Developer:** Dev Kushwah ([@kdev76979-mac](https://github.com/kdev76979-mac))
- **License:** Distributed under the [MIT License](LICENSE).

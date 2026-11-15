"""
Pytest Test Suite for Road Lane Detection & ADAS Telemetry System.
Covers modular components: Image Processing, Lane Detection, Lane Departure,
Road Curvature, Vehicle Detection, Telemetry, and Video Processing.
"""

import os
import pytest
import numpy as np
import pandas as pd
import cv2

from modules.image_processing import (
    to_grayscale,
    apply_gaussian_blur,
    detect_canny_edges,
    apply_roi_mask,
    detect_hough_lines,
    get_default_roi_vertices
)
from modules.lane_detection import LaneDetector
from modules.lane_departure import LaneDepartureWarningSystem
from modules.road_curvature import RoadCurvatureEstimator
from modules.vehicle_detection import VehicleDetector
from modules.telemetry import TelemetryCollector
from modules.report_generator import generate_markdown_report
from modules.video_processor import VideoProcessor
from src.utils import generate_sample_road_image


@pytest.fixture
def sample_road_frame():
    """Generates synthetic 1280x720 highway frame for consistent unit testing."""
    return generate_sample_road_image(width=1280, height=720)


@pytest.fixture
def blank_frame():
    """Generates 1280x720 black frame to test missing lanes and edge cases."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ------------------------------------------------------------------------------
# 1. Image Processing Tests
# ------------------------------------------------------------------------------
def test_to_grayscale(sample_road_frame):
    gray = to_grayscale(sample_road_frame)
    assert len(gray.shape) == 2
    assert gray.shape == (720, 1280)
    assert gray.dtype == np.uint8


def test_gaussian_blur(sample_road_frame):
    gray = to_grayscale(sample_road_frame)
    blurred = apply_gaussian_blur(gray, kernel_size=5)
    assert blurred.shape == gray.shape
    assert blurred.dtype == np.uint8


def test_canny_edges(sample_road_frame):
    gray = to_grayscale(sample_road_frame)
    blurred = apply_gaussian_blur(gray, kernel_size=5)
    edges = detect_canny_edges(blurred, 50, 150)
    assert edges.shape == gray.shape
    unique = set(np.unique(edges))
    assert unique.issubset({0, 255})


def test_roi_mask(sample_road_frame):
    gray = to_grayscale(sample_road_frame)
    edges = detect_canny_edges(gray, 50, 150)
    masked = apply_roi_mask(edges)
    assert masked.shape == edges.shape
    # Top 40% should be completely black (masked out)
    top_portion = masked[:int(720 * 0.40), :]
    assert np.count_nonzero(top_portion) == 0


def test_hough_lines(sample_road_frame):
    gray = to_grayscale(sample_road_frame)
    edges = detect_canny_edges(gray, 50, 150)
    masked = apply_roi_mask(edges)
    lines = detect_hough_lines(masked)
    assert lines is not None
    assert lines.shape[1] == 4


# ------------------------------------------------------------------------------
# 2. Lane Detection Tests
# ------------------------------------------------------------------------------
def test_lane_detector_on_sample(sample_road_frame):
    detector = LaneDetector()
    res = detector.process_frame(sample_road_frame)
    assert res["status"] in {"BOTH_LANES_DETECTED", "LEFT_LANE_ONLY", "RIGHT_LANE_ONLY"}
    assert res["left_line"] is not None
    assert res["right_line"] is not None
    assert res["lane_center"] is not None
    assert res["annotated_frame"].shape == sample_road_frame.shape


def test_lane_detector_missing_lanes(blank_frame):
    detector = LaneDetector()
    res = detector.process_frame(blank_frame)
    assert res["status"] == "NO_LANES_DETECTED"
    assert res["left_line"] is None
    assert res["right_line"] is None
    assert res["lane_center"] is None


def test_lane_temporal_smoothing():
    detector = LaneDetector(smoothing_factor=0.80)
    line1 = (200, 720, 500, 450)
    line2 = (250, 720, 520, 450)
    smoothed = detector._smooth_line(line1, line2)
    # EMA: 0.8 * 200 + 0.2 * 250 = 160 + 50 = 210
    assert smoothed[0] == 210


# ------------------------------------------------------------------------------
# 3. ADAS Lane Departure Warning Tests
# ------------------------------------------------------------------------------
def test_ldw_centered():
    ldw = LaneDepartureWarningSystem(departure_threshold_px=100.0)
    # Car at 640, lane center at 640
    res = ldw.evaluate(lane_center=640.0, frame_width=1280)
    assert res["warning_level"] == LaneDepartureWarningSystem.STATUS_SAFE
    assert res["position_label"] == "CENTERED"
    assert res["deviation_px"] == 0.0


def test_ldw_right_departure():
    ldw = LaneDepartureWarningSystem(departure_threshold_px=100.0)
    # Car at 640, lane center at 450 -> deviation = +190px (car drifted right)
    res = ldw.evaluate(lane_center=450.0, frame_width=1280, frame_idx=1)
    assert res["warning_level"] == LaneDepartureWarningSystem.STATUS_DEPARTURE
    assert res["position_label"] == "RIGHT_OFFSET"
    assert res["deviation_px"] == 190.0
    assert res["total_departure_events"] == 1


def test_ldw_left_departure():
    ldw = LaneDepartureWarningSystem(departure_threshold_px=100.0)
    # Car at 640, lane center at 800 -> deviation = -160px (car drifted left)
    res = ldw.evaluate(lane_center=800.0, frame_width=1280, frame_idx=1)
    assert res["warning_level"] == LaneDepartureWarningSystem.STATUS_DEPARTURE
    assert res["position_label"] == "LEFT_OFFSET"
    assert res["deviation_px"] == -160.0
    assert res["total_departure_events"] == 1


def test_ldw_cooldown_logic():
    ldw = LaneDepartureWarningSystem(departure_threshold_px=100.0, event_cooldown_frames=30)
    # Trigger first event
    res1 = ldw.evaluate(lane_center=400.0, frame_width=1280, frame_idx=1)
    assert res1["total_departure_events"] == 1

    # Immediate next frame still departing
    res2 = ldw.evaluate(lane_center=400.0, frame_width=1280, frame_idx=2)
    # Event count must NOT increment due to cooldown
    assert res2["total_departure_events"] == 1


# ------------------------------------------------------------------------------
# 4. Road Curvature Tests
# ------------------------------------------------------------------------------
def test_road_curvature_straight():
    estimator = RoadCurvatureEstimator(straight_threshold_px=20.0)
    left_line = (200, 720, 540, 450)
    right_line = (1080, 720, 740, 450)
    # Top mid = (540 + 740) / 2 = 640 (exact center of 1280)
    res = estimator.estimate_curvature(left_line, right_line, frame_width=1280)
    assert res["direction"] == RoadCurvatureEstimator.CURVE_STRAIGHT


def test_road_curvature_right():
    estimator = RoadCurvatureEstimator(straight_threshold_px=20.0)
    left_line = (300, 720, 700, 450)
    right_line = (1180, 720, 900, 450)
    # Top mid = 800 (offset +160px right)
    res = estimator.estimate_curvature(left_line, right_line, frame_width=1280)
    assert res["direction"] == RoadCurvatureEstimator.CURVE_RIGHT


def test_road_curvature_left():
    estimator = RoadCurvatureEstimator(straight_threshold_px=20.0)
    left_line = (100, 720, 400, 450)
    right_line = (980, 720, 600, 450)
    # Top mid = 500 (offset -140px left)
    res = estimator.estimate_curvature(left_line, right_line, frame_width=1280)
    assert res["direction"] == RoadCurvatureEstimator.CURVE_LEFT


# ------------------------------------------------------------------------------
# 5. Vehicle Detection (YOLO / Fallback) Tests
# ------------------------------------------------------------------------------
def test_vehicle_detector_initialization():
    detector = VehicleDetector(model_name="yolo11n.pt")
    assert detector.model_name == "yolo11n.pt"
    # Detect on dummy frame without crashing
    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    annotated, detections, count = detector.detect_vehicles(dummy)
    assert count == len(detections)
    assert annotated.shape == dummy.shape


# ------------------------------------------------------------------------------
# 6. Telemetry & Reporting Tests
# ------------------------------------------------------------------------------
def test_telemetry_collector_dataframe_and_csv(tmp_path):
    collector = TelemetryCollector()
    collector.record_frame(
        frame_idx=1,
        timestamp_sec=0.04,
        deviation_px=10.5,
        deviation_m=0.06,
        position_label="CENTERED",
        detection_status="BOTH_LANES_DETECTED",
        curvature_direction="STRAIGHT",
        curvature_offset_px=2.1,
        warning_level="SAFE",
        total_warning_events=0,
        total_departure_events=0,
        vehicle_count=2,
        processing_fps=28.5
    )

    df = collector.to_dataframe()
    assert len(df) == 1
    assert df["frame_idx"].iloc[0] == 1
    assert df["deviation_px"].iloc[0] == 10.5

    stats = collector.get_summary_statistics()
    assert stats["total_frames_processed"] == 1
    assert stats["lane_tracking_rate_pct"] == 100.0

    csv_out = os.path.join(tmp_path, "telemetry.csv")
    collector.export_csv(csv_out)
    assert os.path.exists(csv_out)
    loaded_df = pd.read_csv(csv_out)
    assert len(loaded_df) == 1


def test_markdown_report_generation(tmp_path):
    summary = {
        "total_frames_processed": 100,
        "session_duration_sec": 4.0,
        "average_fps": 25.0,
        "total_departure_events": 0,
        "total_warning_events": 0,
        "lane_tracking_rate_pct": 95.0,
        "mean_deviation_px": 8.5,
        "mean_deviation_m": 0.05,
        "max_deviation_px": 25.0,
        "max_deviation_m": 0.15,
        "warning_level_counts": {"SAFE": 95, "CAUTION": 5},
        "max_vehicles_detected": 1
    }
    meta = {
        "filename": "test.mp4",
        "width": 1280,
        "height": 720,
        "fps": 25.0,
        "total_frames": 100
    }
    report_file = os.path.join(tmp_path, "report.md")
    report = generate_markdown_report(summary, meta, output_path=report_file)
    assert "ADAS Telemetry & Lane Detection Safety Report" in report
    assert "EXCELLENT (A+)" in report
    assert os.path.exists(report_file)


# ------------------------------------------------------------------------------
# 7. Video Processor Integration Tests
# ------------------------------------------------------------------------------
def test_video_metadata_and_processing(tmp_path):
    video_path = "dataset/road_videos/sample_highway.mp4"
    if not os.path.exists(video_path):
        pytest.skip("Sample video not found.")

    processor = VideoProcessor()
    meta = processor.get_video_metadata(video_path)
    assert meta["width"] == 1280
    assert meta["height"] == 720
    assert meta["fps"] > 0
    assert meta["total_frames"] > 0

    # Process first 10 frames
    out_video = os.path.join(tmp_path, "test_out.mp4")
    res = processor.process_video(
        input_video_path=video_path,
        output_video_path=out_video,
        max_frames=10
    )
    assert res["processed_frames"] == 10
    assert os.path.exists(out_video)
    assert os.path.getsize(out_video) > 0


def test_invalid_video_path():
    processor = VideoProcessor()
    with pytest.raises(FileNotFoundError):
        processor.get_video_metadata("non_existent_file.mp4")


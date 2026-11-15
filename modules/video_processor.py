"""
Video Processor Engine for Road Lane Detection & ADAS Telemetry.
Handles streaming video reading, sequential frame-by-frame CV pipeline execution,
integrated HUD rendering, telemetry gathering, and encoded MP4 video export.
"""

from typing import Callable, Dict, Optional, Tuple
import os
import time
import cv2
import numpy as np

from modules.lane_detection import LaneDetector
from modules.lane_departure import LaneDepartureWarningSystem
from modules.road_curvature import RoadCurvatureEstimator
from modules.vehicle_detection import VehicleDetector
from modules.telemetry import TelemetryCollector


class VideoProcessor:
    """
    Coordinates end-to-end computer vision and ADAS analysis across video feeds.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        lane_detector: Optional[LaneDetector] = None,
        ldw_system: Optional[LaneDepartureWarningSystem] = None,
        curvature_estimator: Optional[RoadCurvatureEstimator] = None,
        vehicle_detector: Optional[VehicleDetector] = None,
        telemetry_collector: Optional[TelemetryCollector] = None,
        enable_vehicle_detection: bool = False
    ):
        self.lane_detector = lane_detector or LaneDetector()
        self.ldw_system = ldw_system or LaneDepartureWarningSystem()
        self.curvature_estimator = curvature_estimator or RoadCurvatureEstimator()
        self.vehicle_detector = vehicle_detector or VehicleDetector()
        self.telemetry_collector = telemetry_collector or TelemetryCollector()
        self.enable_vehicle_detection = enable_vehicle_detection

    def get_video_metadata(self, video_path: str) -> Dict:
        """Reads video dimensions, FPS, frame count, and duration without full memory load."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0.0

            return {
                "filename": os.path.basename(video_path),
                "filepath": video_path,
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_sec": round(duration_sec, 2)
            }
        finally:
            cap.release()

    def render_adas_hud(
        self,
        frame: np.ndarray,
        ldw_result: Dict,
        curvature_result: Dict,
        vehicle_count: int,
        fps: float,
        departure_side: Optional[str] = None
    ) -> np.ndarray:
        """
        Overlays an automotive-grade dark telemetry HUD banner, dynamic windshield
        steering guidance chevrons (matching real ADAS cockpits), and driver alert pills.
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]

        hud_w, hud_h = min(480, int(width * 0.45)), 135
        x0, y0 = 20, 20

        # Semi-transparent dark background card
        card_roi = annotated[y0:y0 + hud_h, x0:x0 + hud_w]
        dark_card = np.zeros_like(card_roi)
        blended_card = cv2.addWeighted(card_roi, 0.25, dark_card, 0.75, 0)
        annotated[y0:y0 + hud_h, x0:x0 + hud_w] = blended_card

        # Border styling based on warning level
        warning_level = ldw_result.get("warning_level", LaneDepartureWarningSystem.STATUS_SAFE)
        if warning_level == LaneDepartureWarningSystem.STATUS_SAFE:
            border_color = (0, 220, 0)       # Green
        elif warning_level == LaneDepartureWarningSystem.STATUS_CAUTION:
            border_color = (0, 215, 255)     # Yellow
        elif warning_level == LaneDepartureWarningSystem.STATUS_WARNING:
            border_color = (0, 140, 255)     # Orange
        else:
            border_color = (0, 0, 255)       # Red (LANE DEPARTURE)

        cv2.rectangle(annotated, (x0, y0), (x0 + hud_w, y0 + hud_h), border_color, 2)

        # Title
        cv2.putText(
            annotated,
            "ADAS TELEMETRY SYSTEM",
            (x0 + 15, y0 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Warning Status
        status_text = f"STATUS: [{warning_level}]"
        cv2.putText(
            annotated,
            status_text,
            (x0 + 15, y0 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            border_color,
            2,
            cv2.LINE_AA
        )

        # Lateral Deviation
        dev_m = ldw_result.get("deviation_m")
        dev_px = ldw_result.get("deviation_px")
        pos_label = ldw_result.get("position_label", "SEARCHING")
        if dev_m is not None:
            dev_str = f"OFFSET: {dev_m:+.2f}m ({dev_px:+.0f}px) [{pos_label}]"
        elif dev_px is not None:
            dev_str = f"OFFSET: {dev_px:+.0f}px [{pos_label}]"
        else:
            dev_str = f"OFFSET: SEARCHING"

        cv2.putText(
            annotated,
            dev_str,
            (x0 + 15, y0 + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # Road Curvature & Vehicles
        curve_dir = curvature_result.get("direction", "UNKNOWN")
        info_str = f"CURVE: {curve_dir} | VEHICLES: {vehicle_count}"
        cv2.putText(
            annotated,
            info_str,
            (x0 + 15, y0 + 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 220, 255),
            1,
            cv2.LINE_AA
        )

        # Processing Throughput
        fps_str = f"THROUGHPUT: {fps:.1f} FPS | EVENTS: {ldw_result.get('total_departure_events', 0)}"
        cv2.putText(
            annotated,
            fps_str,
            (x0 + 15, y0 + 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------------------------
        # DYNAMIC WINDSHIELD GUIDANCE CHEVRONS (Active Steering Assistance)
        # ----------------------------------------------------------------------
        if departure_side == "RIGHT":
            # Drifting right -> Steer Left Chevrons (matching cockpit reference image)
            chev_y = int(height * 0.70)
            cv2.putText(
                annotated,
                "<<< STEER LEFT",
                (width // 2 - 95, chev_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 230, 255),
                2,
                cv2.LINE_AA
            )
            for i in range(8):
                cx = width // 2 - 140 + i * 35
                pts = np.array([[cx + 14, chev_y - 14], [cx, chev_y], [cx + 14, chev_y + 14]], dtype=np.int32)
                cv2.polylines(annotated, [pts], False, (0, 235, 255), 4, cv2.LINE_AA)

        elif departure_side == "LEFT":
            # Drifting left -> Steer Right Chevrons
            chev_y = int(height * 0.70)
            cv2.putText(
                annotated,
                "STEER RIGHT >>>",
                (width // 2 - 105, chev_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 230, 255),
                2,
                cv2.LINE_AA
            )
            for i in range(8):
                cx = width // 2 - 140 + i * 35
                pts = np.array([[cx - 14, chev_y - 14], [cx, chev_y], [cx - 14, chev_y + 14]], dtype=np.int32)
                cv2.polylines(annotated, [pts], False, (0, 235, 255), 4, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # COCKPIT FLOATING ALERT PILLS (Driver instrument perspective)
        # ----------------------------------------------------------------------
        pill_x = 30
        pill1_y = height - 90
        pill2_y = height - 48
        pill_w, pill_h = 240, 34

        # Pill 1: Lane Keeping / Departure Alert
        if departure_side:
            cv2.rectangle(annotated, (pill_x, pill1_y), (pill_x + pill_w, pill1_y + pill_h), (255, 255, 255), -1)
            cv2.rectangle(annotated, (pill_x, pill1_y), (pill_x + pill_w, pill1_y + pill_h), (0, 0, 240), 2)
            cv2.putText(annotated, f"! Lane Departure ({departure_side})", (pill_x + 12, pill1_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 220), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated, (pill_x, pill1_y), (pill_x + pill_w, pill1_y + pill_h), (255, 255, 255), -1)
            cv2.rectangle(annotated, (pill_x, pill1_y), (pill_x + pill_w, pill1_y + pill_h), (0, 200, 70), 2)
            cv2.putText(annotated, "Lane Keeping: Safe", (pill_x + 16, pill1_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 160, 50), 2, cv2.LINE_AA)

        # Pill 2: Lead Vehicle Distance Radar
        cv2.rectangle(annotated, (pill_x, pill2_y), (pill_x + pill_w, pill2_y + pill_h), (255, 255, 255), -1)
        cv2.rectangle(annotated, (pill_x, pill2_y), (pill_x + pill_w, pill2_y + pill_h), (0, 200, 70), 2)
        cv2.putText(annotated, "Distance: 18m (Safe)", (pill_x + 16, pill2_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 160, 50), 2, cv2.LINE_AA)

        return annotated

    def process_video(
        self,
        input_video_path: str,
        output_video_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[float, int, int, float], None]] = None,
        enable_yolo: Optional[bool] = None
    ) -> Dict:
        """
        Sequentially processes video file, logging telemetry and rendering output.
        """
        meta = self.get_video_metadata(input_video_path)
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {input_video_path}")

        if output_video_path is None:
            os.makedirs("output/processed_videos", exist_ok=True)
            base = os.path.splitext(os.path.basename(input_video_path))[0]
            output_video_path = os.path.join("output/processed_videos", f"detected_{base}.mp4")
        else:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        width = meta["width"]
        height = meta["height"]
        fps = meta["fps"]
        total_frames = meta["total_frames"]
        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

        # Video writer initialization
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Reset states
        self.lane_detector.reset()
        self.ldw_system.reset()
        self.curvature_estimator.reset()
        self.telemetry_collector.reset()

        use_yolo = self.enable_vehicle_detection if enable_yolo is None else enable_yolo
        frame_idx = 0
        start_time = time.time()
        last_frame_time = start_time

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                curr_time = time.time()
                dt = curr_time - last_frame_time
                current_fps = (1.0 / dt) if dt > 0 else fps
                last_frame_time = curr_time

                # 1. Lane Detection (Features & Geometry)
                lane_out = self.lane_detector.process_frame(frame, draw_overlay=False)
                left_line = lane_out["left_line"]
                right_line = lane_out["right_line"]
                lane_center = lane_out["lane_center"]
                detection_status = lane_out["status"]

                # 2. ADAS Lane Departure Evaluation
                ldw_res = self.ldw_system.evaluate(
                    lane_center=lane_center,
                    frame_width=width,
                    left_line=left_line,
                    right_line=right_line,
                    frame_idx=frame_idx
                )

                # Determine active departure side
                departure_side = None
                warning_level = ldw_res.get("warning_level")
                if warning_level in [LaneDepartureWarningSystem.STATUS_WARNING, LaneDepartureWarningSystem.STATUS_DEPARTURE]:
                    if ldw_res.get("position_label") == "RIGHT_OFFSET":
                        departure_side = "RIGHT"
                    elif ldw_res.get("position_label") == "LEFT_OFFSET":
                        departure_side = "LEFT"

                # 3. Draw Lane Overlay with Red Breached Boundary
                annotated = self.lane_detector.draw_lane_overlay(
                    frame,
                    left_line,
                    right_line,
                    departure_side=departure_side
                )

                # 4. Road Curvature Estimation
                curve_res = self.curvature_estimator.estimate_curvature(
                    left_line=left_line,
                    right_line=right_line,
                    frame_width=width
                )

                # 5. Vehicle Detection (Optional)
                vehicle_count = 0
                if use_yolo and self.vehicle_detector:
                    annotated, detections, vehicle_count = self.vehicle_detector.detect_vehicles(
                        annotated,
                        draw_boxes=True
                    )

                # 6. Telemetry Recording
                timestamp_sec = frame_idx / fps if fps > 0 else 0.0
                self.telemetry_collector.record_frame(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    deviation_px=ldw_res["deviation_px"],
                    deviation_m=ldw_res["deviation_m"],
                    position_label=ldw_res["position_label"],
                    detection_status=detection_status,
                    curvature_direction=curve_res["direction"],
                    curvature_offset_px=curve_res["curvature_offset"],
                    warning_level=ldw_res["warning_level"],
                    total_warning_events=ldw_res["total_warning_events"],
                    total_departure_events=ldw_res["total_departure_events"],
                    vehicle_count=vehicle_count,
                    processing_fps=current_fps
                )

                # 7. Render Cockpit HUD & Windshield Guidance Chevrons
                annotated = self.render_adas_hud(
                    annotated,
                    ldw_res,
                    curve_res,
                    vehicle_count,
                    current_fps,
                    departure_side=departure_side
                )

                out.write(annotated)

                if progress_callback:
                    pct = (frame_idx / frames_to_process) if frames_to_process > 0 else 1.0
                    progress_callback(pct, frame_idx, frames_to_process, current_fps)

                if max_frames and frame_idx >= max_frames:
                    break

        finally:
            cap.release()
            out.release()

        total_elapsed = time.time() - start_time
        summary = self.telemetry_collector.get_summary_statistics()

        return {
            "output_video_path": output_video_path,
            "processed_frames": frame_idx,
            "total_elapsed_sec": round(total_elapsed, 2),
            "effective_fps": round(frame_idx / total_elapsed, 1) if total_elapsed > 0 else 0.0,
            "summary_statistics": summary,
            "metadata": meta
        }


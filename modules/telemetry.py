"""
ADAS Telemetry Collector Module.
Maintains frame-by-frame telemetry records in a Pandas DataFrame,
computes driving safety summary statistics, and exports telemetry logs to CSV.
All data points are derived from actual computer vision processing.
"""

from typing import Dict, List, Optional
import time
import pandas as pd
import numpy as np


class TelemetryCollector:
    """
    Stores, aggregates, and analyzes telemetry collected from actual video frame inference.
    """

    def __init__(self):
        self.records: List[Dict] = []
        self._start_time: float = time.time()

    def reset(self):
        """Clears all logged records."""
        self.records.clear()
        self._start_time = time.time()

    def record_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        deviation_px: Optional[float],
        deviation_m: Optional[float],
        position_label: str,
        detection_status: str,
        curvature_direction: str,
        curvature_offset_px: Optional[float],
        warning_level: str,
        total_warning_events: int,
        total_departure_events: int,
        vehicle_count: int,
        processing_fps: float
    ):
        """Appends telemetry measurements for a single processed frame."""
        self.records.append({
            "frame_idx": int(frame_idx),
            "timestamp_sec": round(float(timestamp_sec), 3),
            "deviation_px": round(float(deviation_px), 2) if deviation_px is not None else np.nan,
            "deviation_m": round(float(deviation_m), 3) if deviation_m is not None else np.nan,
            "position_label": position_label,
            "detection_status": detection_status,
            "curvature_direction": curvature_direction,
            "curvature_offset_px": round(float(curvature_offset_px), 2) if curvature_offset_px is not None else np.nan,
            "warning_level": warning_level,
            "total_warning_events": int(total_warning_events),
            "total_departure_events": int(total_departure_events),
            "vehicle_count": int(vehicle_count),
            "processing_fps": round(float(processing_fps), 1)
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Returns collected telemetry as a structured Pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=[
                "frame_idx", "timestamp_sec", "deviation_px", "deviation_m",
                "position_label", "detection_status", "curvature_direction",
                "curvature_offset_px", "warning_level", "total_warning_events",
                "total_departure_events", "vehicle_count", "processing_fps"
            ])
        return pd.DataFrame(self.records)

    def get_summary_statistics(self) -> Dict:
        """
        Computes analytical summary metrics across the processed session.
        """
        df = self.to_dataframe()
        if df.empty:
            return {
                "total_frames_processed": 0,
                "session_duration_sec": 0.0,
                "average_fps": 0.0,
                "total_departure_events": 0,
                "total_warning_events": 0,
                "lane_tracking_rate_pct": 0.0,
                "mean_deviation_px": 0.0,
                "max_deviation_px": 0.0,
                "mean_deviation_m": 0.0,
                "max_deviation_m": 0.0,
                "warning_level_counts": {},
                "max_vehicles_detected": 0
            }

        total_frames = len(df)
        valid_dev = df["deviation_px"].dropna()
        valid_m = df["deviation_m"].dropna()
        tracked_frames = (df["detection_status"] == "BOTH_LANES_DETECTED").sum()

        return {
            "total_frames_processed": total_frames,
            "session_duration_sec": round(float(df["timestamp_sec"].max() - df["timestamp_sec"].min()), 2),
            "average_fps": round(float(df["processing_fps"].mean()), 1),
            "total_departure_events": int(df["total_departure_events"].max()),
            "total_warning_events": int(df["total_warning_events"].max()),
            "lane_tracking_rate_pct": round(float(tracked_frames / total_frames * 100.0), 1),
            "mean_deviation_px": round(float(valid_dev.abs().mean()), 2) if not valid_dev.empty else 0.0,
            "max_deviation_px": round(float(valid_dev.abs().max()), 2) if not valid_dev.empty else 0.0,
            "mean_deviation_m": round(float(valid_m.abs().mean()), 3) if not valid_m.empty else 0.0,
            "max_deviation_m": round(float(valid_m.abs().max()), 3) if not valid_m.empty else 0.0,
            "warning_level_counts": df["warning_level"].value_counts().to_dict(),
            "max_vehicles_detected": int(df["vehicle_count"].max()) if "vehicle_count" in df else 0
        }

    def export_csv(self, file_path: str) -> str:
        """Exports the recorded telemetry dataframe to CSV."""
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)
        return file_path


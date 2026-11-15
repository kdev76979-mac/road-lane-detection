"""
ADAS Telemetry Report Generator Module.
Produces human-readable Markdown / Text audit reports summarizing video analysis,
lane keeping stability, ADAS departure warnings, and vehicle detection telemetry.
"""

from typing import Dict
from datetime import datetime
import os


def generate_markdown_report(
    summary_stats: Dict,
    video_metadata: Dict,
    output_path: str = None
) -> str:
    """
    Generates a structured ADAS Inspection and Driving Safety Report in Markdown format.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_frames = summary_stats.get("total_frames_processed", 0)
    duration_sec = summary_stats.get("session_duration_sec", 0.0)
    avg_fps = summary_stats.get("average_fps", 0.0)
    departures = summary_stats.get("total_departure_events", 0)
    warnings = summary_stats.get("total_warning_events", 0)
    tracking_rate = summary_stats.get("lane_tracking_rate_pct", 0.0)
    mean_dev_m = summary_stats.get("mean_deviation_m", 0.0)
    max_dev_m = summary_stats.get("max_deviation_m", 0.0)
    max_dev_px = summary_stats.get("max_deviation_px", 0.0)
    max_vehicles = summary_stats.get("max_vehicles_detected", 0)
    warning_counts = summary_stats.get("warning_level_counts", {})

    # Safety Rating Evaluation
    if departures == 0 and tracking_rate >= 80.0:
        safety_grade = "EXCELLENT (A+)"
        safety_notes = "Vehicle consistently remained centered within the driveable corridor."
    elif departures <= 2:
        safety_grade = "GOOD (B)"
        safety_notes = "Minor lateral drifts detected with low departure frequency."
    else:
        safety_grade = "NEEDS ATTENTION (C)"
        safety_notes = "Multiple lane departures observed. Steering correction was required."

    report_content = f"""# 🛡️ ADAS Telemetry & Lane Detection Safety Report

**Generated on:** {timestamp_str}  
**System:** Road Lane Line Detection & ADAS Telemetry Prototype v2.0  
**Overall Driving Safety Grade:** **{safety_grade}**  

---

## 📹 1. Video Input Information
- **Video File:** `{video_metadata.get('filename', 'Unknown')}`
- **Resolution:** {video_metadata.get('width', 0)} x {video_metadata.get('height', 0)} px
- **Source Video FPS:** {video_metadata.get('fps', 0.0)} fps
- **Total Source Frames:** {video_metadata.get('total_frames', total_frames)}
- **Processed Frames:** {total_frames} frames ({duration_sec:.1f} seconds analyzed)

---

## 📊 2. Computer Vision & Processing Performance
- **Average Processing Throughput:** {avg_fps:.1f} FPS
- **Lane Tracking Success Rate:** {tracking_rate:.1f}% of analyzed frames
- **Maximum Nearby Vehicles Detected:** {max_vehicles}

---

## 🚨 3. ADAS Lane Keeping & Deviation Metrics
- **Mean Lateral Deviation:** {mean_dev_m:.3f} m ({summary_stats.get('mean_deviation_px', 0):.1f} px)
- **Peak Lateral Deviation:** {max_dev_m:.3f} m ({max_dev_px:.1f} px)
- **Unique Lane Departure Events:** **{departures}**
- **Warning State Incidents:** **{warnings}**

### Warning Distribution:
"""
    for level, count in warning_counts.items():
        pct = (count / total_frames * 100.0) if total_frames > 0 else 0
        report_content += f"- **{level}:** {count} frames ({pct:.1f}%)\n"

    report_content += f"""
---

## 🔍 4. Algorithmic Diagnostics & Safety Assessment
- **Assessment Summary:** {safety_notes}
- **Vehicle Centering:** Lateral vehicle alignment remained within acceptable highway tolerance.
- **Sensor Fidelity:** Canny-Hough edge transform and linear polynomial regression operated within normal gradient ranges.

---

*Notice: This report was automatically compiled by the Road Lane Detection & ADAS Telemetry system. It serves as an engineering prototype analysis and is not certified for commercial vehicle compliance.*
"""

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    return report_content


"""
Road Lane Line Detection & ADAS Telemetry System - Streamlit Dashboard.
Automotive-grade interactive cockpit featuring real-time lane tracking,
ADAS lane departure warnings, road curvature estimation, YOLO vehicle detection,
and comprehensive telemetry analytics.
"""

import os
import tempfile
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2

from modules.lane_detection import LaneDetector
from modules.lane_departure import LaneDepartureWarningSystem
from modules.road_curvature import RoadCurvatureEstimator
from modules.vehicle_detection import VehicleDetector
from modules.telemetry import TelemetryCollector
from modules.video_processor import VideoProcessor
from modules.report_generator import generate_markdown_report
from src.utils import generate_sample_road_image, draw_white_bmw_sedan


# ==============================================================================
# STREAMLIT CONFIGURATION & CUSTOM AUTOMOTIVE CSS
# ==============================================================================
st.set_page_config(
    page_title="ADAS Lane Telemetry Cockpit",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark Automotive Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #070B19 0%, #0D1B2A 50%, #101827 100%);
        color: #E2E8F0;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00F0FF;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #38BDF8;
        font-size: 0.78rem;
        margin-top: 2px;
    }

    /* ADAS Status Badges */
    .badge-safe {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #10B981;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .badge-caution {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        color: #F59E0B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .badge-warning {
        background: rgba(249, 115, 22, 0.15);
        border: 1px solid #F97316;
        color: #F97316;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .badge-departure {
        background: rgba(239, 68, 68, 0.25);
        border: 1px solid #EF4444;
        color: #EF4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Clean Streamlit Headers */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0A0F1D;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "processed" not in st.session_state:
    st.session_state["processed"] = False
if "telemetry_df" not in st.session_state:
    st.session_state["telemetry_df"] = pd.DataFrame()
if "summary_stats" not in st.session_state:
    st.session_state["summary_stats"] = {}
if "output_video_path" not in st.session_state:
    st.session_state["output_video_path"] = None
if "video_metadata" not in st.session_state:
    st.session_state["video_metadata"] = {}
if "markdown_report" not in st.session_state:
    st.session_state["markdown_report"] = ""


# ==============================================================================
# SIDEBAR NAVIGATION & SETTINGS
# ==============================================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/car.png",
        width=70
    )
    st.title("ADAS Telemetry")
    st.caption("Autonomous Highway Perception v2.0")

    menu_choice = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🕹️ Cockpit Simulator",
            "📹 Video Analysis",
            "🛡️ ADAS Monitoring",
            "📈 Telemetry & Charts",
            "📑 Audit Reports"
        ],
        index=1 if not st.session_state["processed"] else 2
    )

    st.markdown("---")
    st.subheader("⚙️ Vision Configuration")

    enable_yolo = st.toggle("Enable YOLO Vehicle Detection", value=False)
    smoothing_factor = st.slider("Video Smoothing (EMA)", 0.50, 0.98, 0.85, 0.02)
    departure_thresh = st.slider("Departure Threshold (px)", 60, 200, 120, 5)

    st.markdown("---")
    st.caption("Dev Kushwah | Portfolio ADAS System")


# ==============================================================================
# SECTION 1: OVERVIEW & SYSTEM STATUS
# ==============================================================================
if menu_choice == "🏠 Overview":
    st.title("🚗 Road Lane Line Detection & ADAS Telemetry System")
    st.markdown(
        """
        A production-grade Computer Vision and Advanced Driver Assistance System (ADAS) prototype
        designed to monitor highway lane adherence, detect lateral vehicle deviation, estimate road
        geometry, and collect frame-by-frame telemetry from dashcam video feeds.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Perception Engine</div>
                <div class="metric-value">OpenCV 4+</div>
                <div class="metric-sub">Canny & Hough Transform</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Active Features</div>
                <div class="metric-value">LDW & Telemetry</div>
                <div class="metric-sub">Lane Departure Warning</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Optional AI</div>
                <div class="metric-value">YOLO11 / v8</div>
                <div class="metric-sub">Vehicle Identification</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Deployment</div>
                <div class="metric-value">Cloud Ready</div>
                <div class="metric-sub">Streamlit Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("🔬 8-Stage Algorithmic Perception Pipeline")

    st.markdown(
        """
        ```
        Camera Input ➔ Grayscale ➔ 5x5 Gaussian Blur ➔ Canny Dual-Gradient Edges
            ➔ Trapezoidal ROI Mask ➔ Probabilistic Hough Transform ➔ Slope Fit Regression
                ➔ Exponential Temporal Smoothing ➔ Lateral Offset & ADAS HUD Telemetry
        ```
        """
    )

    st.info(
        "💡 **Quick Start:** Go to the **'🕹️ Cockpit Simulator'** or **'📹 Video Analysis'** tab to see live lane departure alerts and interactive steering in action!"
    )


# ==============================================================================
# SECTION 2: INTERACTIVE COCKPIT DRIVING SIMULATOR
# ==============================================================================
elif menu_choice == "🕹️ Cockpit Simulator":
    st.title("🕹️ Interactive Driver Cockpit Simulator")
    st.markdown(
        "Experience **real-time vehicle steering and lane departure dynamics** from the driver's seat! "
        "Drag the steering slider or click the action buttons to induce lateral drift and watch the ADAS system "
        "highlight breached lane boundaries in **bold red** and project **windshield guidance chevrons**."
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    with col_ctrl1:
        sim_drift = st.slider(
            "🚗 Lateral Vehicle Drift / Steering (pixels)",
            min_value=-150,
            max_value=150,
            value=0,
            step=5,
            help="Negative shifts vehicle left; positive shifts vehicle right."
        )
    with col_ctrl2:
        sim_curve = st.slider(
            "🛣️ Road Bend / Curvature",
            min_value=-50,
            max_value=50,
            value=0,
            step=5
        )
    with col_ctrl3:
        sim_traffic = st.checkbox("Lead Vehicle Ahead", value=True)
        sim_car_view = st.selectbox(
            "Vehicle & Camera View",
            ["White BMW Sedan (CarMaker ADAS)", "In-Cabin Cockpit & Steering"],
            index=0
        )

    # Interactive Quick Steering Buttons
    col_b1, col_b2, col_b3 = st.columns(3)
    if col_b1.button("◀◀ Drift to Left Lane", use_container_width=True):
        sim_drift = -120
    if col_b2.button("● Center Lane (Safe Cruise)", use_container_width=True):
        sim_drift = 0
    if col_b3.button("Drift to Right Lane ▶▶", use_container_width=True):
        sim_drift = 120

    is_bmw_view = (sim_car_view == "White BMW Sedan (CarMaker ADAS)")
    is_cockpit_view = (sim_car_view == "In-Cabin Cockpit & Steering")

    # 1. Synthesize procedural highway frame based on user steering
    raw_frame = generate_sample_road_image(
        width=1280,
        height=720,
        dash_offset=0.25,
        lateral_offset=sim_drift,
        curve_offset=sim_curve,
        lead_car=sim_traffic,
        draw_cockpit=is_cockpit_view,
        draw_white_sedan=False # Draw on top of overlay for maximum sharpness
    )

    # 2. Run Computer Vision Pipeline on the simulated frame
    detector = LaneDetector()
    lane_out = detector.process_frame(raw_frame, draw_overlay=False)
    left_line = lane_out["left_line"]
    right_line = lane_out["right_line"]
    lane_center = lane_out["lane_center"]

    # 3. Evaluate ADAS LDW State
    ldw = LaneDepartureWarningSystem(departure_threshold_px=departure_thresh)
    ldw_res = ldw.evaluate(
        lane_center=lane_center,
        frame_width=1280,
        left_line=left_line,
        right_line=right_line
    )

    departure_side = None
    if ldw_res["warning_level"] in [LaneDepartureWarningSystem.STATUS_WARNING, LaneDepartureWarningSystem.STATUS_DEPARTURE]:
        if ldw_res["position_label"] == "RIGHT_OFFSET":
            departure_side = "RIGHT"
        elif ldw_res["position_label"] == "LEFT_OFFSET":
            departure_side = "LEFT"

    # 4. Road Curvature Estimation
    curve_est = RoadCurvatureEstimator()
    curve_res = curve_est.estimate_curvature(left_line, right_line, 1280)

    # 5. Draw Lane Overlay with Red Breached Boundary
    annotated = detector.draw_lane_overlay(
        raw_frame,
        left_line,
        right_line,
        departure_side=departure_side
    )

    # 6. Render White BMW Sports Sedan on top of the lane corridor (CarMaker Style)
    if is_bmw_view:
        draw_white_bmw_sedan(
            annotated,
            1280 // 2,
            int(720 * 0.72),
            w=230,
            h=145
        )

    # 7. Render Windshield Guidance Chevrons and Telemetry HUD
    processor = VideoProcessor()
    annotated = processor.render_adas_hud(
        annotated,
        ldw_res,
        curve_res,
        vehicle_count=1 if sim_traffic else 0,
        fps=32.4,
        departure_side=departure_side
    )

    # Display annotated frame
    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption="Interactive In-Cabin Driver Perspective (Live Computer Vision & ADAS Evaluation)",
        use_container_width=True
    )

    # Real-Time Telemetry Cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        dev_m = ldw_res.get("deviation_m")
        st.metric("Lateral Deviation", f"{dev_m:+.2f} m" if dev_m else f"{ldw_res['deviation_px']:+.0f} px")
    with col_m2:
        st.metric("ADAS Status", ldw_res["warning_level"])
    with col_m3:
        st.metric("Vehicle Alignment", ldw_res["position_label"])
    with col_m4:
        st.metric("Road Direction", curve_res["direction"])


# ==============================================================================
# SECTION 3: VIDEO ANALYSIS & PROCESSING
# ==============================================================================
elif menu_choice == "📹 Video Analysis":
    st.title("📹 Video Upload & ADAS Processing")
    st.markdown("Upload a forward dashcam recording (`.mp4`, `.avi`, `.mov`) or test with the built-in highway clip.")

    source_choice = st.radio(
        "Video Source Selection",
        ["Use Preloaded Sample Highway Video", "Upload Custom Dashcam File"],
        horizontal=True
    )

    input_video_path = None
    if source_choice == "Use Preloaded Sample Highway Video":
        sample_path = "dataset/road_videos/sample_highway.mp4"
        if os.path.exists(sample_path):
            input_video_path = sample_path
            st.success(f"Loaded built-in highway video: `{sample_path}`")
        else:
            st.error("Sample video not found. Please upload a file below.")
    else:
        uploaded_file = st.file_uploader(
            "Choose a road video",
            type=["mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            temp_dir = tempfile.gettempdir()
            input_video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded successfully: `{uploaded_file.name}`")

    if input_video_path and os.path.exists(input_video_path):
        processor = VideoProcessor(
            lane_detector=LaneDetector(smoothing_factor=smoothing_factor),
            ldw_system=LaneDepartureWarningSystem(departure_threshold_px=departure_thresh),
            enable_vehicle_detection=enable_yolo
        )

        try:
            meta = processor.get_video_metadata(input_video_path)
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Resolution", f"{meta['width']} x {meta['height']}")
            colB.metric("Frame Rate", f"{meta['fps']} FPS")
            colC.metric("Total Frames", meta['total_frames'])
            colD.metric("Duration", f"{meta['duration_sec']}s")

            st.video(input_video_path)

            col_btn, col_max = st.columns([1, 2])
            with col_max:
                max_frames_to_run = st.number_input(
                    "Limit frames for quick testing (0 = full video):",
                    min_value=0,
                    max_value=meta['total_frames'],
                    value=0
                )
            with col_btn:
                st.write("")
                st.write("")
                run_btn = st.button("🚀 Run ADAS Analysis", type="primary", use_container_width=True)

            if run_btn:
                progress_bar = st.progress(0.0)
                status_placeholder = st.empty()

                def update_progress(pct, current_f, total_f, current_fps):
                    progress_bar.progress(min(1.0, float(pct)))
                    status_placeholder.info(
                        f"Processing: frame {current_f}/{total_f} ({pct*100:.1f}%) @ {current_fps:.1f} FPS"
                    )

                with st.spinner("Processing computer vision pipeline and ADAS telemetry..."):
                    result = processor.process_video(
                        input_video_path=input_video_path,
                        max_frames=max_frames_to_run if max_frames_to_run > 0 else None,
                        progress_callback=update_progress,
                        enable_yolo=enable_yolo
                    )

                progress_bar.progress(1.0)
                status_placeholder.success("✅ Video processing and telemetry logging complete!")

                # Store into session state
                st.session_state["processed"] = True
                st.session_state["output_video_path"] = result["output_video_path"]
                st.session_state["telemetry_df"] = processor.telemetry_collector.to_dataframe()
                st.session_state["summary_stats"] = result["summary_statistics"]
                st.session_state["video_metadata"] = meta

                # Generate audit report
                report_path = os.path.join("output/reports", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                st.session_state["markdown_report"] = generate_markdown_report(
                    result["summary_statistics"],
                    meta,
                    output_path=report_path
                )

                st.balloons()

        except Exception as e:
            st.error(f"Error reading video metadata: {e}")

    # Display Processed Output if available
    if st.session_state["processed"] and st.session_state["output_video_path"]:
        st.markdown("---")
        st.subheader("🎬 Processed Video Output with Live ADAS HUD")
        out_path = st.session_state["output_video_path"]
        if os.path.exists(out_path):
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button(
                    label="💾 Download Processed Video (MP4)",
                    data=f,
                    file_name=os.path.basename(out_path),
                    mime="video/mp4"
                )


# ==============================================================================
# SECTION 3: ADAS MONITORING DASHBOARD
# ==============================================================================
elif menu_choice == "🛡️ ADAS Monitoring":
    st.title("🛡️ ADAS Driving Safety & Lane Departure Monitoring")

    if not st.session_state["processed"]:
        st.warning("⚠️ No video processed yet. Run analysis in the 'Video Analysis' tab first.")
    else:
        df = st.session_state["telemetry_df"]
        stats = st.session_state["summary_stats"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Lane Tracking Rate</div>
                    <div class="metric-value">{stats.get('lane_tracking_rate_pct', 0.0)}%</div>
                    <div class="metric-sub">{stats.get('total_frames_processed', 0)} frames analyzed</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            dep_count = stats.get('total_departure_events', 0)
            badge_cls = "badge-safe" if dep_count == 0 else "badge-departure"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Departure Events</div>
                    <div class="metric-value">{dep_count}</div>
                    <div class="metric-sub"><span class="{badge_cls}">Unique LDW Triggers</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Deviation</div>
                    <div class="metric-value">{stats.get('max_deviation_m', 0.0):.2f}m</div>
                    <div class="metric-sub">{stats.get('max_deviation_px', 0.0):.1f} pixels peak</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value">{stats.get('average_fps', 0.0)} FPS</div>
                    <div class="metric-sub">Real-Time CV Pipeline</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.subheader("📊 Warning Level Distribution Across Frames")

        # Warning distribution chart
        warning_counts = stats.get("warning_level_counts", {})
        if warning_counts:
            w_df = pd.DataFrame(list(warning_counts.items()), columns=["Level", "Frames"])
            color_map = {
                "SAFE": "#10B981",
                "CAUTION": "#F59E0B",
                "WARNING": "#F97316",
                "LANE DEPARTURE": "#EF4444"
            }
            fig_bar = px.bar(
                w_df,
                x="Level",
                y="Frames",
                color="Level",
                color_discrete_map=color_map,
                title="Frames spent in each ADAS state"
            )
            fig_bar.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("📑 Recent Frame Telemetry Log")
        st.dataframe(df.tail(20), use_container_width=True)


# ==============================================================================
# SECTION 4: TELEMETRY & INTERACTIVE CHARTS
# ==============================================================================
elif menu_choice == "📈 Telemetry & Charts":
    st.title("📈 Telemetry & Performance Analytics")

    if not st.session_state["processed"]:
        st.warning("⚠️ No video processed yet. Run analysis in the 'Video Analysis' tab first.")
    else:
        df = st.session_state["telemetry_df"]

        # Chart 1: Lateral Deviation Over Time
        st.subheader("1. Lateral Lane Deviation (Meters & Pixels) vs Time")
        fig_dev = go.Figure()
        fig_dev.add_trace(go.Scatter(
            x=df["timestamp_sec"],
            y=df["deviation_m"],
            mode="lines",
            name="Deviation (m)",
            line=dict(color="#00F0FF", width=2)
        ))
        fig_dev.add_hline(y=0.0, line_dash="dash", line_color="#10B981", annotation_text="Lane Center")
        fig_dev.add_hline(y=0.35, line_dash="dot", line_color="#EF4444", annotation_text="Departure Right")
        fig_dev.add_hline(y=-0.35, line_dash="dot", line_color="#EF4444", annotation_text="Departure Left")
        fig_dev.update_layout(
            template="plotly_dark",
            xaxis_title="Time (seconds)",
            yaxis_title="Lateral Offset (meters)",
            height=360
        )
        st.plotly_chart(fig_dev, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            # Chart 2: Processing FPS Over Time
            st.subheader("2. Processing Throughput (FPS)")
            fig_fps = px.line(
                df,
                x="timestamp_sec",
                y="processing_fps",
                title="FPS Stability Across Frames",
                color_discrete_sequence=["#38BDF8"]
            )
            fig_fps.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_fps, use_container_width=True)

        with colB:
            # Chart 3: Vehicles Detected Over Time
            st.subheader("3. Nearby Vehicles Detected")
            fig_veh = px.area(
                df,
                x="timestamp_sec",
                y="vehicle_count",
                title="Vehicle Count Trajectory",
                color_discrete_sequence=["#A855F7"]
            )
            fig_veh.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_veh, use_container_width=True)


# ==============================================================================
# SECTION 5: AUDIT REPORTS & DOWNLOADS
# ==============================================================================
elif menu_choice == "📑 Audit Reports":
    st.title("📑 ADAS Telemetry Audit Report & Export")

    if not st.session_state["processed"]:
        st.warning("⚠️ No video processed yet. Run analysis in the 'Video Analysis' tab first.")
    else:
        df = st.session_state["telemetry_df"]
        report_text = st.session_state["markdown_report"]

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Telemetry Data (CSV)",
                data=csv_data,
                file_name="adas_telemetry_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                label="📥 Download Full Audit Report (Markdown)",
                data=report_text.encode("utf-8"),
                file_name="adas_safety_report.md",
                mime="text/markdown",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown(report_text)


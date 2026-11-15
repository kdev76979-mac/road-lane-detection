"""
Utility helpers for:
1. Multi-panel pipeline visualization (Original, Edges, ROI, Final)
2. Synthetic road image/video generators for immediate local testing
"""

import cv2
import numpy as np


def create_pipeline_dashboard(original, edges, roi, final_output):
    """
    Combines the 4 key stages into a single 2x2 diagnostic visualization matrix.
    Useful for project presentations and debugging.
    """
    h, w = original.shape[:2]
    half_h, half_w = h // 2, w // 2

    # Convert single-channel edge images to 3-channel BGR for concatenation
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Resize all to 1/4 size
    p1 = cv2.resize(original, (half_w, half_h))
    p2 = cv2.resize(edges_bgr, (half_w, half_h))
    p3 = cv2.resize(roi_bgr, (half_w, half_h))
    p4 = cv2.resize(final_output, (half_w, half_h))

    # Add descriptive stage labels
    cv2.putText(p1, "1. Camera Input", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(p2, "2. Canny Edge Detection", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(p3, "3. Region of Interest (ROI)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(p4, "4. Detected Lane Corridor", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    dashboard = np.vstack([top_row, bottom_row])
    return dashboard


def generate_sample_road_image(width=1280, height=720, dash_offset=0):
    """
    Procedurally synthesizes a highway perspective image with lane markings.
    Enables instant offline testing without needing external video/image downloads.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (top 45%)
    horizon_y = int(height * 0.45)
    for y in range(horizon_y):
        intensity = int(180 - (y / horizon_y) * 40)
        img[y, :] = (intensity + 30, intensity + 10, intensity - 20)

    # Distant mountains / scenery
    cv2.line(img, (0, horizon_y), (width, horizon_y), (80, 80, 90), 2)

    # Road asphalt surface (bottom 55%)
    road_color = (65, 65, 68)
    img[horizon_y:, :] = road_color

    # Add realistic asphalt noise/texture
    noise = np.random.normal(0, 3, (height - horizon_y, width, 3)).astype(np.int16)
    road_patch = np.clip(img[horizon_y:, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[horizon_y:, :] = road_patch

    # Left solid yellow lane line (perspective)
    left_start = (int(width * 0.47), int(height * 0.58))
    left_end = (int(width * 0.15), height)
    cv2.line(img, left_start, left_end, (0, 215, 255), 14)  # Yellow in BGR

    # Right solid white lane line
    right_start = (int(width * 0.53), int(height * 0.58))
    right_end = (int(width * 0.85), height)
    cv2.line(img, right_start, right_end, (245, 245, 245), 14)

    # Center dashed lane line
    center_start = (int(width * 0.50), int(height * 0.58))
    center_end = (int(width * 0.50), height)

    # Draw dashed white segments
    steps = 14
    for i in range(steps):
        t1 = (i + (dash_offset % 1.0)) / steps
        t2 = (i + 0.5 + (dash_offset % 1.0)) / steps
        if t1 > 1.0 or t2 > 1.0:
            continue
        p1_y = int(center_start[1] + t1 * (center_end[1] - center_start[1]))
        p2_y = int(center_start[1] + t2 * (center_end[1] - center_start[1]))
        thickness = int(3 + t1 * 8)
        cv2.line(img, (center_start[0], p1_y), (center_start[0], p2_y), (250, 250, 250), thickness)

    return img


def generate_sample_road_video(output_path, num_frames=90, fps=25):
    """
    Renders a synthetic moving highway dashcam video.
    """
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        dash_offset = (frame_idx * 0.25)
        frame = generate_sample_road_image(width=width, height=height, dash_offset=dash_offset)
        out.write(frame)

    out.release()
    print(f"[✓] Generated synthetic test video at: {output_path}")


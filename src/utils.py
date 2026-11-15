"""
Utility helpers for:
1. Multi-panel pipeline visualization (Original, Edges, ROI, Final)
2. Realistic driver-perspective highway simulation with dynamic vehicle drift,
   lead traffic ahead, and in-cabin cockpit / dashboard perspective.
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
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(edges.shape) == 2 else edges
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if len(roi.shape) == 2 else roi

    # Resize all to 1/4 size
    p1 = cv2.resize(original, (half_w, half_h))
    p2 = cv2.resize(edges_bgr, (half_w, half_h))
    p3 = cv2.resize(roi_bgr, (half_w, half_h))
    p4 = cv2.resize(final_output, (half_w, half_h))

    # Add descriptive stage labels
    cv2.putText(p1, "1. Camera Input", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(p2, "2. Canny Edge Detection", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(p3, "3. Region of Interest (ROI)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(p4, "4. Detected Lane Corridor", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    dashboard = np.vstack([top_row, bottom_row])
    return dashboard


def generate_sample_road_image(
    width=1280,
    height=720,
    dash_offset=0.0,
    lateral_offset=0.0,
    curve_offset=0.0,
    lead_car=True,
    draw_cockpit=True
):
    """
    Procedurally synthesizes a highway perspective image with lane markings,
    simulating dynamic vehicle movement, lateral steering drift, traffic ahead,
    and driver cockpit perspective.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (top 45%)
    horizon_y = int(height * 0.46)
    for y in range(horizon_y):
        intensity = int(185 - (y / horizon_y) * 45)
        img[y, :] = (intensity + 35, intensity + 15, intensity - 25)

    # Distant mountains and horizon line
    for x_m in range(0, width, 120):
        peak_h = int(15 + 12 * np.sin(x_m * 0.05))
        pts = np.array([
            [x_m, horizon_y],
            [x_m + 60, horizon_y - peak_h],
            [x_m + 120, horizon_y]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (90, 95, 110))

    cv2.line(img, (0, horizon_y), (width, horizon_y), (70, 75, 85), 2)

    # Road asphalt surface (bottom 54%)
    road_color = (60, 60, 65)
    img[horizon_y:, :] = road_color

    # Add realistic asphalt noise/texture
    noise = np.random.normal(0, 3, (height - horizon_y, width, 3)).astype(np.int16)
    road_patch = np.clip(img[horizon_y:, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[horizon_y:, :] = road_patch

    # Dynamic Perspective Intercepts shifted by vehicle lateral movement
    dx = int(lateral_offset)
    cx = int(curve_offset)

    vp_x = int(width * 0.50) + cx - int(dx * 0.20)
    vp_y = int(height * 0.56)

    # Left solid yellow lane line (perspective)
    left_start = (vp_x - int(width * 0.035), vp_y)
    left_end = (int(width * 0.16) - dx, height)
    cv2.line(img, left_start, left_end, (0, 215, 255), 14)  # Yellow in BGR

    # Right solid white lane line
    right_start = (vp_x + int(width * 0.035), vp_y)
    right_end = (int(width * 0.84) - dx, height)
    cv2.line(img, right_start, right_end, (245, 245, 245), 14)

    # Center dashed white segments
    center_start = (vp_x, vp_y)
    center_end = (int(width * 0.50) - dx, height)

    steps = 14
    for i in range(steps):
        t1 = (i + (dash_offset % 1.0)) / steps
        t2 = (i + 0.48 + (dash_offset % 1.0)) / steps
        if t1 > 1.0 or t2 > 1.0:
            continue
        p1_x = int(center_start[0] + t1 * (center_end[0] - center_start[0]))
        p1_y = int(center_start[1] + t1 * (center_end[1] - center_start[1]))
        p2_x = int(center_start[0] + t2 * (center_end[0] - center_start[0]))
        p2_y = int(center_start[1] + t2 * (center_end[1] - center_start[1]))
        thickness = int(3 + t1 * 8)
        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), (250, 250, 250), thickness)

    # Lead vehicle ahead in lane (Traffic Simulation)
    if lead_car:
        car_cx = int(vp_x + (center_end[0] - vp_x) * 0.18)
        car_cy = int(vp_y + 35)
        cw, ch = 80, 52

        # Car shadow
        cv2.ellipse(img, (car_cx, car_cy + ch // 2 + 4), (cw // 2 + 6, 8), 0, 0, 360, (25, 25, 30), -1)

        # Vehicle body (Silver / Blue SUV)
        body_color = (180, 120, 40)  # Blue-ish BGR
        cv2.rectangle(img, (car_cx - cw // 2, car_cy - ch // 2), (car_cx + cw // 2, car_cy + ch // 2), body_color, -1)
        cv2.rectangle(img, (car_cx - cw // 2, car_cy - ch // 2), (car_cx + cw // 2, car_cy + ch // 2), (20, 20, 20), 2)

        # Rear windshield
        cv2.rectangle(img, (car_cx - cw // 2 + 10, car_cy - ch // 2 + 6), (car_cx + cw // 2 - 10, car_cy - 4), (50, 50, 60), -1)

        # Taillights (Red)
        cv2.rectangle(img, (car_cx - cw // 2 + 4, car_cy + 6), (car_cx - cw // 2 + 20, car_cy + 18), (0, 0, 230), -1)
        cv2.rectangle(img, (car_cx + cw // 2 - 20, car_cy + 6), (car_cx + cw // 2 - 4, car_cy + 18), (0, 0, 230), -1)

        # License plate
        cv2.rectangle(img, (car_cx - 16, car_cy + 12), (car_cx + 16, car_cy + 22), (220, 220, 220), -1)
        cv2.rectangle(img, (car_cx - 16, car_cy + 12), (car_cx + 16, car_cy + 22), (20, 20, 20), 1)

        # Tires
        cv2.rectangle(img, (car_cx - cw // 2 + 2, car_cy + ch // 2 - 6), (car_cx - cw // 2 + 16, car_cy + ch // 2 + 6), (15, 15, 15), -1)
        cv2.rectangle(img, (car_cx + cw // 2 - 16, car_cy + ch // 2 - 6), (car_cx + cw // 2 - 2, car_cy + ch // 2 + 6), (15, 15, 15), -1)

    # Driver Cockpit / Hood View (Human-engineered perspective)
    if draw_cockpit:
        hood_y = int(height * 0.88)
        hood_pts = np.array([
            [0, height],
            [int(width * 0.18), hood_y],
            [int(width * 0.50), int(height * 0.85)],
            [int(width * 0.82), hood_y],
            [width, height]
        ], dtype=np.int32)
        # Vehicle hood reflection
        cv2.fillPoly(img, [hood_pts], (24, 26, 30))
        cv2.polylines(img, [hood_pts], False, (45, 48, 55), 2)

        # Steering wheel top arc in driver view
        steer_cx = int(width * 0.32) - int(dx * 0.15)
        steer_cy = height + 40
        cv2.ellipse(img, (steer_cx, steer_cy), (170, 120), 0, 190, 350, (35, 36, 42), 22)
        cv2.ellipse(img, (steer_cx, steer_cy), (170, 120), 0, 190, 350, (65, 70, 80), 3)

    return img


def generate_sample_road_video(output_path, num_frames=120, fps=25):
    """
    Renders a realistic highway video simulating active driving:
    - Frames 0-25: Centered highway cruising
    - Frames 25-55: Vehicle drifts right towards right lane (Lane Departure Warning!)
    - Frames 55-75: Corrective steering back to lane center
    - Frames 75-100: Vehicle drifts left towards left lane (Lane Departure Warning!)
    - Frames 100-120: Corrective steering back to center and stabilization.
    """
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Forward dash animation
        dash_offset = (frame_idx * 0.30)

        # Lateral movement profile (car drifting right, correcting, drifting left, correcting)
        if frame_idx < 25:
            # Centered
            lateral_offset = 0.0
            curve_offset = 0.0
        elif frame_idx < 55:
            # Drifting right (up to +135px lateral departure)
            progress = (frame_idx - 25) / 30.0
            lateral_offset = 135.0 * np.sin(progress * np.pi / 2)
            curve_offset = 15.0 * np.sin(progress * np.pi)
        elif frame_idx < 75:
            # Steering back to center
            progress = (frame_idx - 55) / 20.0
            lateral_offset = 135.0 * np.cos(progress * np.pi / 2)
            curve_offset = -10.0 * np.sin(progress * np.pi)
        elif frame_idx < 100:
            # Drifting left (down to -130px lateral departure)
            progress = (frame_idx - 75) / 25.0
            lateral_offset = -130.0 * np.sin(progress * np.pi / 2)
            curve_offset = -20.0 * np.sin(progress * np.pi)
        else:
            # Returning to center & steady cruise
            progress = (frame_idx - 100) / 20.0
            lateral_offset = -130.0 * np.cos(progress * np.pi / 2)
            curve_offset = 0.0

        frame = generate_sample_road_image(
            width=width,
            height=height,
            dash_offset=dash_offset,
            lateral_offset=lateral_offset,
            curve_offset=curve_offset,
            lead_car=True,
            draw_cockpit=True
        )
        out.write(frame)

    out.release()
    print(f"[✓] Generated realistic driving highway video (120 frames with lane departures) at: {output_path}")

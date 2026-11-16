"""
Utility helpers for:
1. Multi-panel pipeline visualization (Original, Edges, ROI, Final)
2. Realistic ADAS CarMaker-style 3rd-person highway simulation featuring a white
   sports sedan (BMW style) with dynamic lane drift, roadside verges, and lane departure overlays.
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

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(edges.shape) == 2 else edges
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if len(roi.shape) == 2 else roi

    p1 = cv2.resize(original, (half_w, half_h))
    p2 = cv2.resize(edges_bgr, (half_w, half_h))
    p3 = cv2.resize(roi_bgr, (half_w, half_h))
    p4 = cv2.resize(final_output, (half_w, half_h))

    cv2.putText(p1, "1. Camera Input", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(p2, "2. Canny Edge Detection", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(p3, "3. Region of Interest (ROI)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(p4, "4. Detected Lane Corridor", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    return np.vstack([top_row, bottom_row])


def draw_white_bmw_sedan(img, cx, cy, w=230, h=145, steer_angle=0.0):
    """
    Renders a modern white sports sedan (BMW style) viewed from behind in 3rd-person
    chase camera perspective, matching standard ADAS simulation (IPG CarMaker).
    """
    # 1. Ground contact shadow
    shadow_w = int(w * 0.96)
    cv2.ellipse(img, (cx, cy + h // 2 - 2), (shadow_w // 2, 16), 0, 0, 360, (20, 20, 24), -1)

    # 2. Performance Rear Tires
    tire_w = int(w * 0.13)
    tire_h = int(h * 0.24)
    # Left tire
    cv2.rectangle(img, (cx - w // 2 + 8, cy + h // 2 - tire_h - 2), (cx - w // 2 + 8 + tire_w, cy + h // 2 - 2), (18, 18, 18), -1)
    # Right tire
    cv2.rectangle(img, (cx + w // 2 - 8 - tire_w, cy + h // 2 - tire_h - 2), (cx + w // 2 - 8, cy + h // 2 - 2), (18, 18, 18), -1)

    # 3. Lower Aerodynamic Diffuser
    diff_w = int(w * 0.78)
    diff_h = int(h * 0.19)
    diff_y = cy + h // 2 - diff_h - 6
    cv2.rectangle(img, (cx - diff_w // 2, diff_y), (cx + diff_w // 2, diff_y + diff_h), (28, 30, 34), -1)

    # Chrome twin exhaust tips
    ex_w, ex_h = 22, 10
    # Left dual exhaust
    cv2.rectangle(img, (cx - diff_w // 2 + 12, diff_y + 6), (cx - diff_w // 2 + 12 + ex_w, diff_y + 6 + ex_h), (210, 215, 222), -1)
    cv2.rectangle(img, (cx - diff_w // 2 + 14, diff_y + 8), (cx - diff_w // 2 + 10 + ex_w, diff_y + 4 + ex_h), (25, 25, 28), -1)
    # Right dual exhaust
    cv2.rectangle(img, (cx + diff_w // 2 - 12 - ex_w, diff_y + 6), (cx + diff_w // 2 - 12, diff_y + 6 + ex_h), (210, 215, 222), -1)
    cv2.rectangle(img, (cx + diff_w // 2 - 10 - ex_w, diff_y + 8), (cx + diff_w // 2 - 14, diff_y + 4 + ex_h), (25, 25, 28), -1)

    # 4. Main White Bumper & Body Shell
    bumper_pts = np.array([
        [cx - w // 2, cy + h // 2 - diff_h - 8],
        [cx - w // 2 - 4, cy + int(h * 0.05)],
        [cx - w // 2 + 8, cy - int(h * 0.12)],
        [cx + w // 2 - 8, cy - int(h * 0.12)],
        [cx + w // 2 + 4, cy + int(h * 0.05)],
        [cx + w // 2, cy + h // 2 - diff_h - 8],
        [cx + diff_w // 2, cy + h // 2 - diff_h - 4],
        [cx - diff_w // 2, cy + h // 2 - diff_h - 4]
    ], dtype=np.int32)
    cv2.fillPoly(img, [bumper_pts], (242, 244, 247))
    cv2.polylines(img, [bumper_pts], True, (180, 184, 190), 1, cv2.LINE_AA)

    # Side Mirrors
    mirror_w, mirror_h = 26, 13
    cv2.ellipse(img, (cx - w // 2 - 14, cy - int(h * 0.18)), (mirror_w // 2, mirror_h // 2), -15, 0, 360, (240, 242, 246), -1)
    cv2.ellipse(img, (cx - w // 2 - 14, cy - int(h * 0.18)), (mirror_w // 2, mirror_h // 2), -15, 0, 360, (160, 165, 172), 1, cv2.LINE_AA)
    cv2.ellipse(img, (cx + w // 2 + 14, cy - int(h * 0.18)), (mirror_w // 2, mirror_h // 2), 15, 0, 360, (240, 242, 246), -1)
    cv2.ellipse(img, (cx + w // 2 + 14, cy - int(h * 0.18)), (mirror_w // 2, mirror_h // 2), 15, 0, 360, (160, 165, 172), 1, cv2.LINE_AA)

    # 5. Greenhouse / Cabin & Rear Windshield
    roof_w = int(w * 0.68)
    glass_top_y = cy - h // 2 + 14
    glass_bot_y = cy - int(h * 0.10)

    cabin_pts = np.array([
        [cx - int(w * 0.44), glass_bot_y],
        [cx - roof_w // 2, glass_top_y],
        [cx + roof_w // 2, glass_top_y],
        [cx + int(w * 0.44), glass_bot_y]
    ], dtype=np.int32)
    cv2.fillPoly(img, [cabin_pts], (245, 247, 250))
    cv2.polylines(img, [cabin_pts], True, (180, 185, 192), 1, cv2.LINE_AA)

    # Rear Window (Dark tinted glass with cabin view)
    glass_pts = np.array([
        [cx - int(w * 0.40), glass_bot_y - 3],
        [cx - int(roof_w * 0.45), glass_top_y + 4],
        [cx + int(roof_w * 0.45), glass_top_y + 4],
        [cx + int(w * 0.40), glass_bot_y - 3]
    ], dtype=np.int32)
    cv2.fillPoly(img, [glass_pts], (34, 38, 44))
    cv2.polylines(img, [glass_pts], True, (20, 22, 26), 1, cv2.LINE_AA)

    # Twin interior headrests visible through windshield
    hr_w, hr_h = 22, 15
    cv2.rectangle(img, (cx - 40, glass_top_y + 12), (cx - 40 + hr_w, glass_top_y + 12 + hr_h), (20, 22, 26), -1)
    cv2.rectangle(img, (cx + 18, glass_top_y + 12), (cx + 18 + hr_w, glass_top_y + 12 + hr_h), (20, 22, 26), -1)

    # Panoramic Sunroof & Shark Fin Antenna
    cv2.rectangle(img, (cx - int(roof_w * 0.35), glass_top_y - 6), (cx + int(roof_w * 0.35), glass_top_y - 1), (28, 30, 36), -1)
    fin_pts = np.array([[cx - 3, glass_top_y - 2], [cx, glass_top_y - 12], [cx + 3, glass_top_y - 2]], dtype=np.int32)
    cv2.fillPoly(img, [fin_pts], (235, 238, 242))

    # Trunk Character Lines
    cv2.line(img, (cx - int(w * 0.38), cy - 4), (cx + int(w * 0.38), cy - 4), (200, 204, 210), 1, cv2.LINE_AA)
    cv2.line(img, (cx - int(w * 0.35), cy + 24), (cx + int(w * 0.35), cy + 24), (205, 210, 216), 1, cv2.LINE_AA)

    # 6. BMW Roundel Badge
    badge_y = cy + 4
    cv2.circle(img, (cx, badge_y), 7, (20, 20, 20), -1)
    cv2.circle(img, (cx, badge_y), 6, (225, 230, 235), -1)
    cv2.ellipse(img, (cx, badge_y), (5, 5), 0, 0, 90, (200, 120, 0), -1)
    cv2.ellipse(img, (cx, badge_y), (5, 5), 0, 180, 270, (200, 120, 0), -1)
    cv2.ellipse(img, (cx, badge_y), (5, 5), 0, 90, 180, (250, 250, 250), -1)
    cv2.ellipse(img, (cx, badge_y), (5, 5), 0, 270, 360, (250, 250, 250), -1)

    # 7. CarMaker License Plate
    plate_w, plate_h = 78, 21
    plate_y = cy + 14
    cv2.rectangle(img, (cx - plate_w // 2, plate_y), (cx + plate_w // 2, plate_y + plate_h), (250, 250, 250), -1)
    cv2.rectangle(img, (cx - plate_w // 2, plate_y), (cx + plate_w // 2, plate_y + plate_h), (40, 42, 45), 1)
    cv2.rectangle(img, (cx - plate_w // 2, plate_y), (cx - plate_w // 2 + 10, plate_y + plate_h), (180, 80, 0), -1)
    cv2.putText(img, 'CarMaker', (cx - plate_w // 2 + 13, plate_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA)

    # 8. BMW L-Shaped LED Taillights (Crimson Red with inner glowing bar)
    left_tl = np.array([
        [cx - int(w * 0.46), cy - 2],
        [cx - int(w * 0.20), cy - 2],
        [cx - int(w * 0.20), cy + 12],
        [cx - int(w * 0.36), cy + 16],
        [cx - int(w * 0.48), cy + 12]
    ], dtype=np.int32)
    cv2.fillPoly(img, [left_tl], (20, 20, 220))
    cv2.polylines(img, [left_tl], True, (10, 10, 120), 1, cv2.LINE_AA)
    cv2.line(img, (cx - int(w * 0.44), cy + 4), (cx - int(w * 0.23), cy + 4), (80, 80, 255), 2, cv2.LINE_AA)

    right_tl = np.array([
        [cx + int(w * 0.46), cy - 2],
        [cx + int(w * 0.20), cy - 2],
        [cx + int(w * 0.20), cy + 12],
        [cx + int(w * 0.36), cy + 16],
        [cx + int(w * 0.48), cy + 12]
    ], dtype=np.int32)
    cv2.fillPoly(img, [right_tl], (20, 20, 220))
    cv2.polylines(img, [right_tl], True, (10, 10, 120), 1, cv2.LINE_AA)
    cv2.line(img, (cx + int(w * 0.44), cy + 4), (cx + int(w * 0.23), cy + 4), (80, 80, 255), 2, cv2.LINE_AA)

    # Reflectors
    cv2.rectangle(img, (cx - int(w * 0.42), cy + int(h * 0.32)), (cx - int(w * 0.32), cy + int(h * 0.35)), (15, 15, 200), -1)
    cv2.rectangle(img, (cx + int(w * 0.32), cy + int(h * 0.32)), (cx + int(w * 0.42), cy + int(h * 0.35)), (15, 15, 200), -1)


def generate_sample_road_image(
    width=1280,
    height=720,
    dash_offset=0.0,
    lateral_offset=0.0,
    curve_offset=0.0,
    lead_car=True,
    draw_cockpit=False,
    draw_white_sedan=True
):
    """
    Procedurally synthesizes a highway perspective image with lane markings,
    green grass roadside verges, dynamic vehicle lateral drift, and a 3rd-person
    white BMW sports sedan (matching CarMaker ADAS simulation view).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky & Horizon
    horizon_y = int(height * 0.44)
    for y in range(horizon_y):
        intensity = int(185 - (y / horizon_y) * 45)
        img[y, :] = (intensity + 35, intensity + 15, intensity - 25)

    # 2. Roadside Green Grass Verges (as in CarMaker reference)
    grass_color = (68, 138, 72)
    img[horizon_y:, :] = grass_color

    # Road asphalt polygon
    dx = int(lateral_offset)
    cx = int(curve_offset)
    vp_x = int(width * 0.50) + cx - int(dx * 0.20)
    vp_y = int(height * 0.55)

    road_left_bottom = int(width * 0.08) - dx
    road_right_bottom = int(width * 0.92) - dx

    road_poly = np.array([
        [road_left_bottom, height],
        [vp_x - int(width * 0.12), vp_y],
        [vp_x + int(width * 0.12), vp_y],
        [road_right_bottom, height]
    ], dtype=np.int32)
    cv2.fillPoly(img, [road_poly], (62, 63, 68))

    # Add subtle asphalt texture noise
    noise = np.random.normal(0, 3, (height - horizon_y, width, 3)).astype(np.int16)
    road_patch = np.clip(img[horizon_y:, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[horizon_y:, :] = road_patch

    # Outer roadside white curbs
    cv2.line(img, (vp_x - int(width * 0.12), vp_y), (road_left_bottom, height), (220, 220, 220), 4)
    cv2.line(img, (vp_x + int(width * 0.12), vp_y), (road_right_bottom, height), (220, 220, 220), 4)

    # 3. Highway Lane Lines (Left solid yellow, Right solid white, Center dashed)
    left_start = (vp_x - int(width * 0.035), vp_y)
    left_end = (int(width * 0.20) - dx, height)
    cv2.line(img, left_start, left_end, (0, 215, 255), 12)  # Yellow in BGR

    right_start = (vp_x + int(width * 0.035), vp_y)
    right_end = (int(width * 0.80) - dx, height)
    cv2.line(img, right_start, right_end, (245, 245, 245), 12)  # White in BGR

    # Center dashed lane
    center_start = (vp_x, vp_y)
    center_end = (int(width * 0.50) - dx, height)

    steps = 14
    for i in range(steps):
        t1 = (i + (dash_offset % 1.0)) / steps
        t2 = (i + 0.50 + (dash_offset % 1.0)) / steps
        if t1 > 1.0 or t2 > 1.0:
            continue
        p1_x = int(center_start[0] + t1 * (center_end[0] - center_start[0]))
        p1_y = int(center_start[1] + t1 * (center_end[1] - center_start[1]))
        p2_x = int(center_start[0] + t2 * (center_end[0] - center_start[0]))
        p2_y = int(center_start[1] + t2 * (center_end[1] - center_start[1]))
        thickness = int(3 + t1 * 8)
        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), (250, 250, 250), thickness)

    # 4. Lead traffic ahead (distant car)
    if lead_car:
        car_cx = int(vp_x + (center_end[0] - vp_x) * 0.20)
        car_cy = int(vp_y + 36)
        cw, ch = 80, 52
        cv2.ellipse(img, (car_cx, car_cy + ch // 2 + 4), (cw // 2 + 6, 8), 0, 0, 360, (25, 25, 30), -1)
        cv2.rectangle(img, (car_cx - cw // 2, car_cy - ch // 2), (car_cx + cw // 2, car_cy + ch // 2), (180, 120, 40), -1)
        cv2.rectangle(img, (car_cx - cw // 2 + 10, car_cy - ch // 2 + 6), (car_cx + cw // 2 - 10, car_cy - 4), (50, 50, 60), -1)
        cv2.rectangle(img, (car_cx - cw // 2 + 4, car_cy + 6), (car_cx - cw // 2 + 20, car_cy + 18), (0, 0, 230), -1)
        cv2.rectangle(img, (car_cx + cw // 2 - 20, car_cy + 6), (car_cx + cw // 2 - 4, car_cy + 18), (0, 0, 230), -1)

    # 5. The White BMW Sports Sedan (CarMaker Chase-Camera perspective)
    if draw_white_sedan:
        car_center_x = int(width * 0.50)
        car_center_y = int(height * 0.72)
        draw_white_bmw_sedan(img, car_center_x, car_center_y, w=230, h=145)

    return img


def generate_sample_road_video(output_path, num_frames=120, fps=25):
    """
    Renders an ADAS simulation video with the white BMW sports sedan driving
    down the highway, showcasing active lane tracking and lateral departure dynamics.
    """
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        dash_offset = (frame_idx * 0.30)

        # Vehicle lateral movement profile
        if frame_idx < 25:
            lateral_offset = 0.0
            curve_offset = 0.0
        elif frame_idx < 55:
            progress = (frame_idx - 25) / 30.0
            lateral_offset = 135.0 * np.sin(progress * np.pi / 2)
            curve_offset = 15.0 * np.sin(progress * np.pi)
        elif frame_idx < 75:
            progress = (frame_idx - 55) / 20.0
            lateral_offset = 135.0 * np.cos(progress * np.pi / 2)
            curve_offset = -10.0 * np.sin(progress * np.pi)
        elif frame_idx < 100:
            progress = (frame_idx - 75) / 25.0
            lateral_offset = -130.0 * np.sin(progress * np.pi / 2)
            curve_offset = -20.0 * np.sin(progress * np.pi)
        else:
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
            draw_white_sedan=True
        )
        out.write(frame)

    out.release()
    print(f"[✓] Generated CarMaker ADAS video with White BMW Sedan at: {output_path}")

"""
Road Lane Line Detection Module.
Identifies left and right lane markings, extrapolates boundaries using linear polynomial fitting,
applies temporal exponential smoothing across video frames, and renders visual overlays.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from modules.image_processing import (
    to_grayscale,
    apply_gaussian_blur,
    detect_canny_edges,
    apply_roi_mask,
    detect_hough_lines,
    get_default_roi_vertices
)


class LaneDetector:
    """
    Stateful Lane Detector supporting temporal smoothing across continuous video frames.
    """

    def __init__(
        self,
        blur_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 30,
        min_line_len: int = 35,
        max_line_gap: int = 25,
        smoothing_factor: float = 0.85,
        min_slope: float = 0.40,
        max_slope: float = 2.50
    ):
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.smoothing_factor = smoothing_factor
        self.min_slope = min_slope
        self.max_slope = max_slope

        self.prev_left_line: Optional[Tuple[int, int, int, int]] = None
        self.prev_right_line: Optional[Tuple[int, int, int, int]] = None
        self.frame_count: int = 0

    def reset(self):
        """Resets temporal smoothing history."""
        self.prev_left_line = None
        self.prev_right_line = None
        self.frame_count = 0

    def separate_and_fit_lanes(
        self,
        lines: Optional[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """
        Classifies detected Hough line segments into left and right lane candidates
        based on slope and coordinate positioning, fitting linear polynomial regression.
        Left Lane: slope < 0 and x < 55% width
        Right Lane: slope > 0 and x > 45% width
        """
        height, width = image_shape[:2]
        left_x, left_y = [], []
        right_x, right_y = [], []

        if lines is not None and len(lines) > 0:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = line
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)

                # Filter out horizontal and near-vertical noise
                if abs(slope) < self.min_slope or abs(slope) > self.max_slope:
                    continue

                if slope < 0 and x1 < width * 0.55 and x2 < width * 0.55:
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                elif slope > 0 and x1 > width * 0.45 and x2 > width * 0.45:
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

        y_bottom = height
        y_top = int(height * 0.60)

        left_line = None
        if len(left_x) >= 2:
            try:
                left_fit = np.polyfit(left_y, left_x, deg=1)
                x_b = int(left_fit[0] * y_bottom + left_fit[1])
                x_t = int(left_fit[0] * y_top + left_fit[1])
                # Sanity check: left line bottom must stay on left half
                if 0 <= x_b <= width * 0.70 and 0 <= x_t <= width * 0.65:
                    left_line = (x_b, y_bottom, x_t, y_top)
            except (np.linalg.LinAlgError, ValueError):
                left_line = None

        right_line = None
        if len(right_x) >= 2:
            try:
                right_fit = np.polyfit(right_y, right_x, deg=1)
                x_b = int(right_fit[0] * y_bottom + right_fit[1])
                x_t = int(right_fit[0] * y_top + right_fit[1])
                # Sanity check: right line bottom must stay on right half
                if width * 0.30 <= x_b <= width and width * 0.35 <= x_t <= width:
                    right_line = (x_b, y_bottom, x_t, y_top)
            except (np.linalg.LinAlgError, ValueError):
                right_line = None

        # Cross check: left must not intersect or cross to the right of right lane
        if left_line and right_line and left_line[0] >= right_line[0]:
            # Degenerate intersection, drop the one with fewer points
            if len(left_x) >= len(right_x):
                right_line = None
            else:
                left_line = None

        # Temporal smoothing
        left_line = self._smooth_line(self.prev_left_line, left_line)
        right_line = self._smooth_line(self.prev_right_line, right_line)

        self.prev_left_line = left_line
        self.prev_right_line = right_line

        return left_line, right_line

    def _smooth_line(
        self,
        prev_line: Optional[Tuple[int, int, int, int]],
        curr_line: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Applies Exponential Moving Average (EMA) smoothing between consecutive frames."""
        if curr_line is None:
            return prev_line
        if prev_line is None:
            return curr_line

        alpha = self.smoothing_factor
        smoothed = tuple(
            int(alpha * p + (1.0 - alpha) * c)
            for p, c in zip(prev_line, curr_line)
        )
        return smoothed

    def calculate_lane_center(
        self,
        left_line: Optional[Tuple[int, int, int, int]],
        right_line: Optional[Tuple[int, int, int, int]],
        width: int
    ) -> Tuple[Optional[float], str]:
        """
        Calculates the horizontal center coordinate of the lane at the bottom of the image.
        Returns: (lane_center_px, detection_status)
        """
        if left_line is not None and right_line is not None:
            lane_center = (left_line[0] + right_line[0]) / 2.0
            status = "BOTH_LANES_DETECTED"
        elif left_line is not None:
            # Estimate lane center assuming standard lane width (~600px at bottom of 1280w)
            lane_center = left_line[0] + (width * 0.28)
            status = "LEFT_LANE_ONLY"
        elif right_line is not None:
            lane_center = right_line[0] - (width * 0.28)
            status = "RIGHT_LANE_ONLY"
        else:
            lane_center = None
            status = "NO_LANES_DETECTED"

        return lane_center, status

    def draw_lane_overlay(
        self,
        frame: np.ndarray,
        left_line: Optional[Tuple[int, int, int, int]],
        right_line: Optional[Tuple[int, int, int, int]],
        departure_side: Optional[str] = None,
        alpha: float = 0.40
    ) -> np.ndarray:
        """
        Renders transparent driveable corridor polygon and highlighted boundary lines.
        If a lane departure is active on a side (LEFT/RIGHT), that boundary turns VIBRANT RED,
        simulating real automotive ADAS cockpit visuals.
        """
        overlay = np.zeros_like(frame)

        # Dynamic color coding based on lane departure state
        fill_color = (0, 220, 80)     # Default emerald green corridor
        left_color = (0, 255, 255)    # Default yellow/cyan boundary
        right_color = (0, 255, 255)

        if departure_side == "RIGHT":
            right_color = (0, 0, 255)  # Bright Red on departed boundary
            fill_color = (0, 140, 230) # Amber warning corridor
        elif departure_side == "LEFT":
            left_color = (0, 0, 255)   # Bright Red on departed boundary
            fill_color = (0, 140, 230)

        # Driveable corridor polygon
        if left_line is not None and right_line is not None:
            poly_pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [poly_pts], fill_color)

        # Left Boundary line
        if left_line is not None:
            thick = 12 if departure_side == "LEFT" else 8
            cv2.line(
                overlay,
                (left_line[0], left_line[1]),
                (left_line[2], left_line[3]),
                left_color,
                thick
            )

        # Right Boundary line
        if right_line is not None:
            thick = 12 if departure_side == "RIGHT" else 8
            cv2.line(
                overlay,
                (right_line[0], right_line[1]),
                (right_line[2], right_line[3]),
                right_color,
                thick
            )

        return cv2.addWeighted(frame, 1.0, overlay, alpha, 0)

    def process_frame(
        self,
        frame: np.ndarray,
        draw_overlay: bool = True,
        departure_side: Optional[str] = None
    ) -> Dict:
        """
        Complete lane detection pipeline for a single frame.
        Returns dict containing processed frame, stages, lines, lane center and status.
        """
        self.frame_count += 1
        height, width = frame.shape[:2]

        # 1. Grayscale
        gray = to_grayscale(frame)

        # 2. Gaussian Blur
        blurred = apply_gaussian_blur(gray, kernel_size=self.blur_kernel)

        # 3. Canny Edge Detection
        edges = detect_canny_edges(
            blurred,
            low_threshold=self.canny_low,
            high_threshold=self.canny_high
        )

        # 4. Region of Interest Masking
        roi_vertices = get_default_roi_vertices(height, width)
        masked_edges = apply_roi_mask(edges, roi_vertices)

        # 5. Hough Line Transform
        lines = detect_hough_lines(
            masked_edges,
            threshold=self.hough_threshold,
            min_line_len=self.min_line_len,
            max_line_gap=self.max_line_gap
        )

        # 6. Separate and Fit Left / Right Lanes
        left_line, right_line = self.separate_and_fit_lanes(lines, frame.shape)

        # 7. Lane Center & Detection Status
        lane_center, status = self.calculate_lane_center(left_line, right_line, width)

        # 8. Render Visual Overlay
        if draw_overlay:
            annotated_frame = self.draw_lane_overlay(
                frame,
                left_line,
                right_line,
                departure_side=departure_side
            )
        else:
            annotated_frame = frame.copy()

        return {
            "annotated_frame": annotated_frame,
            "gray": gray,
            "edges": edges,
            "masked_edges": masked_edges,
            "lines": lines,
            "left_line": left_line,
            "right_line": right_line,
            "lane_center": lane_center,
            "status": status,
            "image_shape": (height, width)
        }


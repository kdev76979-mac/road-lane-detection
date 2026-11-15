"""
Lane Line Detection Module
Implements standard computer vision pipeline for detecting lane boundaries:
1. Grayscale conversion
2. Gaussian Blur (noise reduction)
3. Canny Edge Detection
4. Region of Interest (ROI) Masking
5. Hough Line Transform
6. Slope classification (left vs right) and line regression/averaging
7. Temporal smoothing for video feeds
"""

import cv2
import numpy as np


class LaneDetector:
    def __init__(self, blur_kernel=5, low_threshold=50, high_threshold=150, smoothing_factor=0.85):
        self.blur_kernel = blur_kernel
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.smoothing_factor = smoothing_factor

        # Moving averages for video stability
        self.prev_left_line = None
        self.prev_right_line = None

    def reset_smoothing(self):
        """Reset historical line tracking (useful when starting a new video)."""
        self.prev_left_line = None
        self.prev_right_line = None

    def to_grayscale(self, image):
        """Converts BGR image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, gray_image):
        """Applies Gaussian smoothing to suppress high-frequency noise."""
        return cv2.GaussianBlur(gray_image, (self.blur_kernel, self.blur_kernel), 0)

    def detect_edges(self, blurred_image):
        """Applies Canny edge detector to find strong image gradients."""
        return cv2.Canny(blurred_image, self.low_threshold, self.high_threshold)

    def region_of_interest(self, edges_image):
        """
        Applies an image mask so only the trapezoidal road region ahead of
        the vehicle is kept, discarding sky, side scenery, and bonnet.
        """
        height, width = edges_image.shape
        mask = np.zeros_like(edges_image)

        # Standard trapezoid polygon covering typical forward dashboard view
        vertices = np.array([[
            (int(width * 0.08), height),
            (int(width * 0.45), int(height * 0.60)),
            (int(width * 0.55), int(height * 0.60)),
            (int(width * 0.94), height)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(edges_image, mask)

    def detect_hough_lines(self, masked_edges):
        """
        Applies Probabilistic Hough Line Transform to detect line segments.
        """
        rho = 2                # Distance resolution in pixels
        theta = np.pi / 180    # Angular resolution in radians
        threshold = 30         # Minimum votes
        min_line_len = 35      # Minimum line length
        max_line_gap = 25      # Maximum allowed gap between segments

        lines = cv2.HoughLinesP(
            masked_edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
            lines=np.array([]),
            minLineLength=min_line_len,
            maxLineGap=max_line_gap
        )
        return lines

    def separate_and_fit_lanes(self, lines, image_shape):
        """
        Separates Hough line segments into Left and Right lane candidates based
        on slope, then fits linear polynomial (regression) to extrapolate lines.
        """
        height, width = image_shape[:2]
        left_x = []
        left_y = []
        right_x = []
        right_y = []

        if lines is not None and len(lines) > 0:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = line
                if x2 == x1:
                    continue  # Avoid vertical zero division
                slope = (y2 - y1) / (x2 - x1)

                # Filter out nearly horizontal noise lines
                if abs(slope) < 0.4 or abs(slope) > 2.5:
                    continue

                # In OpenCV image coordinates (origin top-left):
                # Left lane has negative slope and is on the left half
                if slope < 0 and x1 < width * 0.55 and x2 < width * 0.55:
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                # Right lane has positive slope and is on the right half
                elif slope > 0 and x1 > width * 0.45 and x2 > width * 0.45:
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

        y_bottom = height
        y_top = int(height * 0.62)

        # Fit left lane line
        left_line = None
        if len(left_x) >= 2:
            left_fit = np.polyfit(left_y, left_x, deg=1)
            x_bottom_left = int(left_fit[0] * y_bottom + left_fit[1])
            x_top_left = int(left_fit[0] * y_top + left_fit[1])
            left_line = (x_bottom_left, y_bottom, x_top_left, y_top)

        # Fit right lane line
        right_line = None
        if len(right_x) >= 2:
            right_fit = np.polyfit(right_y, right_x, deg=1)
            x_bottom_right = int(right_fit[0] * y_bottom + right_fit[1])
            x_top_right = int(right_fit[0] * y_top + right_fit[1])
            right_line = (x_bottom_right, y_bottom, x_top_right, y_top)

        # Temporal smoothing for video stability
        left_line = self._smooth_line(self.prev_left_line, left_line)
        right_line = self._smooth_line(self.prev_right_line, right_line)

        self.prev_left_line = left_line
        self.prev_right_line = right_line

        return left_line, right_line

    def _smooth_line(self, prev_line, curr_line):
        """Applies exponential moving average to avoid lane jitter in video."""
        if curr_line is None:
            return prev_line
        if prev_line is None:
            return curr_line

        smoothed = tuple(
            int(self.smoothing_factor * p + (1 - self.smoothing_factor) * c)
            for p, c in zip(prev_line, curr_line)
        )
        return smoothed

    def compute_vehicle_offset(self, left_line, right_line, image_width):
        """
        Calculates vehicle lateral offset from lane center in meters.
        Positive offset indicates vehicle is to the right of center;
        Negative offset indicates vehicle is to the left of center.
        """
        if left_line is None or right_line is None:
            return None, "Searching"

        lane_center = (left_line[0] + right_line[0]) / 2.0
        car_center = image_width / 2.0
        lane_width_px = max(1, right_line[0] - left_line[0])
        offset_m = ((car_center - lane_center) / lane_width_px) * 3.7

        if abs(offset_m) < 0.15:
            status = "Centered"
        elif offset_m > 0:
            status = "Drift Right"
        else:
            status = "Drift Left"

        return offset_m, status

    def draw_lanes(self, original_image, left_line, right_line, fill_lane=True, show_hud=True):
        """
        Draws glowing lane boundaries, driveable corridor, and telemetry HUD
        on top of the original image.
        """
        overlay = np.zeros_like(original_image)
        height, width = original_image.shape[:2]

        # Fill driveable lane corridor polygon
        if fill_lane and left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 200, 70))  # Green lane highlight

        # Draw left lane line (Yellow)
        if left_line is not None:
            cv2.line(overlay, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 255), 10)

        # Draw right lane line (Yellow)
        if right_line is not None:
            cv2.line(overlay, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 255), 10)

        # Blend with original image using alpha transparency
        result = cv2.addWeighted(original_image, 1.0, overlay, 0.45, 0)

        # Telemetry HUD
        if show_hud:
            offset_m, status = self.compute_vehicle_offset(left_line, right_line, width)
            hud_bg = result.copy()
            cv2.rectangle(hud_bg, (20, 20), (430, 110), (20, 20, 20), -1)
            cv2.rectangle(hud_bg, (20, 20), (430, 110), (0, 255, 255), 2)
            result = cv2.addWeighted(result, 0.3, hud_bg, 0.7, 0)

            status_color = (0, 255, 0) if status == "Centered" else ((0, 255, 255) if "Drift" in status else (180, 180, 180))
            cv2.putText(result, "AUTONOMOUS LANE TRACKING", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if offset_m is not None:
                cv2.putText(result, f"Vehicle Offset: {offset_m:+.2f}m [{status}]", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            else:
                cv2.putText(result, f"Vehicle Status: [{status}]", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            cv2.putText(result, "Lane Departure System: ON", (35, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        return result

    def process_frame(self, frame, fill_lane=True, show_hud=True):
        """
        End-to-end pipeline: Input frame -> Output frame with detected lanes.
        """
        gray = self.to_grayscale(frame)
        blur = self.apply_blur(gray)
        edges = self.detect_edges(blur)
        roi = self.region_of_interest(edges)
        lines = self.detect_hough_lines(roi)
        left_line, right_line = self.separate_and_fit_lanes(lines, frame.shape)
        annotated_frame = self.draw_lanes(frame, left_line, right_line, fill_lane=fill_lane, show_hud=show_hud)
        offset_m, status = self.compute_vehicle_offset(left_line, right_line, frame.shape[1])
        return annotated_frame, {
            "gray": gray,
            "edges": edges,
            "roi": roi,
            "left_line": left_line,
            "right_line": right_line,
            "offset_m": offset_m,
            "status": status
        }


"""
Road Curvature Estimation Module.
Estimates highway curvature direction (Straight, Left Curve, Right Curve)
based on observed perspective convergence and midpoint geometry.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class RoadCurvatureEstimator:
    """
    Estimates road direction and curvature trends from lane boundary geometry,
    with temporal exponential smoothing.
    """

    CURVE_STRAIGHT = "STRAIGHT"
    CURVE_LEFT = "CURVING_LEFT"
    CURVE_RIGHT = "CURVING_RIGHT"
    CURVE_UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        straight_threshold_px: float = 25.0,
        smoothing_factor: float = 0.80
    ):
        self.straight_threshold_px = straight_threshold_px
        self.smoothing_factor = smoothing_factor
        self.smoothed_curvature_offset: Optional[float] = None

    def reset(self):
        """Resets smoothed curvature memory."""
        self.smoothed_curvature_offset = None

    def estimate_curvature(
        self,
        left_line: Optional[Tuple[int, int, int, int]],
        right_line: Optional[Tuple[int, int, int, int]],
        frame_width: int
    ) -> Dict:
        """
        Calculates horizontal shift of the lane corridor vanishing point relative to
        bottom road midpoint.
        """
        if left_line is None and right_line is None:
            return {
                "direction": self.CURVE_UNKNOWN,
                "curvature_offset": None,
                "description": "Insufficient lane markings to estimate curvature."
            }

        # Calculate bottom and top midpoint
        if left_line is not None and right_line is not None:
            bottom_mid = (left_line[0] + right_line[0]) / 2.0
            top_mid = (left_line[2] + right_line[2]) / 2.0
            dy = max(1.0, float(left_line[1] - left_line[3]))
        elif left_line is not None:
            # Estimate with left line slope
            dx = left_line[2] - left_line[0]
            dy = max(1.0, float(left_line[1] - left_line[3]))
            bottom_mid = float(left_line[0])
            top_mid = float(left_line[2])
        else:
            dx = right_line[2] - right_line[0]
            dy = max(1.0, float(right_line[1] - right_line[3]))
            bottom_mid = float(right_line[0])
            top_mid = float(right_line[2])

        # Raw curvature offset: how much the upper trajectory veers relative to bottom center
        raw_offset = float(top_mid - (frame_width / 2.0))

        # Exponential smoothing
        if self.smoothed_curvature_offset is None:
            self.smoothed_curvature_offset = raw_offset
        else:
            self.smoothed_curvature_offset = (
                self.smoothing_factor * self.smoothed_curvature_offset +
                (1.0 - self.smoothing_factor) * raw_offset
            )

        offset = self.smoothed_curvature_offset

        if abs(offset) <= self.straight_threshold_px:
            direction = self.CURVE_STRAIGHT
            description = "Road geometry is straight ahead."
        elif offset > self.straight_threshold_px:
            direction = self.CURVE_RIGHT
            description = f"Gentle right bend detected (drift: {offset:+.1f}px)."
        else:
            direction = self.CURVE_LEFT
            description = f"Gentle left bend detected (drift: {offset:+.1f}px)."

        return {
            "direction": direction,
            "curvature_offset": round(offset, 2),
            "description": description
        }

"""
ADAS Lane Departure Warning (LDW) Module.
Monitors vehicle position relative to lane boundaries, calculates lateral deviation,
evaluates warning states (SAFE, CAUTION, WARNING, LANE DEPARTURE), and tracks departure events
with cooldown hysteresis to avoid duplicate counting.

DISCLAIMER: This is an algorithmic driver-assistance software prototype for research/portfolio
purposes, NOT a certified automotive safety system.
"""

from typing import Dict, List, Optional, Tuple


class LaneDepartureWarningSystem:
    """
    Evaluates vehicle lane keeping status, calculates lateral pixel and metric deviations,
    and maintains historical departure telemetry with cooldown filtering.
    """

    # ADAS Status Levels
    STATUS_SAFE = "SAFE"
    STATUS_CAUTION = "CAUTION"
    STATUS_WARNING = "WARNING"
    STATUS_DEPARTURE = "LANE DEPARTURE"

    def __init__(
        self,
        caution_threshold_px: float = 35.0,
        warning_threshold_px: float = 75.0,
        departure_threshold_px: float = 125.0,
        event_cooldown_frames: int = 30,
        standard_lane_width_meters: float = 3.7
    ):
        self.caution_threshold_px = caution_threshold_px
        self.warning_threshold_px = warning_threshold_px
        self.departure_threshold_px = departure_threshold_px
        self.event_cooldown_frames = event_cooldown_frames
        self.standard_lane_width_meters = standard_lane_width_meters

        # State tracking
        self.total_caution_events: int = 0
        self.total_warning_events: int = 0
        self.total_departure_events: int = 0
        self.frames_since_last_departure: int = event_cooldown_frames
        self.warning_history: List[Dict] = []

    def reset(self):
        """Resets all event counts and history."""
        self.total_caution_events = 0
        self.total_warning_events = 0
        self.total_departure_events = 0
        self.frames_since_last_departure = self.event_cooldown_frames
        self.warning_history.clear()

    def calculate_lateral_deviation(
        self,
        lane_center: Optional[float],
        width: int,
        vehicle_x_override: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates lateral deviation in pixels and percentage.
        Positive deviation: Vehicle is to the RIGHT of lane center.
        Negative deviation: Vehicle is to the LEFT of lane center.
        """
        if lane_center is None:
            return None, None

        vehicle_x = width / 2.0 if vehicle_x_override is None else vehicle_x_override
        deviation_px = float(vehicle_x - lane_center)

        # Percentage deviation relative to half-frame
        deviation_pct = (deviation_px / (width / 2.0)) * 100.0
        return deviation_px, deviation_pct

    def estimate_deviation_in_meters(
        self,
        deviation_px: Optional[float],
        left_line: Optional[Tuple[int, int, int, int]],
        right_line: Optional[Tuple[int, int, int, int]]
    ) -> Optional[float]:
        """
        Estimates deviation in real-world meters if both lane boundaries are detected.
        Scales deviation using the ratio of standard lane width (3.7m) to observed lane width (px).
        """
        if deviation_px is None or left_line is None or right_line is None:
            return None

        lane_width_px = right_line[0] - left_line[0]
        if lane_width_px <= 50:
            return None

        meters_per_pixel = self.standard_lane_width_meters / lane_width_px
        return float(deviation_px * meters_per_pixel)

    def evaluate(
        self,
        lane_center: Optional[float],
        frame_width: int,
        left_line: Optional[Tuple[int, int, int, int]] = None,
        right_line: Optional[Tuple[int, int, int, int]] = None,
        frame_idx: int = 0
    ) -> Dict:
        """
        Evaluates current frame and generates ADAS warning level and event metrics.
        """
        self.frames_since_last_departure += 1

        deviation_px, deviation_pct = self.calculate_lateral_deviation(lane_center, frame_width)
        deviation_m = self.estimate_deviation_in_meters(deviation_px, left_line, right_line)

        if deviation_px is None:
            warning_level = self.STATUS_CAUTION
            position_label = "SEARCHING"
            message = "Lane markers not clearly detected. Hold steering."
            is_departure_event = False
        else:
            abs_dev = abs(deviation_px)

            # Position direction
            if abs_dev < self.caution_threshold_px:
                position_label = "CENTERED"
                warning_level = self.STATUS_SAFE
                message = "Vehicle within lane corridor."
                is_departure_event = False
            elif deviation_px > 0:
                position_label = "RIGHT_OFFSET"
                if abs_dev >= self.departure_threshold_px:
                    warning_level = self.STATUS_DEPARTURE
                    message = "LANE DEPARTURE WARNING: Drifting Right!"
                    is_departure_event = True
                elif abs_dev >= self.warning_threshold_px:
                    warning_level = self.STATUS_WARNING
                    message = "WARNING: Steer Left to center."
                    is_departure_event = False
                else:
                    warning_level = self.STATUS_CAUTION
                    message = "CAUTION: Minor right deviation."
                    is_departure_event = False
            else:
                position_label = "LEFT_OFFSET"
                if abs_dev >= self.departure_threshold_px:
                    warning_level = self.STATUS_DEPARTURE
                    message = "LANE DEPARTURE WARNING: Drifting Left!"
                    is_departure_event = True
                elif abs_dev >= self.warning_threshold_px:
                    warning_level = self.STATUS_WARNING
                    message = "WARNING: Steer Right to center."
                    is_departure_event = False
                else:
                    warning_level = self.STATUS_CAUTION
                    message = "CAUTION: Minor left deviation."
                    is_departure_event = False

        # Apply event cooldown to prevent repeated counting on every contiguous frame
        if is_departure_event and (self.frames_since_last_departure >= self.event_cooldown_frames):
            self.total_departure_events += 1
            self.frames_since_last_departure = 0
            self.warning_history.append({
                "frame": frame_idx,
                "level": warning_level,
                "deviation_px": deviation_px,
                "deviation_m": deviation_m,
                "message": message
            })
        elif warning_level == self.STATUS_WARNING and (self.frames_since_last_departure % 20 == 0):
            self.total_warning_events += 1

        return {
            "warning_level": warning_level,
            "position_label": position_label,
            "deviation_px": deviation_px,
            "deviation_pct": deviation_pct,
            "deviation_m": deviation_m,
            "message": message,
            "total_departure_events": self.total_departure_events,
            "total_warning_events": self.total_warning_events
        }

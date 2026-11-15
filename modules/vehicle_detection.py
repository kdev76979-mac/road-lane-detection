"""
YOLO Vehicle Detection Module.
Detects nearby road vehicles (cars, motorcycles, buses, trucks) using lightweight Ultralytics YOLO.
Provides bounding box annotations, confidence scores, and vehicle counts.
Designed for CPU and Apple Silicon (MPS) inference with graceful offline fallbacks.

DISCLAIMER: This module performs 2D object detection for ADAS situational awareness.
It does NOT perform certified forward collision warning, distance estimation, or automatic braking.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class VehicleDetector:
    """
    Lightweight YOLO vehicle detection wrapper with lazy loading and error resilience.
    """

    # COCO Class IDs for road vehicles
    VEHICLE_CLASS_IDS = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    # Distinct BGR colors for vehicle categories
    CLASS_COLORS = {
        "car": (255, 140, 0),        # Deep Sky Blue (in BGR)
        "motorcycle": (0, 215, 255), # Amber
        "bus": (205, 0, 205),        # Purple
        "truck": (0, 165, 255)       # Orange
    }

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device

        self._model = None
        self._is_loaded = False
        self._load_error: Optional[str] = None

    def load_model(self) -> bool:
        """Lazily imports ultralytics and loads the YOLO model."""
        if self._is_loaded:
            return True

        try:
            from ultralytics import YOLO
            import torch

            # Determine best device
            if self.device == "auto":
                if torch.backends.mps.is_available():
                    chosen_device = "mps"
                elif torch.cuda.is_available():
                    chosen_device = "cuda"
                else:
                    chosen_device = "cpu"
            else:
                chosen_device = self.device

            self._model = YOLO(self.model_name)
            self._model.to(chosen_device)
            self._is_loaded = True
            self._load_error = None
            return True
        except Exception as e:
            self._is_loaded = False
            self._load_error = str(e)
            return False

    def is_available(self) -> bool:
        """Returns True if YOLO model is ready for inference."""
        return self._is_loaded

    def get_status_message(self) -> str:
        """Returns current status or diagnostic error."""
        if self._is_loaded:
            return f"YOLO active ({self.model_name} on {self.device})"
        if self._load_error:
            return f"YOLO unavailable: {self._load_error}"
        return "YOLO not loaded"

    def detect_vehicles(
        self,
        frame: np.ndarray,
        draw_boxes: bool = True
    ) -> Tuple[np.ndarray, List[Dict], int]:
        """
        Runs object detection on frame and filters for road vehicle classes.
        Returns: (annotated_frame, list_of_detections, vehicle_count)
        """
        if frame is None or frame.size == 0:
            return frame, [], 0

        annotated = frame.copy()
        detections: List[Dict] = []

        if not self._is_loaded:
            # Attempt lazy load
            if not self.load_model():
                return annotated, detections, 0

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device if self.device != "auto" else None
            )

            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id in self.VEHICLE_CLASS_IDS:
                        label = self.VEHICLE_CLASS_IDS[cls_id]
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy

                        detections.append({
                            "class": label,
                            "confidence": round(conf, 2),
                            "bbox": (int(x1), int(y1), int(x2), int(y2))
                        })

                        if draw_boxes:
                            color = self.CLASS_COLORS.get(label, (0, 255, 0))
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                            badge = f"{label.upper()} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(
                                annotated,
                                (x1, max(0, y1 - th - 6)),
                                (x1 + tw + 6, y1),
                                color,
                                -1
                            )
                            cv2.putText(
                                annotated,
                                badge,
                                (x1 + 3, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )

        except Exception as e:
            # Safe recovery from frame processing errors
            self._load_error = f"Inference warning: {e}"

        return annotated, detections, len(detections)


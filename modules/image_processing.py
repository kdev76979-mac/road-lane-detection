"""
OpenCV Image Processing Primitives for Road Lane Detection.
Provides fundamental transformations: Grayscale, Gaussian Blur, Canny Edge Detection,
Trapezoidal Region of Interest (ROI) Masking, and Probabilistic Hough Line Transform.
"""

import cv2
import numpy as np


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Converts a BGR image frame to single-channel grayscale."""
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty or invalid.")
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Suppresses high-frequency asphalt and texture noise using Gaussian smoothing.
    Kernel size must be an odd positive integer.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)


def detect_canny_edges(
    blurred: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """
    Computes dual-threshold intensity gradients using Canny Edge Detector.
    Pixels with intensity gradient above high_threshold are marked as edges;
    pixels below low_threshold are discarded.
    """
    return cv2.Canny(blurred, low_threshold, high_threshold)


def get_default_roi_vertices(height: int, width: int) -> np.ndarray:
    """
    Returns standard trapezoidal vertices focusing on the forward highway drive corridor.
    Discards the sky (top 40%), vehicle hood, and lateral non-road surroundings.
    """
    return np.array([[
        (int(width * 0.08), height),
        (int(width * 0.44), int(height * 0.60)),
        (int(width * 0.56), int(height * 0.60)),
        (int(width * 0.94), height)
    ]], dtype=np.int32)


def apply_roi_mask(edges: np.ndarray, vertices: np.ndarray = None) -> np.ndarray:
    """
    Applies a polygonal mask to isolate edges only within the Region of Interest.
    """
    if edges is None or edges.size == 0:
        raise ValueError("Input edge image is empty.")
    
    height, width = edges.shape[:2]
    if vertices is None:
        vertices = get_default_roi_vertices(height, width)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def detect_hough_lines(
    masked_edges: np.ndarray,
    rho: int = 2,
    theta: float = np.pi / 180,
    threshold: int = 30,
    min_line_len: int = 35,
    max_line_gap: int = 25
) -> np.ndarray:
    """
    Executes Probabilistic Hough Line Transform to detect line segments.
    Returns array of shape (N, 4) or None if no lines detected.
    """
    if masked_edges is None or masked_edges.size == 0:
        return None

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        lines=np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )

    if lines is not None and len(lines) > 0:
        return lines.reshape(-1, 4)
    return None

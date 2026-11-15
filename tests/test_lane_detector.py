"""
Unit tests for Road Lane Detection system.
Covers core computer vision pipeline stages and utilities.
"""

import unittest
import numpy as np
import cv2

from src.lane_detector import LaneDetector
from src.utils import (
    create_pipeline_dashboard,
    generate_sample_road_image
)


class TestLaneDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LaneDetector()
        self.test_image = generate_sample_road_image(width=640, height=360)

    def test_grayscale_conversion(self):
        gray = self.detector.to_grayscale(self.test_image)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (360, 640))
        self.assertEqual(gray.dtype, np.uint8)

    def test_apply_blur(self):
        gray = self.detector.to_grayscale(self.test_image)
        blurred = self.detector.apply_blur(gray)
        self.assertEqual(blurred.shape, gray.shape)
        self.assertEqual(blurred.dtype, np.uint8)

    def test_detect_edges(self):
        gray = self.detector.to_grayscale(self.test_image)
        blurred = self.detector.apply_blur(gray)
        edges = self.detector.detect_edges(blurred)
        self.assertEqual(edges.shape, gray.shape)
        unique_vals = set(np.unique(edges))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_region_of_interest(self):
        gray = self.detector.to_grayscale(self.test_image)
        edges = self.detector.detect_edges(gray)
        roi = self.detector.region_of_interest(edges)
        self.assertEqual(roi.shape, edges.shape)
        # Top 40% of ROI should be completely masked (all zeros)
        top_cut = int(roi.shape[0] * 0.40)
        self.assertEqual(np.count_nonzero(roi[:top_cut, :]), 0)

    def test_detect_hough_lines(self):
        gray = self.detector.to_grayscale(self.test_image)
        edges = self.detector.detect_edges(gray)
        roi = self.detector.region_of_interest(edges)
        lines = self.detector.detect_hough_lines(roi)
        self.assertIsNotNone(lines)
        self.assertGreater(len(lines), 0)

    def test_separate_and_fit_lanes(self):
        gray = self.detector.to_grayscale(self.test_image)
        edges = self.detector.detect_edges(gray)
        roi = self.detector.region_of_interest(edges)
        lines = self.detector.detect_hough_lines(roi)
        left_line, right_line = self.detector.separate_and_fit_lanes(lines, self.test_image.shape)
        self.assertIsNotNone(left_line)
        self.assertIsNotNone(right_line)
        self.assertEqual(len(left_line), 4)
        self.assertEqual(len(right_line), 4)
        # Left line bottom X should be to the left of Right line bottom X
        self.assertLess(left_line[0], right_line[0])

    def test_temporal_smoothing(self):
        line1 = (100, 360, 250, 200)
        line2 = (120, 360, 270, 200)
        detector = LaneDetector(smoothing_factor=0.5)
        smoothed = detector._smooth_line(line1, line2)
        expected = (110, 360, 260, 200)
        self.assertEqual(smoothed, expected)

    def test_process_frame_end_to_end(self):
        annotated, stages = self.detector.process_frame(self.test_image, fill_lane=True)
        self.assertEqual(annotated.shape, self.test_image.shape)
        self.assertIn("edges", stages)
        self.assertIn("roi", stages)
        self.assertIn("left_line", stages)
        self.assertIn("right_line", stages)

    def test_empty_or_black_frame(self):
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        detector = LaneDetector()
        annotated, stages = detector.process_frame(blank)
        self.assertEqual(annotated.shape, blank.shape)
        self.assertIsNone(stages["left_line"])
        self.assertIsNone(stages["right_line"])

    def test_dashboard_creation(self):
        annotated, stages = self.detector.process_frame(self.test_image)
        dashboard = create_pipeline_dashboard(
            self.test_image,
            stages["edges"],
            stages["roi"],
            annotated
        )
        self.assertEqual(dashboard.shape, self.test_image.shape)


if __name__ == "__main__":
    unittest.main()

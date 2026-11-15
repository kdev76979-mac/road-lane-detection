"""
Road Lane Line Detection - Main Execution Script

Usage:
  1. Generate sample test images & video:
     python main.py --generate-samples

  2. Process a road image:
     python main.py --image data/test_images/sample_road.jpg

  3. Process a video:
     python main.py --video data/test_videos/sample_highway.mp4

  4. Run on default sample and display 2x2 debug dashboard:
     python main.py --demo
"""

import argparse
import os
import sys
import cv2

from src.lane_detector import LaneDetector
from src.utils import (
    create_pipeline_dashboard,
    generate_sample_road_image,
    generate_sample_road_video
)


def ensure_samples_exist():
    """Generates starter sample image and video if test directories are empty."""
    img_dir = os.path.join("data", "test_images")
    vid_dir = os.path.join("data", "test_videos")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    sample_img_path = os.path.join(img_dir, "sample_road.jpg")
    sample_vid_path = os.path.join(vid_dir, "sample_highway.mp4")

    if not os.path.exists(sample_img_path):
        print("[+] Creating sample road image...")
        img = generate_sample_road_image()
        cv2.imwrite(sample_img_path, img)
        print(f"[✓] Created: {sample_img_path}")

    if not os.path.exists(sample_vid_path):
        print("[+] Creating sample road video (90 frames)...")
        generate_sample_road_video(sample_vid_path, num_frames=90, fps=25)

    return sample_img_path, sample_vid_path


def process_image(image_path, output_path=None, show_dashboard=True):
    """Detects lane lines on an input image and saves/displays the result."""
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[!] Error: Could not read image at {image_path}")
        return

    detector = LaneDetector()
    annotated, stages = detector.process_frame(frame, fill_lane=True)
    dashboard = create_pipeline_dashboard(frame, stages["edges"], stages["roi"], annotated)

    if output_path is None:
        base_name = os.path.basename(image_path)
        output_path = os.path.join("output", f"detected_{base_name}")

    cv2.imwrite(output_path, annotated)
    dashboard_path = os.path.join("output", f"dashboard_{os.path.basename(image_path)}")
    cv2.imwrite(dashboard_path, dashboard)

    print(f"[✓] Lane detection completed!")
    print(f"    - Output Image:    {output_path}")
    print(f"    - 4-Stage Matrix:  {dashboard_path}")


def process_video(video_path, output_path=None, display=False):
    """Processes video frame-by-frame with temporal lane line smoothing."""
    if not os.path.exists(video_path):
        print(f"[!] Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        base_name = os.path.basename(video_path)
        output_path = os.path.join("output", f"detected_{base_name}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = LaneDetector(smoothing_factor=0.88)
    frame_idx = 0

    print(f"[+] Processing video: {video_path} ({total_frames} frames @ {fps:.1f} fps)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = detector.process_frame(frame, fill_lane=True)
        out.write(annotated)
        frame_idx += 1

        if frame_idx % 25 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"    Processing: frame {frame_idx}/{total_frames} ({pct:.1f}%)")

        if display:
            cv2.imshow("Lane Detection Stream", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()

    print(f"[✓] Video saved successfully to: {output_path}")


def process_webcam(cam_index=0, show_hud=True):
    """Processes live camera/webcam feed for real-time lane tracking."""
    # Use native AVFoundation backend on macOS for better compatibility
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
    else:
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"\n[!] Camera Error: Could not access camera device (index {cam_index}).")
        print("    macOS Camera Permission Guide:")
        print("    1. Open: System Settings -> Privacy & Security -> Camera")
        print("    2. Toggle ON permission for your Terminal app (Terminal, iTerm, or VS Code).")
        print("    3. Restart your terminal and re-run: python main.py --webcam 0\n")
        return

    detector = LaneDetector(smoothing_factor=0.85)
    print(f"[+] Starting live camera feed (device {cam_index}). Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = detector.process_frame(frame, fill_lane=True, show_hud=show_hud)
        cv2.imshow("Lane Detection Stream (Press 'q' to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Camera feed terminated.")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Road Lane Line Detection")
    parser.add_argument("--image", type=str, help="Path to input road image")
    parser.add_argument("--video", type=str, help="Path to input dashcam video")
    parser.add_argument("--webcam", type=int, nargs="?", const=0, default=None, help="Process live camera/webcam feed (default index: 0)")
    parser.add_argument("--no-hud", action="store_true", help="Disable telemetry HUD overlay")
    parser.add_argument("--generate-samples", action="store_true", help="Generate synthetic road test data")
    parser.add_argument("--demo", action="store_true", help="Run quick demo on built-in sample road")
    parser.add_argument("--output", type=str, default=None, help="Output destination file path")
    args = parser.parse_args()

    show_hud = not args.no_hud

    if args.generate_samples:
        ensure_samples_exist()
        return

    if args.webcam is not None:
        process_webcam(cam_index=args.webcam, show_hud=show_hud)
        return

    if args.image:
        process_image(args.image, output_path=args.output)
        return

    if args.video:
        process_video(args.video, output_path=args.output)
        return

    # Default / Demo mode
    sample_img, sample_vid = ensure_samples_exist()
    print("\n--- Running Road Lane Line Detection Demo ---")
    process_image(sample_img)
    process_video(sample_vid)
    print("\n[✓] Demo complete! Check the 'output/' folder for all results.")


if __name__ == "__main__":
    main()


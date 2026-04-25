"""
UTD Parking Spot Occupancy Detector
CS 6384 - Group 34

Pipeline:
    Source frame -> YOLOv8 (cars/trucks/motorcycles/buses)
                 -> for each static spot ROI, IoU(spot, vehicle) > THRESHOLD
                 -> render green = empty, red = occupied + write demo_video.mp4

The --source argument accepts THREE kinds of input:
    1. a video file path        e.g.  ../data/videos/utd_parking_sample.mp4
    2. an integer webcam id     e.g.  0
    3. an IP / RTSP / HTTP URL  e.g.  rtsp://user:pass@192.168.1.10:554/stream
                                      http://192.168.1.10:8080/video

Usage examples:
    # 1. Recorded UTD video
    python main.py --source ../data/videos/utd_parking_sample.mp4 \
                   --rois   ../data/rois.json \
                   --out    ../results/demo_video.mp4

    # 2. Public sample (already shipped, 69 spots)
    python main.py --source ../data/videos/carPark.mp4 \
                   --rois   ../data/rois_carpark.json \
                   --out    ../results/carpark_demo.mp4 --loop --max-frames 600

    # 3. Live webcam (press 'q' to stop)
    python main.py --source 0 --rois ../data/rois.json \
                   --out ../results/live.mp4

    # 4. Live IP camera (any phone with the "IP Webcam" Android app works)
    python main.py --source http://192.168.1.42:8080/video \
                   --rois   ../data/rois.json \
                   --out    ../results/live.mp4
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# COCO ids for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


def calculate_iou(box_a, box_b):
    """IoU between two [x1,y1,x2,y2] axis-aligned boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return float(inter) / union if union > 0 else 0.0


def load_rois(path):
    with open(path) as f:
        data = json.load(f)
    return [list(map(int, s)) for s in data["spots"]]


def open_source(src):
    """Open a cv2.VideoCapture from a path / int / URL.

    Returns (cap, is_stream) where is_stream=True for live sources
    (webcam, RTSP, HTTP) and False for video files.
    """
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW if os.name == "nt"
                                                       else cv2.CAP_ANY)
        return cap, True
    if src.startswith(("rtsp://", "http://", "https://")):
        return cv2.VideoCapture(src), True
    if not os.path.exists(src):
        sys.exit(f"[ERROR] Source not found: {src}")
    return cv2.VideoCapture(src), False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True,
                   help="Video file path, integer webcam id (e.g. 0), or "
                        "rtsp/http URL.")
    p.add_argument("--rois",    required=True)
    p.add_argument("--out",     default="../results/demo_video.mp4")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.5,
                   help="Occupancy threshold (IoU between vehicle and spot).")
    p.add_argument("--imgsz",   type=int, default=640,
                   help="YOLO inference image size (320/480/640/960/1280).")
    p.add_argument("--no-show", action="store_true",
                   help="Headless mode (no cv2.imshow window).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = unlimited).")
    p.add_argument("--loop", action="store_true",
                   help="For file sources, loop back to frame 0 at EOF.")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Run YOLO every Nth frame; reuse the last result on "
                        "the others. Use 2-3 to boost FPS on slow CPUs.")
    return p.parse_args()


def main():
    args = parse_args()
    spots = load_rois(args.rois)
    print(f"[INFO] Loaded {len(spots)} parking spots from {args.rois}")

    model = YOLO(args.weights)
    print(f"[INFO] Loaded YOLO weights: {args.weights}")

    cap, is_stream = open_source(args.source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open source: {args.source}")
    print(f"[INFO] Source opened (live stream = {is_stream})")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        ok, frame = cap.read()
        if not ok:
            sys.exit("[ERROR] Could not read first frame to determine size.")
        height, width = frame.shape[:2]
        # rewind
        if not is_stream:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc,
                             min(max(fps_in, 5.0), 60.0), (width, height))

    # Per-frame log of (frame_idx, [0/1 for each spot]) for evaluation
    log_path = os.path.splitext(args.out)[0] + "_predictions.json"
    predictions = []

    inf_times = deque(maxlen=60)   # last 60 inference latencies for live FPS
    last_vehicles = []
    last_status = [0] * len(spots)
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if not is_stream and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            frame_idx += 1

            # Run YOLO every Nth frame to boost FPS
            if frame_idx % args.frame_stride == 1 or args.frame_stride == 1:
                t0 = time.time()
                results = model(frame, classes=VEHICLE_CLASS_IDS,
                                conf=args.conf, imgsz=args.imgsz,
                                verbose=False)
                inf_times.append(time.time() - t0)

                vehicles = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes.xyxy.cpu().numpy():
                        vehicles.append(box.tolist())
                last_vehicles = vehicles

                spot_status = []
                for spot in spots:
                    occupied = any(calculate_iou(spot, v) > args.iou
                                   for v in vehicles)
                    spot_status.append(1 if occupied else 0)
                last_status = spot_status
            else:
                vehicles = last_vehicles
                spot_status = last_status

            predictions.append({"frame": frame_idx, "spots": spot_status})

            for v in vehicles:
                x1, y1, x2, y2 = map(int, v)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

            occupied = 0
            for i, (spot, occ) in enumerate(zip(spots, spot_status)):
                color = (0, 0, 255) if occ else (0, 255, 0)
                cv2.rectangle(frame, (spot[0], spot[1]),
                              (spot[2], spot[3]), color, 2)
                cv2.putText(frame, f"#{i + 1}",
                            (spot[0] + 4, spot[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                occupied += occ

            live_fps = (len(inf_times) / sum(inf_times)
                        if sum(inf_times) > 0 else 0.0)
            cv2.rectangle(frame, (10, 10), (430, 95), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {live_fps:5.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Occupied: {occupied}/{len(spots)}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            writer.write(frame)
            if not args.no_show:
                cv2.imshow("UTD Parking Spot Occupancy Detector", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if args.max_frames and frame_idx >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    avg_inf_fps = (len(inf_times) / sum(inf_times)
                   if sum(inf_times) > 0 else 0.0)

    with open(log_path, "w") as f:
        json.dump({"source": args.source,
                   "weights": args.weights,
                   "iou_threshold": args.iou,
                   "conf_threshold": args.conf,
                   "imgsz": args.imgsz,
                   "frame_stride": args.frame_stride,
                   "num_spots": len(spots),
                   "frames": frame_idx,
                   "wall_fps": avg_fps,
                   "inference_fps": avg_inf_fps,
                   "predictions": predictions}, f, indent=2)

    print(f"[OK] Demo video : {args.out}")
    print(f"[OK] Predictions: {log_path}")
    print(f"[OK] Frames     : {frame_idx}")
    print(f"[OK] Wall FPS   : {avg_fps:.2f}")
    print(f"[OK] Infer FPS  : {avg_inf_fps:.2f}  (model-only)")


if __name__ == "__main__":
    main()

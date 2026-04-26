"""Auto-extract parking-spot ROIs from the first frame of a video.

Runs YOLOv8n on frame 1, filters to vehicle classes, and writes the resulting
bounding boxes as `rois.json` with the schema main.py expects:
  {"image_size": [W, H], "spots": [[x1, y1, x2, y2], ...]}

This is a fast alternative to roi_picker.py when every parking spot you care
about happens to have a vehicle in it at the start of the recording.

Usage:
    python auto_rois.py --video <path> --out <rois.json> [--conf 0.25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--frame", type=int, default=0,
                    help="frame index to extract ROIs from (default 0)")
    ap.add_argument("--preview", type=Path, default=None,
                    help="optional path to save an annotated preview JPG")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame}")

    h, w = frame.shape[:2]
    print(f"[auto_rois] frame {args.frame}: {w}x{h}")

    model = YOLO(args.weights)
    res = model.predict(frame, classes=VEHICLE_CLASSES,
                        conf=args.conf, iou=args.iou, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    print(f"[auto_rois] detected {len(boxes)} vehicles -> {len(boxes)} ROIs")

    spots = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"image_size": [w, h], "spots": spots}, indent=2))
    print(f"[auto_rois] wrote {args.out}")

    if args.preview is not None:
        preview = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(spots):
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, str(i + 1), (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), preview)
        print(f"[auto_rois] wrote preview {args.preview}")


if __name__ == "__main__":
    main()

"""One-shot inspector for candidate UTD parking videos.

For each input video, prints metadata (resolution, fps, duration), saves three
sample frames (start / middle / end) as JPGs, runs YOLOv8n on the middle frame
filtered to vehicle classes (car / motorcycle / bus / truck), and writes an
annotated copy of the middle frame so a human (or vision model) can see what
YOLO would label.

Usage:
    python inspect_video.py <video> [<video> ...] --out ../data/frames
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def grab(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def annotate(frame, boxes_xyxy, classes, confs):
    out = frame.copy()
    for (x1, y1, x2, y2), c, p in zip(boxes_xyxy, classes, confs):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{VEHICLE_CLASSES.get(int(c), c)} {p:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def inspect(video_path: Path, out_dir: Path, model: YOLO) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "error": "cv2 could not open"}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (n / fps) if fps > 0 else 0.0

    stem = video_path.stem
    saved = []
    middle_detections = None

    for label, idx in (("start", 0), ("mid", n // 2), ("end", max(0, n - 5))):
        frame = grab(cap, idx)
        if frame is None:
            continue
        out_path = out_dir / f"{stem}_{label}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(str(out_path))

        if label == "mid":
            t0 = time.time()
            res = model.predict(frame, classes=list(VEHICLE_CLASSES.keys()),
                                conf=0.25, iou=0.5, verbose=False)[0]
            infer_ms = (time.time() - t0) * 1000.0
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
            cls = res.boxes.cls.cpu().numpy() if res.boxes is not None else []
            conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else []

            ann = annotate(frame, boxes, cls, conf)
            ann_path = out_dir / f"{stem}_mid_annotated.jpg"
            cv2.imwrite(str(ann_path), ann)
            saved.append(str(ann_path))

            counts = {VEHICLE_CLASSES[int(c)]: 0 for c in cls if int(c) in VEHICLE_CLASSES}
            for c in cls:
                counts[VEHICLE_CLASSES[int(c)]] = counts.get(VEHICLE_CLASSES[int(c)], 0) + 1

            middle_detections = {
                "total_vehicles": int(len(cls)),
                "by_class": counts,
                "infer_ms": round(infer_ms, 1),
            }

    cap.release()
    return {
        "video": str(video_path),
        "size_mb": round(video_path.stat().st_size / 1e6, 2),
        "resolution": f"{w}x{h}",
        "fps": round(fps, 2),
        "frames": n,
        "duration_sec": round(duration, 2),
        "saved_frames": saved,
        "middle_detections": middle_detections,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../data/frames"))
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[inspect] loading model {args.weights} ...")
    model = YOLO(args.weights)
    print("[inspect] model ready\n")

    report = []
    for v in args.videos:
        if not v.exists():
            print(f"[inspect] SKIP {v} (not found)")
            continue
        print(f"[inspect] {v.name}")
        info = inspect(v, args.out, model)
        report.append(info)
        print(json.dumps(info, indent=2))
        print()

    out_json = args.out / "inspection_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[inspect] wrote {out_json}")


if __name__ == "__main__":
    sys.exit(main())

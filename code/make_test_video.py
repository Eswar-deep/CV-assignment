"""
Synthesize a controlled test video + ground-truth from a single parking-lot
still image (e.g. data/frames/sample_lot.jpg).

Why?
    * Lets us run a real end-to-end test of YOLOv8 + IoU + evaluation pipeline
      with a known ground truth, before any UTD video is recorded.
    * The carPark.mp4 sample we also ship is a near-top-down view that COCO
      pre-trained YOLO does not detect; this synthetic test uses a realistic
      ~45 deg angle where YOLO works perfectly (mirroring what a UTD shoot
      from a 3rd/4th floor of PS3/PS4 will look like).

How it works:
    1. Load a single still image of a parking lot.
    2. Run YOLOv8 once to find every parked vehicle.
    3. Treat each detected vehicle as both:
         - a registered parking-spot ROI (slightly padded), and
         - the "ground-truth occupied" state for that spot in the base frame.
    4. Build N frames; in each, mask out a chosen subset of vehicles by
       overlaying their box with the median asphalt color, simulating
       departures. The masked spots are GT=Empty for that frame.
    5. Write the synthesized video AND a JSON ground-truth file.

Outputs (default paths):
    data/videos/synthetic_lot.mp4
    data/rois_synthetic.json
    data/ground_truth/gt_synthetic.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


def expand_box(box, pad_pct, w, h):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * pad_pct, bh * pad_pct
    return [max(0, int(x1 - dx)),
            max(0, int(y1 - dy)),
            min(w - 1, int(x2 + dx)),
            min(h - 1, int(y2 + dy))]


def median_asphalt(img):
    """Return BGR median of the lower-third of the frame as a proxy for
    ground/asphalt color, used to mask out departed vehicles."""
    h = img.shape[0]
    strip = img[int(h * 0.6):, :, :].reshape(-1, 3)
    return tuple(int(c) for c in np.median(strip, axis=0))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  default="../data/frames/sample_lot.jpg")
    p.add_argument("--out-video", default="../data/videos/synthetic_lot.mp4")
    p.add_argument("--out-rois",  default="../data/rois_synthetic.json")
    p.add_argument("--out-gt",    default="../data/ground_truth/gt_synthetic.json")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--frames",  type=int, default=90,
                   help="Total frames to synthesize.")
    p.add_argument("--fps",     type=float, default=15.0)
    p.add_argument("--pad",     type=float, default=0.05,
                   help="Fractional padding around each detected vehicle "
                        "before treating it as a parking-spot ROI.")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not os.path.exists(args.image):
        sys.exit(f"[ERROR] Image not found: {args.image}")
    base = cv2.imread(args.image)
    if base is None:
        sys.exit("[ERROR] Failed to read image.")
    h, w = base.shape[:2]

    model = YOLO(args.weights)
    print(f"[INFO] Loaded YOLO weights: {args.weights}")
    res = model(base, conf=args.conf, classes=[2, 3, 5, 7], verbose=False)
    boxes = res[0].boxes.xyxy.cpu().numpy().tolist()
    if not boxes:
        sys.exit("[ERROR] No vehicles detected in the still. "
                 "Pick a clearer photo.")
    print(f"[INFO] Detected {len(boxes)} vehicles in the base still.")

    # ROIs = padded vehicle boxes; sort top-to-bottom for stable numbering.
    boxes.sort(key=lambda b: (round(b[1] / 30), b[0]))
    rois = [expand_box(b, args.pad, w, h) for b in boxes]
    n = len(rois)

    asphalt = median_asphalt(base)
    print(f"[INFO] Asphalt color (BGR): {asphalt}")

    # Build a per-frame "mask plan": which spot-indices are EMPTY this frame.
    # Schedule: rolling subset of cars depart and return so that across the
    # video each spot is empty for ~30% of frames.
    plan = []
    empty_count_schedule = np.tile(
        np.array([0, 0, 1, 2, 3, 3, 4, 3, 2, 1]), args.frames // 10 + 1
    )[: args.frames]
    for f in range(args.frames):
        k = int(empty_count_schedule[f])
        empty_idx = sorted(rng.choice(n, size=k, replace=False).tolist()) if k else []
        plan.append(empty_idx)

    # Write video + GT
    os.makedirs(os.path.dirname(os.path.abspath(args.out_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_rois)),  exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_gt)),    exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, (w, h))

    gt = {"video": args.out_video, "rois_file": args.out_rois, "labels": []}
    for f, empty_idx in enumerate(plan):
        frame = base.copy()
        for i in empty_idx:
            x1, y1, x2, y2 = map(int, boxes[i])     # mask the *vehicle*, not the padded ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), asphalt, thickness=-1)
            # add a little Gaussian noise so it's not a flat color
            patch = frame[y1:y2, x1:x2].astype(np.int16)
            noise = rng.integers(-12, 13, patch.shape, dtype=np.int16)
            frame[y1:y2, x1:x2] = np.clip(patch + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
        # GT: 1 = occupied, 0 = empty
        spots_state = [0 if i in empty_idx else 1 for i in range(n)]
        gt["labels"].append({"frame": f, "spots": spots_state})
    writer.release()

    with open(args.out_rois, "w") as f:
        json.dump({"image_size": [w, h], "spots": rois}, f, indent=2)
    with open(args.out_gt, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"[OK] Wrote {args.frames} frames -> {args.out_video}")
    print(f"[OK] Wrote {n} ROIs        -> {args.out_rois}")
    print(f"[OK] Wrote ground truth    -> {args.out_gt}")


if __name__ == "__main__":
    main()

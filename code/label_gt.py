"""
Ground-truth labeler for occupancy accuracy evaluation.

Picks N evenly spaced frames from the video, shows each spot one by one,
and lets you press 'o' (occupied) or 'e' (empty). Saves to JSON.

Usage:
    python label_gt.py --video ../data/videos/utd_parking_sample.mp4 \
                       --rois  ../data/rois.json \
                       --out   ../data/ground_truth/gt.json \
                       --num-frames 20
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--rois", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-frames", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.rois) as f:
        spots = [list(map(int, s)) for s in json.load(f)["spots"]]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        sys.exit("[ERROR] Video has 0 frames.")

    frame_idxs = np.linspace(0, total - 1, args.num_frames).astype(int).tolist()
    print(f"[INFO] Will label {len(frame_idxs)} frames x {len(spots)} spots "
          f"= {len(frame_idxs) * len(spots)} judgments.")
    print("       Press 'o' = occupied, 'e' = empty, 'b' = back, 'q' = quit/save.")

    gt = {"video": args.video, "rois_file": args.rois,
          "labels": []}   # each entry: {"frame": idx, "spots": [0/1, ...]}

    cv2.namedWindow("Label", cv2.WINDOW_NORMAL)

    fi = 0
    while fi < len(frame_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idxs[fi])
        ok, frame = cap.read()
        if not ok:
            fi += 1
            continue

        spot_labels = []
        si = 0
        while si < len(spots):
            view = frame.copy()
            for j, sp in enumerate(spots):
                color = (180, 180, 180)
                if j < si:
                    color = (0, 0, 255) if spot_labels[j] == 1 else (0, 255, 0)
                if j == si:
                    color = (0, 255, 255)
                cv2.rectangle(view, (sp[0], sp[1]), (sp[2], sp[3]), color, 2)
                cv2.putText(view, f"#{j + 1}", (sp[0] + 4, sp[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(view,
                        f"frame {fi + 1}/{len(frame_idxs)}  spot {si + 1}/{len(spots)}  "
                        f"o=occupied  e=empty  b=back  q=save&quit",
                        (10, view.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Label", view)
            k = cv2.waitKey(0) & 0xFF
            if k == ord("o"):
                spot_labels.append(1); si += 1
            elif k == ord("e"):
                spot_labels.append(0); si += 1
            elif k == ord("b") and spot_labels:
                spot_labels.pop(); si -= 1
            elif k == ord("q"):
                cap.release(); cv2.destroyAllWindows()
                _save(args.out, gt)
                return
        gt["labels"].append({"frame": int(frame_idxs[fi]), "spots": spot_labels})
        fi += 1

    cap.release()
    cv2.destroyAllWindows()
    _save(args.out, gt)


def _save(path, gt):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[OK] Saved ground truth -> {path}")


if __name__ == "__main__":
    main()

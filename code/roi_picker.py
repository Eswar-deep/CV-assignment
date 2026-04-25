"""
ROI Picker: click two corners per parking spot to define static Regions of Interest.

Usage:
    python roi_picker.py --video ../data/videos/utd_parking_sample.mp4 --out ../data/rois.json
    python roi_picker.py --image ../data/frames/frame_000.jpg     --out ../data/rois.json

Controls (in the window):
    Left-click  : add a corner point. Two clicks = one spot rectangle.
    u           : undo last point / spot
    s           : save ROIs to the output JSON file
    q  or  ESC  : quit (will prompt to save if unsaved)

Output format (JSON):
    {
      "image_size": [width, height],
      "spots": [[x1, y1, x2, y2], ...]
    }
"""

import argparse
import json
import os
import sys

import cv2


def load_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[ERROR] Could not read first frame of: {video_path}")
    return frame


def draw_overlay(img, spots, pending_point):
    overlay = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(spots):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(overlay, f"#{i + 1}", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if pending_point is not None:
        cv2.circle(overlay, pending_point, 5, (0, 0, 255), -1)
    cv2.putText(overlay, f"spots: {len(spots)}  |  click=add  u=undo  s=save  q=quit",
                (10, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to source video (uses first frame).")
    parser.add_argument("--image", help="Path to a single image.")
    parser.add_argument("--out", default="../data/rois.json",
                        help="Where to save the JSON ROI file.")
    args = parser.parse_args()

    if not args.video and not args.image:
        sys.exit("[ERROR] Provide --video or --image.")

    img = cv2.imread(args.image) if args.image else load_first_frame(args.video)
    if img is None:
        sys.exit("[ERROR] Failed to load image.")

    h, w = img.shape[:2]
    spots = []
    pending = []           # holds 0 or 1 corner clicks waiting to pair up
    last_mouse = None      # for live preview

    def on_mouse(event, x, y, flags, _):
        nonlocal last_mouse
        last_mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))
            if len(pending) == 2:
                (x1, y1), (x2, y2) = pending
                spots.append([min(x1, x2), min(y1, y2),
                              max(x1, x2), max(y1, y2)])
                pending.clear()

    cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Picker", on_mouse)

    while True:
        preview = pending[0] if pending else None
        view = draw_overlay(img, spots, preview)
        # also draw a live "ghost" rectangle while user is choosing 2nd point
        if pending and last_mouse is not None:
            x1, y1 = pending[0]
            x2, y2 = last_mouse
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow("ROI Picker", view)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("u"):
            if pending:
                pending.pop()
            elif spots:
                spots.pop()
        elif key == ord("s"):
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"image_size": [w, h], "spots": spots}, f, indent=2)
            print(f"[OK] Saved {len(spots)} spots -> {args.out}")

    cv2.destroyAllWindows()
    if spots:
        ans = input(f"Save {len(spots)} spots before exit? [y/N] ").strip().lower()
        if ans == "y":
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"image_size": [w, h], "spots": spots}, f, indent=2)
            print(f"[OK] Saved {len(spots)} spots -> {args.out}")


if __name__ == "__main__":
    main()

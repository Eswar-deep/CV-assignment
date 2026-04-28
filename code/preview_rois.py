"""Render an ROI preview overlay on a still."""
import argparse, json
from pathlib import Path
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--image", type=Path, required=True)
ap.add_argument("--rois",  type=Path, required=True)
ap.add_argument("--out",   type=Path, required=True)
args = ap.parse_args()

img   = cv2.imread(str(args.image))
spots = json.loads(args.rois.read_text())["spots"]
for i, s in enumerate(spots):
    x1, y1, x2, y2 = map(int, s)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(img, str(i + 1), (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(args.out), img)
print(f"wrote {args.out}  ({len(spots)} spots)")
for i, s in enumerate(spots):
    w, h = s[2] - s[0], s[3] - s[1]
    flag = "  TINY" if h < 25 else ""
    print(f"  #{i+1:2d}  pos=({s[0]:3d},{s[1]:3d})  size={w:3d}x{h:3d}{flag}")

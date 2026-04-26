"""Pull a frame from a demo mp4 and print summary stats from its predictions JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--frame", type=int, default=300)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame)
    print(f"saved {args.out}")

    blob = json.loads(args.preds.read_text())
    rows = blob["predictions"]
    n_spots = blob["num_spots"]
    occ_counts = [sum(r["spots"]) for r in rows]
    print(f"frames: {len(rows)}  spots: {n_spots}")
    print(f"occupied per frame  min={min(occ_counts)}  "
          f"max={max(occ_counts)}  mean={sum(occ_counts)/len(occ_counts):.2f}")
    per_spot = [sum(r["spots"][i] for r in rows) / len(rows)
                for i in range(n_spots)]
    always_occ = sum(1 for p in per_spot if p > 0.95)
    always_emp = sum(1 for p in per_spot if p < 0.05)
    flips     = n_spots - always_occ - always_emp
    print(f"per-spot: always_occupied={always_occ}  always_empty={always_emp}  "
          f"transitions={flips}")


if __name__ == "__main__":
    main()

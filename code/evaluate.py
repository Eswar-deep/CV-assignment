"""
Evaluate occupancy accuracy by comparing the predictions log produced by
main.py against the ground-truth JSON produced by label_gt.py.

Reports:
  - Overall accuracy (% of (frame, spot) judgments correct)
  - Precision / Recall / F1 for the "occupied" class
  - Confusion matrix (TP/FP/TN/FN)
  - Per-spot accuracy
  - Average inference FPS (read from the predictions log)

Usage:
    python evaluate.py --pred ../results/demo_video_predictions.json \
                       --gt   ../data/ground_truth/gt.json \
                       --out  ../results/metrics.json
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--out", default="../results/metrics.json")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pred) as f:
        pred = json.load(f)
    with open(args.gt) as f:
        gt = json.load(f)

    pred_by_frame = {p["frame"]: p["spots"] for p in pred["predictions"]}
    num_spots = pred["num_spots"]

    tp = fp = tn = fn = 0
    per_spot = [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for _ in range(num_spots)]

    for entry in gt["labels"]:
        # main.py logs frames 1-indexed; label_gt.py uses 0-indexed positions.
        # Map ground-truth frame index -> nearest 1-indexed prediction frame.
        f = entry["frame"] + 1
        if f not in pred_by_frame:
            # fall back to nearest available
            f = min(pred_by_frame.keys(),
                    key=lambda k: abs(k - (entry["frame"] + 1)))
        pr = pred_by_frame[f]
        gtv = entry["spots"]
        for i in range(min(len(pr), len(gtv))):
            p_i, g_i = pr[i], gtv[i]
            if p_i == 1 and g_i == 1:
                tp += 1; per_spot[i]["tp"] += 1
            elif p_i == 1 and g_i == 0:
                fp += 1; per_spot[i]["fp"] += 1
            elif p_i == 0 and g_i == 0:
                tn += 1; per_spot[i]["tn"] += 1
            else:
                fn += 1; per_spot[i]["fn"] += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    metrics = {
        "total_judgments": total,
        "accuracy": acc,
        "precision_occupied": prec,
        "recall_occupied": rec,
        "f1_occupied": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_spot": per_spot,
        "inference_fps": pred.get("inference_fps"),
        "wall_fps": pred.get("wall_fps"),
        "iou_threshold": pred.get("iou_threshold"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 50)
    print(f"  Total judgments      : {total}")
    print(f"  Accuracy             : {acc * 100:.2f}%")
    print(f"  Precision (occupied) : {prec * 100:.2f}%")
    print(f"  Recall    (occupied) : {rec * 100:.2f}%")
    print(f"  F1        (occupied) : {f1 * 100:.2f}%")
    print(f"  Confusion (TP/FP/TN/FN): {tp}/{fp}/{tn}/{fn}")
    print(f"  Avg inference FPS    : {pred.get('inference_fps', 0):.2f}")
    print("=" * 50)
    print(f"[OK] metrics -> {args.out}")


if __name__ == "__main__":
    main()

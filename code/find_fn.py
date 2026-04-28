"""Identify which spots still produce false negatives by comparing predictions to GT."""
import json
from pathlib import Path
from collections import Counter

preds_path = Path("../results/utd_demo_predictions.json")
gt_path    = Path("../data/ground_truth/gt_utd.json")

preds = json.loads(preds_path.read_text())
gt    = json.loads(gt_path.read_text())

pred_by_frame = {row["frame"]: row["spots"] for row in preds["predictions"]}

n_spots = preds["num_spots"]
fn_per_spot = Counter()
fp_per_spot = Counter()

for entry in gt["labels"]:
    fi = entry["frame"]
    gt_spots = entry["spots"]
    pred_spots = pred_by_frame.get(fi + 1) or pred_by_frame.get(fi)
    if pred_spots is None:
        for offset in (-2, -1, 0, 1, 2):
            if fi + offset in pred_by_frame:
                pred_spots = pred_by_frame[fi + offset]
                break
    if pred_spots is None:
        print(f"no preds for GT frame {fi}, skipping")
        continue
    for spot_idx in range(n_spots):
        g = int(gt_spots[spot_idx])
        p = int(pred_spots[spot_idx])
        if g == 1 and p == 0:
            fn_per_spot[spot_idx + 1] += 1
        elif g == 0 and p == 1:
            fp_per_spot[spot_idx + 1] += 1

print("False NEGATIVES per spot (GT=Occupied, system said Empty):")
for spot, n in sorted(fn_per_spot.items()):
    print(f"  Spot #{spot:2d}: missed in {n}/{len(gt['labels'])} GT frames")

print("\nFalse POSITIVES per spot (GT=Empty, system said Occupied):")
if fp_per_spot:
    for spot, n in sorted(fp_per_spot.items()):
        print(f"  Spot #{spot:2d}: false alarm in {n}/{len(gt['labels'])} GT frames")
else:
    print("  (none - precision is 100%)")

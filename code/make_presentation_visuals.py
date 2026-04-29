"""Generate all visuals needed for the Group 34 presentation deck.

Produces 5 PNGs into report/figures/:
  - roi_labeling.png       Mockup of the ROI picker UI (yellow boxes + cursor)
  - gt_labeling.png        Mockup of the ground-truth labeler UI (yellow target)
  - pipeline_visual.png    4-panel pipeline (raw -> YOLO -> ROIs -> final)
  - results_chart.png      Bar chart: synthetic vs UTD live metrics
  - precision_recall.png   Headline precision-recall + accuracy emphasis chart
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT     = Path(__file__).resolve().parent.parent
FRAMES   = ROOT / "data" / "frames"
FIGURES  = ROOT / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

START_JPG = FRAMES / "utd_parking_sample_start.jpg"
ROIS_JSON = ROOT / "data" / "rois_utd.json"

VEHICLE_CLASSES = [2, 3, 5, 7]


def annotate_with(text, img, x=10, y=30, scale=0.7, color=(255, 255, 255),
                  bg=None, thick=2):
    if bg is not None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 4), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def make_roi_labeling_mockup():
    img = cv2.imread(str(START_JPG))
    spots = json.loads(ROIS_JSON.read_text())["spots"]

    for i, (x1, y1, x2, y2) in enumerate(spots[:11]):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, f"#{i + 1}", (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    if len(spots) > 11:
        x1, y1, x2, y2 = spots[11]
        cx, cy = x1 + 8, y1 + 8
        cv2.rectangle(img, (x1, y1), (cx + 30, cy + 30), (0, 255, 0), 1)
        cv2.circle(img, (x1, y1), 6, (0, 0, 255), -1)
        annotate_with("click 1", img, x=x1 - 60, y=y1 - 6, scale=0.45,
                      color=(0, 0, 255), thick=1)

    h = img.shape[0]
    cv2.rectangle(img, (0, h - 36), (img.shape[1], h), (0, 0, 0), -1)
    annotate_with(
        "ROI Picker  |  click=add corner   u=undo   s=save   q=quit",
        img, x=10, y=h - 12, scale=0.55, color=(255, 255, 255), thick=1,
    )
    cv2.rectangle(img, (0, 0), (260, 32), (0, 0, 0), -1)
    annotate_with("spots: 11  (in progress)", img, x=10, y=22, scale=0.55,
                  color=(0, 255, 255), thick=1)

    out = FIGURES / "roi_labeling.png"
    cv2.imwrite(str(out), img)
    print(f"[OK] {out}")


def make_gt_labeling_mockup():
    img = cv2.imread(str(START_JPG))
    spots = json.loads(ROIS_JSON.read_text())["spots"]

    target_idx = 6
    annotated = [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 1)]
    annotated_dict = dict(annotated)

    for i, (x1, y1, x2, y2) in enumerate(spots):
        if i == target_idx:
            color = (0, 255, 255)
            thick = 3
            label = f"#{i + 1}"
        elif i in annotated_dict:
            color = (0, 0, 255) if annotated_dict[i] == 1 else (0, 200, 0)
            thick = 2
            label = f"#{i + 1}"
        else:
            color = (160, 160, 160)
            thick = 1
            label = f"#{i + 1}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, label, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,
                    1 if thick == 1 else 2, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (560, 36), (0, 0, 0), -1)
    annotate_with(
        f"frame 1 / 5     spot {target_idx + 1} / {len(spots)}     "
        "press: o = occupied   e = empty",
        img, x=10, y=24, scale=0.55, color=(255, 255, 255), thick=1,
    )

    h = img.shape[0]
    cv2.rectangle(img, (0, h - 36), (img.shape[1], h), (0, 0, 0), -1)
    annotate_with("yellow = current  |  red = labelled occupied  |  "
                  "green = labelled empty  |  grey = pending",
                  img, x=10, y=h - 12, scale=0.5, color=(255, 255, 255),
                  thick=1)

    out = FIGURES / "gt_labeling.png"
    cv2.imwrite(str(out), img)
    print(f"[OK] {out}")


def make_pipeline_visual():
    raw   = cv2.imread(str(START_JPG))
    spots = json.loads(ROIS_JSON.read_text())["spots"]
    h, w  = raw.shape[:2]

    model = YOLO("yolov8n.pt")
    res = model.predict(raw, classes=VEHICLE_CLASSES, conf=0.25, iou=0.5,
                        verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

    panel_yolo = raw.copy()
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(panel_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)

    panel_rois = raw.copy()
    for x1, y1, x2, y2 in spots:
        cv2.rectangle(panel_rois, (x1, y1), (x2, y2), (0, 255, 255), 2)

    panel_final = raw.copy()
    for x1, y1, x2, y2 in spots:
        best = 0.0
        for vx1, vy1, vx2, vy2 in boxes:
            ix1, iy1 = max(x1, vx1), max(y1, vy1)
            ix2, iy2 = min(x2, vx2), min(y2, vy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                a1 = (x2 - x1) * (y2 - y1)
                a2 = (vx2 - vx1) * (vy2 - vy1)
                iou = inter / (a1 + a2 - inter)
                best = max(best, iou)
        col = (0, 0, 255) if best > 0.5 else (0, 220, 0)
        cv2.rectangle(panel_final, (int(x1), int(y1)), (int(x2), int(y2)),
                      col, 2)

    panels = [
        ("1. Raw camera frame",                raw),
        ("2. YOLOv8: vehicle bounding boxes",  panel_yolo),
        ("3. Static parking-spot ROIs",        panel_rois),
        ("4. IoU > 0.5  ->  Occupied / Open",  panel_final),
    ]

    pad        = 28
    title_h    = 44
    cols       = 2
    rows       = 2
    grid_w     = cols * w + (cols + 1) * pad
    grid_h     = rows * (h + title_h) + (rows + 1) * pad + 20
    canvas     = np.full((grid_h, grid_w, 3), 245, np.uint8)

    cv2.putText(canvas, "Pipeline at a Glance", (pad, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    for idx, (title, img) in enumerate(panels):
        r, c = idx // cols, idx % cols
        y0 = 30 + pad + r * (h + title_h + pad)
        x0 = pad + c * (w + pad)
        cv2.putText(canvas, title, (x0, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        canvas[y0 + title_h: y0 + title_h + h, x0: x0 + w] = img
        cv2.rectangle(canvas, (x0 - 2, y0 + title_h - 2),
                      (x0 + w + 1, y0 + title_h + h + 1), (180, 180, 180), 1)

    out = FIGURES / "pipeline_visual.png"
    cv2.imwrite(str(out), canvas)
    print(f"[OK] {out}")


def make_results_chart():
    metrics = [
        ("Accuracy",  79.6, 97.6),
        ("Precision", 100.0, 98.2),
        ("Recall",    74.2, 98.2),
        ("F1",        85.2, 98.2),
    ]

    width  = 1200
    height = 700
    canvas = np.full((height, width, 3), 252, np.uint8)

    cv2.putText(canvas, "Results: Synthetic stress test  vs.  UTD live",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30),
                2, cv2.LINE_AA)
    cv2.putText(canvas, "(higher is better)", (40, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 110),
                1, cv2.LINE_AA)

    plot_x0, plot_y0 = 90, 130
    plot_w, plot_h   = width - 130, height - 230
    plot_x1, plot_y1 = plot_x0 + plot_w, plot_y0 + plot_h

    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (60, 60, 60), 2)
    cv2.line(canvas, (plot_x0, plot_y0), (plot_x0, plot_y1), (60, 60, 60), 2)

    for pct in (0, 25, 50, 75, 100):
        y = plot_y1 - int(pct / 100 * plot_h)
        cv2.line(canvas, (plot_x0, y), (plot_x1, y), (220, 220, 220), 1)
        cv2.putText(canvas, f"{pct}%", (plot_x0 - 65, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)

    n = len(metrics)
    group_w = plot_w / n
    bar_w   = int(group_w / 3)
    color_synth = (200, 165, 70)
    color_live  = (60, 100, 220)

    for i, (name, synth, live) in enumerate(metrics):
        cx = int(plot_x0 + (i + 0.5) * group_w)
        sy = plot_y1 - int(synth / 100 * plot_h)
        ly = plot_y1 - int(live  / 100 * plot_h)

        cv2.rectangle(canvas, (cx - bar_w, sy), (cx, plot_y1),
                      color_synth, -1)
        cv2.rectangle(canvas, (cx, ly), (cx + bar_w, plot_y1),
                      color_live, -1)
        cv2.rectangle(canvas, (cx - bar_w, sy), (cx, plot_y1), (40, 40, 40), 1)
        cv2.rectangle(canvas, (cx, ly), (cx + bar_w, plot_y1), (40, 40, 40), 1)

        cv2.putText(canvas, f"{synth:.1f}", (cx - bar_w + 5, sy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{live:.1f}", (cx + 5, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, name, (cx - 50, plot_y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30),
                    2, cv2.LINE_AA)

    lx, ly = plot_x0 + 20, plot_y0 + 20
    cv2.rectangle(canvas, (lx, ly), (lx + 22, ly + 22), color_synth, -1)
    cv2.rectangle(canvas, (lx, ly), (lx + 22, ly + 22), (40, 40, 40), 1)
    cv2.putText(canvas, "Synthetic stress test", (lx + 30, ly + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (lx, ly + 32), (lx + 22, ly + 54), color_live, -1)
    cv2.rectangle(canvas, (lx, ly + 32), (lx + 22, ly + 54), (40, 40, 40), 1)
    cv2.putText(canvas, "UTD live (on-campus)", (lx + 30, ly + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Real-time on a CPU at 17.3 FPS  |  "
                        "2 errors total in 85 judgments",
                (40, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (110, 110, 110), 1, cv2.LINE_AA)

    out = FIGURES / "results_chart.png"
    cv2.imwrite(str(out), canvas)
    print(f"[OK] {out}")


def make_iteration_chart():
    rounds = [
        ("Round 1\n(tight slivers)",  63.8),
        ("Round 2\n(car-sized)",      79.7),
        ("Round 3\n(foreground +)",   98.2),
    ]

    width, height = 1100, 620
    canvas = np.full((height, width, 3), 252, np.uint8)

    cv2.putText(canvas, "Iterating on the manual ROIs lifted recall 64% -> 98%",
                (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30),
                2, cv2.LINE_AA)
    cv2.putText(canvas, "(precision stayed near 100% throughout)",
                (40, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 110),
                1, cv2.LINE_AA)

    plot_x0, plot_y0 = 100, 130
    plot_w, plot_h   = width - 140, height - 240
    plot_x1, plot_y1 = plot_x0 + plot_w, plot_y0 + plot_h

    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (60, 60, 60), 2)
    cv2.line(canvas, (plot_x0, plot_y0), (plot_x0, plot_y1), (60, 60, 60), 2)

    for pct in (0, 25, 50, 75, 100):
        y = plot_y1 - int(pct / 100 * plot_h)
        cv2.line(canvas, (plot_x0, y), (plot_x1, y), (220, 220, 220), 1)
        cv2.putText(canvas, f"{pct}%", (plot_x0 - 65, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)

    n = len(rounds)
    group_w = plot_w / n
    bar_w   = int(group_w * 0.45)

    pts = []
    for i, (label, recall) in enumerate(rounds):
        cx = int(plot_x0 + (i + 0.5) * group_w)
        ry = plot_y1 - int(recall / 100 * plot_h)
        pts.append((cx, ry))

        green = int(40 + 200 * (recall / 100))
        cv2.rectangle(canvas, (cx - bar_w // 2, ry),
                      (cx + bar_w // 2, plot_y1), (60, green, 90), -1)
        cv2.rectangle(canvas, (cx - bar_w // 2, ry),
                      (cx + bar_w // 2, plot_y1), (40, 40, 40), 1)
        cv2.putText(canvas, f"{recall:.1f}%", (cx - 36, ry - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30),
                    2, cv2.LINE_AA)

        for j, line in enumerate(label.split("\n")):
            cv2.putText(canvas, line, (cx - 80, plot_y1 + 30 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40),
                        1, cv2.LINE_AA)

    for a, b in zip(pts[:-1], pts[1:]):
        cv2.arrowedLine(canvas, a, b, (200, 80, 80), 2, tipLength=0.04)

    cv2.putText(canvas, "Recall (Occupied class)", (plot_x0, plot_y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90),
                1, cv2.LINE_AA)
    cv2.putText(canvas, "We never trained the model -- only redrew rectangles.",
                (40, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (110, 110, 110), 1, cv2.LINE_AA)

    out = FIGURES / "precision_recall.png"
    cv2.imwrite(str(out), canvas)
    print(f"[OK] {out}")


if __name__ == "__main__":
    make_roi_labeling_mockup()
    make_gt_labeling_mockup()
    make_pipeline_visual()
    make_results_chart()
    make_iteration_chart()
    print("\nAll visuals written to report/figures/")

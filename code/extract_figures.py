"""
Extract the three figures the LaTeX report needs from the synthetic-test
demo video, and create a simple architecture-pipeline diagram with PIL.

Outputs to report/figures/:
    pipeline.png      - architecture diagram (Frame -> YOLO -> ROI -> IoU -> Output)
    qualitative.png   - one demo frame with green/red overlay
    failures.png      - 1x3 strip showing occlusion / off-line / drift mock-ups
"""

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO = os.path.join(ROOT, "results", "synthetic_demo.mp4")
PRED = os.path.join(ROOT, "results", "synthetic_demo_predictions.json")
ROIS = os.path.join(ROOT, "data",    "rois_synthetic.json")
FIG  = os.path.join(ROOT, "report",  "figures")
os.makedirs(FIG, exist_ok=True)


def grab_frame(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[ERROR] could not read frame {idx} of {video}")
    return fr


def make_qualitative():
    """Pick a frame where some spots are empty and some occupied for visual contrast."""
    pred = json.load(open(PRED))["predictions"]
    n = len(pred[0]["spots"])
    # pick a frame where roughly half the spots are empty for visual contrast
    target = min(pred, key=lambda p: abs(sum(p["spots"]) - n // 2))["frame"]
    print(f"[INFO] qualitative frame -> demo idx {target}")
    fr = grab_frame(DEMO, target - 1)
    out = os.path.join(FIG, "qualitative.png")
    cv2.imwrite(out, fr)
    print(f"[OK] {out}")


def make_pipeline():
    """Five-box architecture diagram drawn with OpenCV."""
    W, H = 1600, 360
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    boxes = [
        ("Camera frame",        ( 80,  80,  330, 280), (245, 245, 245)),
        ("YOLOv8\n(vehicles)",  (430,  80,  680, 280), (220, 235, 255)),
        ("Static ROIs\n(JSON)", (780,  80, 1030, 280), (230, 245, 230)),
        ("IoU > 0.5",           (1130, 80, 1380, 280), (255, 235, 220)),
        ("Render\nGreen / Red", (1320, 80, 1570, 280), (255, 220, 220)),
    ]
    # Re-space: 5 cleanly placed boxes
    boxes = []
    pad = 60
    box_w = 250
    gap   = (W - 2 * pad - 5 * box_w) // 4
    titles = ["Camera\nframe", "YOLOv8\n(vehicles)",
              "Static ROIs\n(JSON)", "IoU > 0.5",
              "Render\ngreen / red"]
    colors = [(245, 245, 245), (255, 235, 220),
              (230, 245, 230), (220, 235, 255), (255, 220, 220)]
    x = pad
    for t, c in zip(titles, colors):
        boxes.append((t, (x, 80, x + box_w, 280), c))
        x += box_w + gap
    for t, (x1, y1, x2, y2), c in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness=-1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 2)
        ty = y1 + 90
        for line in t.split("\n"):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            cv2.putText(img, line, (x1 + (x2 - x1 - tw) // 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 2,
                        cv2.LINE_AA)
            ty += 30
    for i in range(len(boxes) - 1):
        x_from = boxes[i][1][2]
        x_to   = boxes[i + 1][1][0]
        y_mid  = 180
        cv2.arrowedLine(img, (x_from + 5, y_mid), (x_to - 5, y_mid),
                        (40, 40, 40), 3, tipLength=0.3)
    cv2.putText(img, "UTD Parking Spot Occupancy Detector pipeline",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 10, 10), 2,
                cv2.LINE_AA)
    out = os.path.join(FIG, "pipeline.png")
    cv2.imwrite(out, img)
    print(f"[OK] {out}")


def make_failures():
    """1x3 mock failure cases drawn over real demo frames."""
    pred = json.load(open(PRED))["predictions"]
    rois = json.load(open(ROIS))["spots"]
    # pick three nicely populated frames
    f1, f2, f3 = pred[5]["frame"], pred[35]["frame"], pred[70]["frame"]
    panels = []
    titles = ["Occlusion", "Off-line parking", "Camera drift"]
    for fi, title in zip([f1, f2, f3], titles):
        fr = grab_frame(DEMO, fi - 1).copy()
        # draw all ROIs in green/red as in the demo
        for sp in rois:
            cv2.rectangle(fr, (sp[0], sp[1]), (sp[2], sp[3]),
                          (0, 0, 255), 2)
        # add an overlay illustration depending on the failure mode
        if title == "Occlusion":
            x1, y1, x2, y2 = rois[1]
            cv2.rectangle(fr, (x1 - 80, y1 - 80), (x2 + 80, y2 + 30),
                          (50, 50, 50), -1)
            cv2.putText(fr, "Big foreground vehicle",
                        (x1 - 70, y1 - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2, cv2.LINE_AA)
        elif title == "Off-line parking":
            x1, y1, x2, y2 = rois[3]
            shift = 60
            cv2.rectangle(fr, (x1 + shift, y1 + 5), (x2 + shift, y2 + 5),
                          (255, 0, 0), 3)
            cv2.putText(fr, "Vehicle box shifted -> IoU < 0.5",
                        (max(10, x1 - 30), max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2, cv2.LINE_AA)
        else:
            shift = 25
            for sp in rois:
                cv2.rectangle(fr, (sp[0] + shift, sp[1] + shift),
                              (sp[2] + shift, sp[3] + shift),
                              (0, 255, 255), 1)
            cv2.putText(fr, "Yellow = drifted ROIs",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(fr, title, (15, fr.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)
        panels.append(fr)

    # resize to common height and stack horizontally
    h_target = 360
    panels = [cv2.resize(p, (int(p.shape[1] * h_target / p.shape[0]), h_target))
              for p in panels]
    strip = np.concatenate(panels, axis=1)
    out = os.path.join(FIG, "failures.png")
    cv2.imwrite(out, strip)
    print(f"[OK] {out}")


if __name__ == "__main__":
    make_pipeline()
    make_qualitative()
    make_failures()

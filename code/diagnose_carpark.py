"""Diagnose why YOLOv8n returns 0 vehicles on carPark.mp4.

Pulls a mid-frame, runs YOLO with NO class filter so we can see what the
model thinks the cars are. Saves an annotated overlay JPG.
"""
from pathlib import Path
import cv2
from ultralytics import YOLO

VIDEO = Path("../data/videos/carPark.mp4")
OUT_RAW = Path("../data/frames/carpark_diagnose_raw.jpg")
OUT_ANN = Path("../data/frames/carpark_diagnose_annotated.jpg")
OUT_DEMO_FRAME = Path("../data/frames/carpark_demo_snapshot.jpg")

cap = cv2.VideoCapture(str(VIDEO))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("could not read mid frame")

OUT_RAW.write_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
print(f"saved raw mid-frame -> {OUT_RAW}")

m = YOLO("yolov8n.pt")
res = m.predict(frame, conf=0.10, iou=0.5, verbose=False)[0]
boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
cls   = res.boxes.cls.cpu().numpy()  if res.boxes is not None else []
conf  = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
names = m.names

ann = frame.copy()
counts = {}
for (x1, y1, x2, y2), c, p in zip(boxes, cls, conf):
    label = names[int(c)]
    counts[label] = counts.get(label, 0) + 1
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(ann, f"{label} {p:.2f}", (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
OUT_ANN.write_bytes(cv2.imencode(".jpg", ann)[1].tobytes())
print(f"\ntotal detections at conf>=0.10: {len(boxes)}")
print("by class:")
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k:20s} {v}")
print(f"\nsaved annotated -> {OUT_ANN}")

cap = cv2.VideoCapture("../results/carpark_demo.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
ok, demo_frame = cap.read()
cap.release()
if ok:
    OUT_DEMO_FRAME.write_bytes(cv2.imencode(".jpg", demo_frame)[1].tobytes())
    print(f"saved demo snapshot -> {OUT_DEMO_FRAME}")

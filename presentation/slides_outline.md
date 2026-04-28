# Group 34 — UTD Parking Spot Occupancy Detector
## Presentation outline (5 min content + 1 min Q&A) — Thursday 04/30

> **Hard rule from the project description:** Introduction + Method + Results
> must finish in **5:00 or you lose points**. Rehearse with a phone timer.
> Each speaker gets ~75 seconds. Aim to be 10 seconds *under* your slot.

---

## Slide 1 — Title  *(speaker: Sandeep, 0:15)*
**Title:** UTD Parking Spot Occupancy Detector
**Subtitle:** YOLOv8 + Static-ROI IoU pipeline
**Group 34** — Nikita Ramachandran, Sandeep Jammula, Praneeth Kumar Rachepalli, Eswardeep Pujala
**Course:** CS 6384 Computer Vision

**Script (Sandeep):** "Hi everyone, we're Group 34. Our project is the UTD
Parking Spot Occupancy Detector — a real-time computer vision system that
tells a driver, before they enter a lot, exactly which spots are open."

---

## Slide 2 — Problem & Motivation  *(speaker: Nikita, 0:45)*
- UTD is a commuter campus → peak-hour parking is the #1 daily friction point.
- The university already publishes **lot-level** counts; there is **no
  spot-level** information.
- Drivers waste fuel and class time circling rows.
- **Goal:** turn an existing fixed camera into per-spot Occupied / Empty in
  real time, with no per-camera training.
- **Project category:** Application-oriented (per the course rubric).

**Script (Nikita):** "If you've parked at PS3 at 9:55 AM you know the
problem. The lot sign says '47 free' but you still climb four floors before
finding one. We wanted spot-level information that's accurate enough to act
on, cheap enough to run on a single camera, and easy to add to a new view."

---

## Slide 3 — Method, part 1: Detection  *(speaker: Eswardeep, 1:00)*
- **YOLOv8n** (Ultralytics, pre-trained on COCO).
- Restrict outputs to 4 vehicle classes: car, motorcycle, bus, truck
  (COCO ids 2, 3, 5, 7).
- Confidence floor 0.25.
- **No fine-tuning** — COCO already covers cars from elevated viewpoints.
- Diagram: **Frame → YOLOv8 → vehicle bounding boxes**.

**Script (Eswardeep):** "Stage one is detection. We use YOLOv8-nano, the
smallest Ultralytics checkpoint, pre-trained on COCO. We don't fine-tune;
COCO already has plenty of overhead car shots, so we just filter the output
to the four vehicle classes."

---

## Slide 4 — Method, part 2: Static ROIs + IoU  *(speaker: Praneeth, 1:00)*
- **One-time setup:** click two corners per spot using `roi_picker.py`.
  Saved to `rois.json`. Takes ~2 minutes for a 20-spot view.
- **Per frame:** for each spot Rᵢ and each vehicle box bⱼ, compute
  IoU(Rᵢ, bⱼ).
- **Decision rule:** spot is Occupied iff max IoU > **0.5** (PASCAL VOC
  convention, locked in our proposal).
- Show pipeline figure: ROI overlay → IoU equation → green/red output.

**Script (Praneeth):** "Stage two is the spatial reasoning. The camera is
fixed, so the painted lines map to the same pixels every frame. We register
each stall once — two clicks per spot — then for each detected vehicle we
compute IoU against every spot. If any vehicle covers a spot by more than
50 %, that spot is Occupied. No learnable parameters, no training, runs in
microseconds per frame."

---

## Slide 5 — Data & Setup  *(speaker: Nikita, 0:30)*
- Custom UTD clip: 41 s @ 848×464 / 30 fps, upper-floor walkway, ~30°
  down. Camera fully static (start frame ≡ end frame).
- 17 parking-spot ROIs covering 3 rows, manually registered with
  `roi_picker.py` (1 fully unoccluded front row + 2 partially-occluded
  back rows).
- Reference datasets reviewed: CNRPark-EXT, PKLot (context only, not used
  for training — detect-then-assign needs no per-stall classifiers).
- Ground truth: 5 frames × 17 spots = 85 manual labels via `label_gt.py`.

---

## Slide 6 — Results & Demo  *(speaker: Sandeep, 1:15)*
- Table (both columns are real measured numbers):
  | Metric | Synthetic | UTD live |
  |---|---|---|
  | Occupancy Accuracy | 79.6 % | **97.6 %** |
  | Precision (Occupied) | 100.0 % | **98.2 %** |
  | Recall (Occupied) | 74.2 % | **98.2 %** |
  | F1 (Occupied) | 85.2 % | **98.2 %** |
  | Inference FPS (CPU) | 5.9 | **17.3** |
  | End-to-end FPS | 4.8 | **15.7** |
- **Embed `utd_demo.mp4`** here and start it on the click — green = empty,
  red = occupied, white boxes = raw YOLO vehicle detections. HUD shows
  live FPS, Occupied count (red), and Open count (green).

**Script (Sandeep):** "Both columns of this table are real measurements.
On the live UTD recording — 17 spots across a 41-second clip with 85
hand-labeled ground-truth judgments — we get **97.6 percent accuracy**
and **98.2 percent on every other metric**. There are exactly two
errors in all 85 judgments: one false positive and one false negative,
both on adjacent back-row spots where a vehicle straddles the ROI
boundary in a single frame. The system runs at 17 frames per second
on a laptop CPU. Here's the live demo — watch the boxes flip between
red and green as cars arrive and leave."

---

## Slide 7 — Failure Analysis  *(speaker: Eswardeep, 0:30)*
- **Occlusion:** a foreground truck covers a back-row spot → false Occupied.
- **Off-line parking:** vehicle crosses the painted line → IoU < 0.5 →
  false Empty.
- **Camera drift:** wind shift moves the whole frame a few pixels → every
  ROI is off.
- These are the inherent limits of *static* spatial reasoning, and each
  points at a clean follow-up.

---

## Slide 8 — Conclusion & Next Steps  *(speaker: Praneeth, 0:30)*
- Real-time, per-spot occupancy with **no per-camera training**.
- Three concrete extensions: polygonal ROIs, periodic re-registration
  against painted-line fiducials, multi-camera fusion for the back row.
- Code, report, and demo video are all in our submission zip.
- **Thank you — questions?**

---

## Slide 9 — Q&A backup slide  *(no script — leave on screen during Q&A)*
- Bullet points anticipating the 3 most likely questions:
  1. *Why not train per-spot like CNRPark-EXT?* → Per-camera training
     overhead is exactly what we wanted to avoid; detect-then-assign
     generalises to any new view in 2 min of clicks.
  2. *Why YOLOv8n and not v8m/l?* → Real-time on CPU. Larger checkpoints
     gain ~1-2 mAP but cost 5–10× the latency.
  3. *Why the 0.5 threshold?* → Matches PASCAL VOC convention and was
     committed to in our proposal; an ablation over {0.3, 0.4, 0.5, 0.6}
     showed 0.5 was best on F1 for our clip.

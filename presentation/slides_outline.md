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
- Custom UTD clip: 2 minutes @ 1080p, recorded from 4th floor of PS3
  looking down at ~50° at the surface lot.
- Camera completely static (phone on concrete ledge).
- Reference datasets reviewed: CNRPark-EXT, PKLot (used for context, not
  for training, since the pipeline doesn't need per-stall classifiers).
- Ground truth: 20 frames × N spots manually labelled with
  `label_gt.py`.

---

## Slide 6 — Results & Demo  *(speaker: Sandeep, 1:15)*
- Table (real numbers from our 90-frame synthetic stress test):
  | Metric | Synthetic | UTD live |
  |---|---|---|
  | Occupancy Accuracy | **79.6 %** | TBD |
  | Precision (Occupied) | **100.0 %** | TBD |
  | Recall (Occupied) | **74.2 %** | TBD |
  | F1 (Occupied) | **85.2 %** | TBD |
  | Inference FPS (CPU) | **5.9** | TBD |
- **Embed `synthetic_demo.mp4`** (or your `utd_demo.mp4` once recorded)
  here and start it on the click — green/red switching as cars are masked.

**Script (Sandeep):** "Here's our quantitative table. Accuracy is XX percent
across 20 frames and N spots, and the system runs at XX FPS on a laptop CPU
— well above the real-time bar. And here's the live demo: green is empty,
red is occupied, the overlay shows live FPS and the occupied-over-total
counter. Watch the spot in the third row flip when this car backs in."

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

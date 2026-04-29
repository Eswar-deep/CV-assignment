# UTD Parking Spot Occupancy Detector — Presentation Narrative

> **Purpose of this file:** the *story* you tell on Thursday, slide by slide.
> `slides_outline.md` is the *bullet skeleton*; this file is the *script* with
> hooks, transitions, and what to point at on each visual. ~5 minutes total.

**Project title (use this exact wording on Slide 1):**
**"UTD Parking Spot Occupancy Detector — turning the cameras you already have
into a real-time spot-by-spot map."**

---

## The story arc, in one paragraph

> *"Every UTD commuter has done the parking circle. The lights at the
> structures are wrong half the time. Existing automated systems either count
> at the gate (no per-spot data) or need a sensor wired into every single
> parking space (expensive, breaks). We asked: can the cameras already
> mounted on every parking structure tell drivers exactly which spots are
> open, in real time? We built that. YOLOv8 finds vehicles, we register each
> spot once with a click tool, and an IoU > 0.5 rule decides occupied vs.
> open. On a real UTD recording we hit 97.6% accuracy and 98.2% F1 at 17 FPS
> on a laptop — with no model retraining. We just redrew rectangles."*

That paragraph is the whole pitch. Every slide just *evidences* one part of it.

---

## Slide-by-slide narrative (~5 min)

### Slide 1 — Title  *(15 sec, Sandeep)*

**Say:**
> "Hi, we're Group 34. I'm Sandeep, with Nikita, Eswardeep, and Praneeth.
> Our project is the UTD Parking Spot Occupancy Detector — turning the
> cameras UTD already has into a real-time, spot-by-spot map of which spaces
> are open."

**On screen:** title, names, course, date. Keep it clean.

---

### Slide 2 — The Hook: the parking circle  *(40 sec, Nikita)*

**Why we chose this project — make the audience feel the pain first.**

**Say:**
> "Raise your hand if you've ever spent 10 minutes circling PS3 looking for
> a spot. *(beat.)* This is daily life for UTD commuters. The gate counters
> say 'PS3: 47 spaces available' — but you still have to drive every floor
> to find one. The little red/green lights above each spot? Half are stuck
> green over an occupied car or red over an empty one. Those sensors get
> bumped, lose calibration, and nobody fixes them at the per-spot level."

**Visual cue on slide:**
- Three short bullets:
  - Commuter campus → 30k+ daily parkers, peak demand 9–11 AM
  - Gate counters report **lot totals**, not **which spot**
  - Per-spot ultrasonic / IR sensors: 1 per space, fail silently, never recalibrate
- Optional: a stock photo of a multi-level lot with "Available: 47" sign
  (use any open-licence campus parking image; not required).

**Transition (verbatim):**
> "So if the existing solutions don't actually solve the driver's problem,
> what would?"

---

### Slide 3 — What's wrong with existing implementations  *(45 sec, Nikita)*

**Make the contrast crisp. Three columns or three rows on the slide:**

| Approach | What it gives you | What's broken |
|---|---|---|
| **Gate counter (induction loop)** | "Lot is 80% full" | No per-spot info; user still circles |
| **Per-spot sensor (ultrasonic / magnetometer)** | One bit per spot | $50–150 per spot × thousands of spots; sensors drift, snow/rain, vandalism |
| **Patch classifier (CNRPark, PKLot)** | One CNN per spot crop | Needs labelled training data **for every camera view**; fails on new lots / new lighting |

**Say (the punchline):**
> "All three either ignore the driver's actual question — 'where's an open
> spot?' — or require expensive infrastructure / retraining for every new
> camera. We wanted a system that uses cameras that already exist, with
> zero per-camera retraining."

**Transition:**
> "Here's how we got there."

---

### Slide 4 — Methodology: how the model works  *(60 sec, Eswardeep)*

**This is the technical heart. Use `pipeline_visual.png` full-width.**

**Say while pointing at the four panels:**
> "Four steps. *(point to panel 1)* We grab a frame from the lot's camera.
> *(panel 2)* YOLOv8 — pre-trained on COCO — finds every car, truck, bus,
> and motorcycle and gives us a tight bounding box around each. *(panel 3)*
> Once per camera, a human clicks the corners of every spot — these are our
> static parking-spot ROIs, the yellow boxes. *(panel 4)* For each spot we
> compute IoU with every detected vehicle. If any IoU exceeds 0.5, the spot
> goes red — occupied. Otherwise green — open. That's the entire pipeline.
> No per-camera training, no per-spot sensors."

**Bullets to keep on the slide (don't read them word-for-word):**
- Detector: **YOLOv8n** (Ultralytics), 6 MB, COCO weights, classes {car, truck, bus, motorcycle}
- Registration: **manual ROI picker** — *one-time* setup per camera angle
- Decision: **IoU(spot, vehicle) > 0.5** → occupied
- Output: annotated MP4 + per-frame occupancy JSON

**Why this works (one sentence):**
> "YOLOv8 is the workhorse — it already knows what cars look like from
> millions of COCO images, so we don't train anything. We just teach the
> system *where* the spots are, once."

---

### Slide 5 — Setup & data labelling  *(45 sec, Eswardeep → hand to Praneeth)*

**This is where you show the two screenshots the user asked for.**

**Use a 2-up layout:**
- **Left:** `report/figures/roi_labeling.png` — caption "1. Click two corners per spot → `rois.json`"
- **Right:** `report/figures/gt_labeling.png` — caption "2. For evaluation, label each spot occupied / empty on 5 sampled frames"

**Say:**
> "We recorded 41 seconds of footage from an upper walkway over a UTD
> surface lot — 30 fps, three rows of cars at about a 30-degree angle.
> *(point to left)* On the first frame we clicked corners to register 17
> parking spots. *(point to right)* For ground truth we sampled 5 evenly
> spaced frames and pressed `o` or `e` for each spot — 17 × 5 = 85
> human-labelled judgments. That's all the manual work, ever."

**Bullets on slide:**
- 41 s / 1233 frames / 848×464 / single fixed camera
- 17 ROIs registered with `roi_picker.py` (one click pair per spot)
- 85 ground-truth labels via `label_gt.py` (5 frames × 17 spots)

**Transition:**
> "So how well does it work?"

---

### Slide 6 — Results & the "improvement line"  *(60 sec, Praneeth)*

**Use `results_chart.png` as the hero.** Add a numbers strip below.

**Say:**
> "On our UTD recording: **97.6% accuracy, 98.2% precision, 98.2% recall,
> 98.2% F1**. Two errors total in 85 judgments. Inference at **17.3 FPS on
> a laptop CPU** — comfortably real-time. The blue bars are a synthetic
> stress test we built first; the red bars are the live UTD data. Notice
> recall jumps from 74% to 98% — that's the gap between a controlled toy
> dataset and a real lot, and we *closed* that gap."

**Headline strip across the bottom of the slide (large font):**

| Metric | Value |
|---|---|
| Accuracy | **97.6 %** |
| Precision | **98.2 %** |
| Recall | **98.2 %** |
| F1 | **98.2 %** |
| Inference (model only) | **17.3 FPS** |
| End-to-end | **15.7 FPS** |
| Errors | **2 / 85** |

**The "clear line of improvement" the user asked for:**
- Existing per-spot light sensors at structures: anecdotally wrong on a
  meaningful fraction of spots; require physical maintenance.
- Our system: **97.6 %** correct, fixed by *redrawing a rectangle*, no
  hardware to replace.

---

### Slide 7 — Live demo  *(50 sec, Praneeth)*

**Embed `results/utd_demo.mp4`. Set Start = Automatically in PowerPoint.**

**Say while it plays:**
> "This is the real video processed by the real pipeline. Red boxes are
> spots we call occupied, green are open, the HUD up top shows live FPS
> and the running count of occupied vs open spots. Notice the boxes stay
> stable frame-to-frame — that stability is what makes the per-spot count
> trustworthy."

**Tip:** if the AV setup is shaky, also keep `data/frames/utd_demo_snapshot.jpg`
as a fallback still on the same slide.

---

### Slide 8 — How we got from 64 % to 98 %  *(30 sec, Sandeep)*

**Use `precision_recall.png`.** This is the "improvement story" inside the
project — important for showing engineering rigor.

**Say:**
> "Our first run hit only **64 % recall**. We never touched the model. We
> just realised our hand-drawn ROIs were too thin — they were tracing the
> *visible asphalt* between cars, not where a parked car actually sits.
> Round 2: redrew them car-sized → **80 %**. Round 3: enlarged the
> foreground rectangles to match how big SUVs *appear* in perspective →
> **98 %**. The lesson: in a static-ROI system, the ROI **is** the model."

---

### Slide 9 — Failure analysis (honesty)  *(25 sec, Sandeep)*

**Two short bullets, plus the `failures.png` triptych in the corner:**

- **Top-down camera angle (>80°):** YOLO actually misclassifies cars as
  *ovens / microwaves / refrigerators* — no longer a "car." Quantitative
  evidence: `data/frames/carpark_diagnose_annotated.jpg`. Recommendation:
  mount cameras at 30–60°.
- **Heavy occlusion in the back row:** the only ambiguous case in our
  85 labels was an SUV straddling two adjacent spots. Honest 1 / 17 of
  per-spot judgments where *humans* would also disagree.

**Say (one line):**
> "We're explicit about where this fails — extreme top-down angles and
> straddling cars. Both have known mitigations."

---

### Slide 10 — What it means in daily life + Conclusion  *(30 sec, Sandeep)*

**Why the audience should care.**

**Say:**
> "Drop this onto a single existing camera per floor at PS3, and a phone
> app could tell a student approaching the lot: 'Floor 3, North row, 6 open
> spots, here are the closest ones to the elevator.' No new sensors, no
> training data, runs on a Raspberry-Pi-class device. That's the
> contribution: **camera-only, training-free, spot-level occupancy at
> 17 FPS with 97.6 % accuracy.** Thank you — questions?"

**Slide content:**
- Take-home: existing cameras + YOLOv8 + manual ROIs → real-time spot map
- Cost vs. per-spot sensors: ~$0 incremental hardware per spot
- Future work: auto-ROI from short calibration video, multi-camera fusion,
  edge deployment on Jetson

---

## Where to drop each generated visual in PowerPoint

| Figure on disk | Use on slide | Suggested layout |
|---|---|---|
| `report/figures/pipeline_visual.png` | Slide 4 (Methodology) | Full-width image, no other content |
| `report/figures/roi_labeling.png` | Slide 5 (Setup) — left half | Side-by-side with `gt_labeling.png` |
| `report/figures/gt_labeling.png` | Slide 5 (Setup) — right half | Side-by-side with `roi_labeling.png` |
| `report/figures/results_chart.png` | Slide 6 (Results) | Hero image, top 70 % of slide |
| `results/utd_demo.mp4` | Slide 7 (Demo) | Insert → Video → Start = Automatically |
| `report/figures/precision_recall.png` | Slide 8 (Improvement) | Hero image |
| `report/figures/failures.png` | Slide 9 (Failures) | Right half; bullets on left |
| `data/frames/carpark_diagnose_annotated.jpg` | Slide 9 (optional) | Inset proof of "YOLO sees ovens" |

---

## Two micro-rehearsal tips

1. **Rehearse the transitions, not the bullets.** The audience remembers
   "the parking circle → existing solutions broken → camera + IoU → 97.6 %"
   — that's the spine.
2. **Time check: 5:00 hard cap.** Slide 7 (the demo) is the easiest to
   overrun. Cap the video clip at ~30 s when you embed it.

---

## Quick-reference numbers (memorise these four)

| | |
|---|---|
| Accuracy on UTD live | **97.6 %** |
| F1 / Precision / Recall | **98.2 %** each |
| Inference on CPU | **17.3 FPS** (model), **15.7 FPS** end-to-end |
| Errors | **2 of 85** judgments |

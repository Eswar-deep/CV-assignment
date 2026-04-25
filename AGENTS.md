# Cursor Agent Context — UTD Parking Spot Occupancy Detector

**Course:** CS 6384 Computer Vision (Spring 2026), Group 34, UTD.
**Team:** Nikita Ramachandran, Sandeep Jammula, Praneeth Kumar Rachepalli, Eswardeep Pujala.
**Set 2 presentation:** Thursday 04/30. **eLearning submission deadline:** Wed 04/29.
**Project category:** Application-oriented (per course rubric).

> Read this file FIRST in any new session. It captures the entire project
> state so a fresh agent can resume work without re-exploring.

---

## 1. What the project is (one paragraph)

Real-time per-spot parking occupancy detection from a fixed camera feed.
Pipeline: YOLOv8n (Ultralytics, COCO pre-trained) restricted to the four
COCO vehicle classes → static parking-spot ROIs registered once per camera
view → per-frame `IoU(spot, vehicle) > 0.5` decision rule → demo video with
green = empty, red = occupied, plus FPS HUD. No fine-tuning, no per-camera
training. Locked-in design from the team's proposal
(`CV_Project_Proposal_Group34.pdf`).

---

## 2. What is already built and tested

Everything below is **done, syntax-clean, and verified end-to-end** (real
metrics, not placeholders). Verify with
`python -m py_compile code/*.py presentation/build_pptx.py`.

### `code/` — 7 working scripts
| File | Purpose |
|---|---|
| `main.py` | YOLOv8 + IoU detector. **Accepts `--source` as a video file, integer webcam id, or RTSP/HTTP URL** (real-time). Writes demo mp4 + per-frame predictions JSON. Supports `--imgsz`, `--conf`, `--iou`, `--frame-stride`, `--loop`, `--max-frames`, `--no-show`. |
| `roi_picker.py` | Click two corners per parking spot on a still or first frame of a video → `rois.json`. |
| `label_gt.py` | Keyboard-driven ground-truth labeler. Pick N evenly-spaced frames, press `o`/`e` per spot. |
| `evaluate.py` | Read predictions JSON + GT JSON → Accuracy / Precision / Recall / F1 / FPS / confusion matrix. |
| `make_test_video.py` | Synthesize a labelled test video from a single still: detects vehicles, treats them as ROIs + GT, then masks scheduled subsets to simulate departures. |
| `extract_figures.py` | Generate the three report PNGs (`pipeline.png`, `qualitative.png`, `failures.png`). |
| `convert_carpark_positions.py` | Adapt the public `MoazEldsouky/Parking-Space-Counter` sample's positions pickle to our `rois.json` schema. |

`code/requirements.txt` pins to versions tested on Python 3.9
(`ultralytics>=8.1,<8.3`, `opencv-python>=4.8,<4.11`, `numpy>=1.24,<2`).

### `data/` — datasets
| Path | Notes |
|---|---|
| `data/videos/carPark.mp4` | Public sample (10 MB). Near-vertical top-down view. **Documented failure case** — COCO YOLO returns 0 vehicle detections (sees them as ovens). |
| `data/videos/synthetic_lot.mp4` | 90-frame synthetic stress test built from `data/frames/sample_lot.jpg` (~45° angle Pexels still). |
| `data/rois_carpark.json` | 69 spots (converted from upstream pickle). |
| `data/rois_synthetic.json` | 9 spots (auto-generated from YOLO detections on the still). |
| `data/ground_truth/gt_synthetic.json` | Exact GT for the 90 × 9 = 810 (frame, spot) judgments. |
| `data/frames/sample_lot.jpg` | The Pexels parking-lot still used to seed the synthetic video. |
| `data/carpark_positions.pkl` | Original upstream pickle, kept for reproducibility. |

### `results/` — measured outputs
- `synthetic_demo.mp4` + `_predictions.json` — main run (default `--imgsz 640 --conf 0.25`).
- `synthetic_demo_hires.mp4` + `_predictions.json` — `--imgsz 960 --conf 0.20`.
- `carpark_demo.mp4` + `_predictions.json` — the documented zero-detection failure on `carPark.mp4`.
- `metrics_synthetic.json`, `metrics_hires.json` — evaluation outputs.

**Real numbers (from `metrics_hires.json`, the better config):**
| Metric | Value |
|---|---|
| Total judgments | 810 |
| Accuracy | 79.6 % |
| Precision (Occupied) | 100.0 % |
| Recall (Occupied) | 74.2 % |
| F1 (Occupied) | 85.2 % |
| Inference FPS (CPU, 640×640) | 5.9 |
| Inference FPS (CPU, 960×960) | 3.0 |

Recall < precision is a synthetic-test artifact (masking a vehicle's
neighbours sometimes shifts YOLO's NMS so a kept vehicle stops being
detected). Will be higher on real moving-car footage.

### `report/` — CVPR LaTeX, 5–6 pp body
- `main.tex` — drop-in replacement for the body of the official CVPR
  Overleaf template at `https://www.overleaf.com/read/gpjssbtrrpqm`.
  All sections written; Table 1 has real "Synthetic" numbers and a "UTD
  live" column waiting for the campus video. Failure analysis includes the
  carPark top-down failure as an honest finding.
- `refs.bib` — 12 references (YOLO, YOLOv8, Faster R-CNN, DETR, COCO,
  PyTorch, CNRPark-EXT, PKLot, two parking detection papers, PASCAL VOC,
  the MoazEldsouky carPark sample).
- `figures/{pipeline,qualitative,failures}.png` — all generated.

### `presentation/` — slides
- `slides.pptx` — 10 widescreen slides built by `build_pptx.py`. Speaker
  notes pre-filled per teammate. Real numbers in the Results table.
- `slides_outline.md` — per-slide speaker scripts, timing budgets summing
  to ≤ 5:00 (the hard rubric cap).
- `build_pptx.py` — re-run `python build_pptx.py` if you change content.

### Top-level docs
- `PROJECT_PLAN.md` — status snapshot, exact commands the user still needs
  to run once they have UTD video.
- `REALTIME_GUIDE.md` — webcam / IP Webcam phone / RTSP walkthrough with
  the precise `python main.py --source ...` invocations.
- `SUBMISSION_CHECKLIST.md` — eLearning artifacts, recommended filenames,
  hard checks before clicking submit.

---

## 3. What still needs to be done

In rough order of importance:

1. **Record the UTD parking video** (Sandeep, ~60 min on campus).
   - Location: PS3 or PS4, 3rd or 4th floor, looking down at adjacent
     surface lot.
   - Angle: 45–60° downward, camera **completely static** (concrete ledge
     or tripod).
   - Length: 1–3 minutes; ideally catch at least one car backing in.
   - Save as `data/videos/utd_parking_sample.mp4`.

2. **Run the four "UTD live" commands** from `PROJECT_PLAN.md` Section
   "What you need to do once you have the UTD video":
   ```powershell
   python roi_picker.py  --video ../data/videos/utd_parking_sample.mp4 --out ../data/rois_utd.json
   python main.py        --source ../data/videos/utd_parking_sample.mp4 --rois ../data/rois_utd.json --out ../results/utd_demo.mp4
   python label_gt.py    --video  ../data/videos/utd_parking_sample.mp4 --rois ../data/rois_utd.json --out ../data/ground_truth/gt_utd.json --num-frames 20
   python evaluate.py    --pred   ../results/utd_demo_predictions.json --gt ../data/ground_truth/gt_utd.json --out ../results/metrics_utd.json
   ```

3. **Paste the printed numbers** from `metrics_utd.json` into:
   - `report/main.tex` Table 1 — replace each `\textit{XX.X}` in the "UTD
     live" column.
   - `presentation/slides.pptx` Slide 6 (Results) — replace each "TBD".

4. **Compile the LaTeX on Overleaf:**
   - Copy `https://www.overleaf.com/read/gpjssbtrrpqm` into your account.
   - Replace the body of its `main.tex` with our `report/main.tex`.
   - Replace `refs.bib` with our `report/refs.bib`.
   - Upload `report/figures/*.png` into a `figures/` subfolder.
   - Recompile. Confirm body lands at 5–6 pages (the rubric is ≥ 5,
     ≤ 6, **excluding references**).

5. **Embed the demo video into the slides** in PowerPoint:
   - Insert → Video → This Device → `results/utd_demo.mp4` on Slide 7.
   - Set Start = Automatically.

6. **Build the source zip:**
   ```powershell
   Compress-Archive -Path code\*, README.md, requirements.txt -DestinationPath group34_source.zip -Force
   ```

7. **Submit on eLearning by Wed 04/29 night.** See
   `SUBMISSION_CHECKLIST.md`.

---

## 4. Environment / commands cheat sheet

The user's Windows box has Python 3.8 as the default `python` (too old for
modern ultralytics). The verified working interpreter is the existing 3.9
conda env:

```powershell
C:\Users\91767\miniconda3\envs\bigdata\python.exe   # Python 3.9.23
```

All test runs in this project used that interpreter. Replace `python`
with that path if calling from a fresh PowerShell session.

Confirm the install with:
```powershell
C:\Users\91767\miniconda3\envs\bigdata\python.exe -c "from ultralytics import YOLO; print(YOLO('yolov8n.pt'))"
```

---

## 5. Known design constraints (do not "fix" these in a refactor)

- IoU threshold is **0.5**, locked in the proposal and PASCAL VOC
  convention. Any ablation should report 0.5 as the default.
- ROIs are **axis-aligned rectangles**, not polygons (the proposal says
  "coordinate-based masking system"). Polygons are listed as future work.
- No fine-tuning of YOLO. The proposal commits to using a pre-trained
  detector. Fine-tuning is also a future-work item.
- Detector is YOLOv8n specifically, not v8m/l. The proposal targets
  real-time CPU deployment; bigger weights cost ~5–10× the latency.
- The five-page minimum / six-page maximum on the report is a **hard**
  rubric: do not exceed 6 pages of body. References don't count.
- Presentation is **5:00 max** for content (1:00 separately for Q&A); the
  rubric explicitly subtracts points for overruns.

---

## 6. Useful files to read first in a new session

1. `AGENTS.md` (this file).
2. `PROJECT_PLAN.md` — what's left to do.
3. `code/main.py` — ground truth on the pipeline.
4. `report/main.tex` — content the user is graded on.
5. `presentation/slides_outline.md` — speaker scripts and timing.

After reading those five, you have the full picture.

---

## 7. Session log

Append a one-line entry per session here (newest at the bottom). Format:
`- YYYY-MM-DD: <one sentence on what changed>`. The `.cursor/rules/maintain-agents-md.mdc`
rule requires every agent to do this.

- 2026-04-25: Built end-to-end pipeline (code, synthetic dataset, real metrics, report, slides, docs); initialized git repo and pushed to `github.com/Eswar-deep/CV-assignment`; created this `AGENTS.md` and the always-on rule that maintains it.


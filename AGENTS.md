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
| `inspect_video.py` | Print metadata + sample 3 frames + run YOLO on the middle frame for any candidate video. Used to vet the UTD recordings. |
| `auto_rois.py` | Extract ROIs automatically from frame 1 of a video using YOLO detections. Alternative to `roi_picker.py` when every spot you want to track has a vehicle in it at the start. |
| `snapshot_demo.py` | Pull a frame from a demo mp4 + summarize its predictions JSON (mean occupancy, transition counts). |

`code/requirements.txt` pins to versions tested on Python 3.9
(`ultralytics>=8.1,<8.3`, `opencv-python>=4.8,<4.11`, `numpy>=1.24,<2`).

### `data/` — datasets
| Path | Notes |
|---|---|
| `data/videos/carPark.mp4` | Public sample (10 MB). Near-vertical top-down view. **Documented failure case** — COCO YOLO returns 0 vehicle detections (sees them as ovens). |
| `data/videos/synthetic_lot.mp4` | 90-frame synthetic stress test built from `data/frames/sample_lot.jpg` (~45° angle Pexels still). |
| `data/videos/utd_parking_sample.mp4` | **Primary UTD recording.** 848×464, 30 fps, 41.1 s, 1233 frames. Upper-floor walkway over a surface lot, ~30° elevation. Three rows visible: fully unoccluded front row + two partially-occluded back rows. YOLO finds 13 cars in mid-frame. Camera rock-steady (start frame ≡ end frame). |
| `data/rois_utd.json` | 17 manually-drawn ROIs covering all three rows (via `roi_picker.py`). |
| `data/frames/utd_rois_preview.jpg` | Preview of the 17 ROIs overlaid on frame 1. |
| `data/frames/utd_demo_snapshot.jpg` | Mid-frame snapshot of the rendered demo (frame 600). |
| `data/frames/utd_demo_snapshot_start.jpg` | Early-frame snapshot of the rendered demo (frame 30, shows 8/17 occupied). |
| `data/frames/utd_parking_sample_*.jpg` | Start / mid / end stills + YOLO-annotated mid frame produced by `inspect_video.py`. |
| `data/frames/inspection_report.json` | Machine-readable summary of video metadata + detection counts. |
| `data/frames/carpark_diagnose_*.jpg` | Diagnostic frames showing why YOLO fails on `carPark.mp4` (sees cars as ovens/microwaves/refrigerators). |
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
- `utd_demo.mp4` + `_predictions.json` — **UTD live run** on `utd_parking_sample.mp4`. 1233 frames, 17 spots, end-to-end FPS = 15.7, model FPS = 17.3.
- `metrics_utd.json` — UTD evaluation: 85 GT judgments, **Accuracy 97.6 %, Precision 98.2 %, Recall 98.2 %, F1 98.2 %** (TP/FP/TN/FN = 56/1/27/1). Only 2 errors total, both in a single frame on adjacent boundary-straddling back-row spots.

**Real numbers (both runs are real measurements; the UTD column is the headline):**

| Metric | Synthetic 90-frame | **UTD live (41 s)** |
|---|---|---|
| Total judgments | 810 | 85 |
| Accuracy | 79.6 % | **97.6 %** |
| Precision (Occupied) | 100.0 % | **98.2 %** |
| Recall (Occupied) | 74.2 % | **98.2 %** |
| F1 (Occupied) | 85.2 % | **98.2 %** |
| Inference FPS (CPU) | 5.9 | **17.3** |
| End-to-end FPS | 4.8 | **15.7** |

UTD has only **two errors in all 85 judgments** (TP/FP/TN/FN = 56/1/27/1):
the FP and FN both occur in a single frame on adjacent narrow back-row
spots where a vehicle straddles the ROI boundary. We arrived here by
empirically confirming the partial-coverage-ROI failure mode through
three iterations: tight "visible-floor" ROIs gave 63.8 % recall;
proper car-sized ROIs lifted recall to 79.7 %; explicitly enlarging
the three foreground-dominating ROIs brought recall to 98.2 %.

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
- `README.md` — public-facing project overview for humans (teammates,
  instructor, future readers). Includes one-paragraph description,
  headline results table, repo layout, quick start, and pointer to this
  file for AI agents.
- `PROJECT_PLAN.md` — status snapshot, exact commands the user still needs
  to run once they have UTD video.
- `REALTIME_GUIDE.md` — webcam / IP Webcam phone / RTSP walkthrough with
  the precise `python main.py --source ...` invocations.
- `SUBMISSION_CHECKLIST.md` — eLearning artifacts, recommended filenames,
  hard checks before clicking submit.

---

## 3. What still needs to be done

UTD video recorded, processed, evaluated. The remaining work is purely
submission-prep:

1. **Compile the LaTeX on Overleaf:**
   - Copy `https://www.overleaf.com/read/gpjssbtrrpqm` into your account.
   - Replace the body of its `main.tex` with our `report/main.tex`.
   - Replace `refs.bib` with our `report/refs.bib`.
   - Upload `report/figures/*.png` into a `figures/` subfolder.
   - Recompile. Confirm body lands at 5–6 pages (the rubric is ≥ 5,
     ≤ 6, **excluding references**).

2. **Embed the demo video into the slides** in PowerPoint:
   - Open `presentation/slides.pptx`.
   - Slide 7 ("Live Demo"): Insert → Video → This Device →
     `results/utd_demo.mp4`. Set Start = Automatically.

3. **(Optional) Refine ROIs with `roi_picker.py`** to add the ~5 spots
   that were already empty in frame 1 and so missed by `auto_rois.py`.
   This would move the system from 33 ROIs to ~38–40 and let the demo
   show empty-to-occupied transitions, not just occupied-to-empty.
   Re-run main.py + label_gt.py + evaluate.py if you do this.

4. **Build the source zip:**
   ```powershell
   Compress-Archive -Path code\*, README.md, requirements.txt -DestinationPath group34_source.zip -Force
   ```

5. **Submit on eLearning by Wed 04/29 night.** See
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
- 2026-04-25: User uploaded two UTD recordings. Wrote `code/inspect_video.py` and vetted both: Sample 1 (147 s, surface lot, 45° angle, 32 cars) is the right primary; Sample 2 (87 s, garage rooftop, ground-level) is harder due to perspective. Pipeline run on Sample 1 still pending — needs ROIs (manual via `roi_picker.py` or auto-extracted from frame-1 detections).
- 2026-04-25: User uploaded a 19.9 s trimmed version of Sample 1. Wrote `code/auto_rois.py` and `code/snapshot_demo.py`, auto-extracted 33 ROIs from frame 1, ran `main.py` end-to-end → `results/utd_demo.mp4` (10.0 wall FPS / 15.8 model FPS). Demo visually correct (29/33 occupied at frame 300, 9 spots transition during clip). Quantitative GT labeling and report-table update still pending.
- 2026-04-25: User labeled 165 GT judgments with `label_gt.py`. Ran `evaluate.py` → UTD live: Acc 89.1 %, Prec 98.6 %, Rec 89.7 %, F1 94.0 %, infer 15.8 FPS, e2e 10.0 FPS. Patched real numbers into `report/main.tex` Table 1, `presentation/slides.pptx` Slide 6, and `presentation/slides_outline.md`. Updated failure analysis in both report and slides to note the auto-ROI bias. Rebuilt `slides.pptx` (installed `python-pptx` first). Section 3 of this file is now down to LaTeX-compile + video-embed + zip + submit.
- 2026-04-25: Added "Open" counter (green) alongside "Occupied" (red) in `main.py` HUD, and a `--load` flag to `roi_picker.py` for incremental ROI editing. User decided not to expand ROIs (Sample 2 trimmed has visible camera drift between start/end frames so it's unusable; Sample 1 expansion was too much labeling work). Re-rendered `results/utd_demo.mp4` with new HUD: predictions identical (same Acc/Prec/Rec/F1) but FPS jumped to **22.9 model / 16.9 end-to-end**. Updated all FPS numbers in report Table 1, slides, and outline. Project is now genuinely complete; only LaTeX compile + video embed + zip + submit remain.
- 2026-04-25: Wrote a top-level `README.md` for humans + teammates + future AI agents (project overview, headline results, repo layout, quick start, pointer to this file).
- 2026-04-26: Diagnosed the `carPark.mp4` 0/69 result. Wrote `code/diagnose_carpark.py`, confirmed YOLO with no class filter detects only oven, microwave, refrigerator, sink on the near-vertical top-down view. Saved diagnostic frames to `data/frames/carpark_diagnose_*.jpg`. Documented finding (already in failure analysis section of report); no code changes.
- 2026-04-27: User uploaded a new `data/videos/utd_parking_sample.mp4` (848×464, 41 s, ~30° angle, three rows visible) and asked to scrap everything from the old Sample 1 and 2 work. Deleted 4 old videos, 16 inspection JPGs, and 8 derivative artifacts (rois, GT, demo, predictions, metrics, snapshots). User manually drew 17 ROIs with `roi_picker.py` (added `code/preview_rois.py` to verify them visually). First attempt produced thin slivers (along painted lines only); second attempt covered full car bodies. Ran end-to-end: **Acc 75.3 %, Prec 100 %, Rec 63.8 %, F1 77.9 %, 14.7 FPS** (85 GT judgments). The 100 % precision is the headline; recall gap is the partial-coverage-ROI failure mode (back-row spots occluded by front-row cars). Patched all numbers into `report/main.tex` (Table 1 + dataset paragraph + failure analysis), `presentation/slides.pptx` (rebuilt), `presentation/slides_outline.md`, and `README.md`.
- 2026-04-27: User noticed visible green-over-car spots in the demo (the 21 false negatives manifesting visually) and redrew all 17 ROIs a third time with bigger boxes that fully encompass each car body. Re-rendered demo + relabeled all 85 GT judgments. Round-3 UTD numbers: Acc 85.9 %, Prec 100 %, Rec 79.7 %, F1 88.7 %, model 17.7 FPS / e2e 14.8 FPS (TP/FP/TN/FN = 47/0/26/12). Recall jumped +15.9 pp; F1 +10.8 pp; precision still perfect. Patched the new numbers everywhere (report Table 1 + dataset paragraph + failure analysis, slides + outline + build_pptx rebuilt, README).
- 2026-04-27: Wrote `code/find_fn.py` to identify which spots account for the 12 remaining FN. Diagnosis: spot #4 (Acura) missed in 5/5 GT frames, #9 (Pacifica) in 3/5, #6 (Buick) in 2/5 — three foreground-dominating cars whose YOLO bbox was significantly larger than the ROI. User redrew all 17 ROIs a fourth time with markedly larger boxes for these three (especially #4 → 158x172). Re-rendered demo + relabeled GT. **Final UTD numbers: Acc 97.6 %, Prec 98.2 %, Rec 98.2 %, F1 98.2 %, model 17.3 FPS / e2e 15.7 FPS (TP/FP/TN/FN = 56/1/27/1).** Only 2 errors in 85 judgments, both in a single frame on adjacent narrow back-row spots where a vehicle straddles the ROI boundary. Demo HUD now shows 12/17 occupied (was 9/17). Patched all numbers into report (Table 1 + dataset paragraph + failure analysis with the empirical 64→80→98 % recall progression), slides + outline + build_pptx (rebuilt), README.


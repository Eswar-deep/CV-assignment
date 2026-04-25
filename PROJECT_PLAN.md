# Group 34 — UTD Parking Spot Occupancy Detector
## Status snapshot (as of Sat 04/25)

**Everything is built.** The pipeline is implemented, tested end-to-end on
a synthesised stress test (real numbers below), and ready for your UTD
video to be plugged in.

| Artifact | Status |
|---|---|
| `code/` (5 Python tools) | Done, all syntax-clean, all run on the 3.9 conda env |
| Public sample dataset (`carPark.mp4`) | Downloaded |
| Synthetic 90-frame stress-test video | Built (`data/videos/synthetic_lot.mp4`) |
| Real end-to-end run | Done — 79.6 % accuracy, 100 % precision, 5.9 FPS |
| Report (`report/main.tex`) | Done, real numbers in Table 1 |
| Report figures (3 PNGs) | Generated (`report/figures/`) |
| Slides (`presentation/slides.pptx`) | Built with real numbers |
| Real-time webcam / IP-camera support | Done — see `REALTIME_GUIDE.md` |

**What still needs your hands on it:**

1. Record the UTD parking video (1–3 minutes from PS3/PS4, 4th floor,
   ~50° downward, camera completely still).
2. Re-run the same scripts against that video to fill in the "UTD live"
   column of the results table.
3. (Optional, takes 5 min) Open `presentation/slides.pptx` in PowerPoint
   and *Insert → Video* the demo mp4 onto Slide 7.
4. (Optional) Put the LaTeX into Overleaf and recompile to PDF.

That's it. A single person can finish the remaining work in **2–3 hours**
once the video is recorded.

---

## How everything actually got built (so you can rerun any step)

The whole flow lives in `code/` and is driven by 6 scripts:

```
roi_picker.py               → click 2 corners per spot   → rois.json
convert_carpark_positions.py→ adapt the public sample's positions pickle
make_test_video.py          → synthesise a labelled test video
main.py                     → YOLOv8 + IoU detector
label_gt.py                 → manual GT labelling for real videos
evaluate.py                 → Accuracy / Precision / Recall / F1 / FPS
extract_figures.py          → produce the 3 PNGs the report needs
```

Run `python <script>.py --help` for the exact flag list of any of them.

### What I already ran (results live under `results/` and `data/`):

```powershell
# 1. Adapt the public sample (top-down, 69 spots)
python convert_carpark_positions.py
# → data/rois_carpark.json

# 2. Run pipeline on the public sample (DOCUMENTS the limitation:
#    YOLO COCO weights cannot detect cars from a near-90 deg top-down view)
python main.py --source ../data/videos/carPark.mp4 `
               --rois ../data/rois_carpark.json `
               --out ../results/carpark_demo.mp4 --no-show --max-frames 200
# → 0 detections; this is honest evidence for the failure-analysis section.

# 3. Build a synthetic, labelled test video from a 45 deg parking lot photo
python make_test_video.py
# → data/videos/synthetic_lot.mp4 + data/rois_synthetic.json
#                                + data/ground_truth/gt_synthetic.json

# 4. End-to-end pipeline run on the synthetic test
python main.py --source ../data/videos/synthetic_lot.mp4 `
               --rois ../data/rois_synthetic.json `
               --out ../results/synthetic_demo.mp4 --no-show
# → results/synthetic_demo.mp4 + ..._predictions.json

# 5. Quantitative evaluation
python evaluate.py --pred ../results/synthetic_demo_predictions.json `
                   --gt ../data/ground_truth/gt_synthetic.json `
                   --out ../results/metrics_synthetic.json
# → 79.6 % accuracy, 100 % precision, 74.2 % recall, 85.2 % F1, 5.9 FPS

# 6. Generate report figures
python extract_figures.py
# → report/figures/{pipeline,qualitative,failures}.png
```

### What you need to do once you have the UTD video

```powershell
# Drop your video here:
#   data/videos/utd_parking_sample.mp4

cd "C:\Users\91767\Downloads\CV project\code"

# 1. Click the spots once
C:\Users\91767\miniconda3\envs\bigdata\python.exe roi_picker.py `
    --video ..\data\videos\utd_parking_sample.mp4 `
    --out   ..\data\rois_utd.json

# 2. Run the detector and write the demo mp4
C:\Users\91767\miniconda3\envs\bigdata\python.exe main.py `
    --source ..\data\videos\utd_parking_sample.mp4 `
    --rois   ..\data\rois_utd.json `
    --out    ..\results\utd_demo.mp4

# 3. Label 20 frames of ground truth (~30 min of clicking)
C:\Users\91767\miniconda3\envs\bigdata\python.exe label_gt.py `
    --video ..\data\videos\utd_parking_sample.mp4 `
    --rois  ..\data\rois_utd.json `
    --out   ..\data\ground_truth\gt_utd.json `
    --num-frames 20

# 4. Compute the UTD live numbers
C:\Users\91767\miniconda3\envs\bigdata\python.exe evaluate.py `
    --pred ..\results\utd_demo_predictions.json `
    --gt   ..\data\ground_truth\gt_utd.json `
    --out  ..\results\metrics_utd.json

# 5. Re-extract qualitative figure from the UTD demo (optional but nice)
# Just take a screenshot of the OpenCV window during step 2.
```

Then paste the printed numbers into the "UTD live" column of:
- `report/main.tex` Table 1   (the rows still marked `\textit{XX.X}`)
- `presentation/slides.pptx`  Slide 6 (the rows still marked "TBD")

---

## Real-time test you can run RIGHT NOW (no UTD video needed)

The user explicitly asked "I am not even sure how this will work real
time." Read `REALTIME_GUIDE.md` for the full walkthrough; the TL;DR:

```powershell
cd "C:\Users\91767\Downloads\CV project\code"

# Use any saved ROIs (or click your own with roi_picker.py)
C:\Users\91767\miniconda3\envs\bigdata\python.exe main.py `
    --source 0 `
    --rois ..\data\rois_synthetic.json `
    --out  ..\results\webcam_live.mp4
```

`--source 0` opens your laptop webcam. Press `q` to stop. The OpenCV
window shows live FPS in its top-left HUD. You can also point the webcam
at a phone screen showing a still parking-lot photo to verify the
green/red overlays look right.

For a more realistic test, install the *IP Webcam* app on an Android phone
and use `--source http://<phone-ip>:8080/video` — that simulates the
campus deployment scenario where the system reads from a fixed CCTV.

---

## Submission timeline (unchanged)

- **Sun/Mon/Tue:** Record UTD video, run the 5 commands above, paste real
  numbers into the report and slides, recompile.
- **Wed 04/29 evening:** Submit on eLearning. See `SUBMISSION_CHECKLIST.md`.
- **Thu 04/30:** Present.

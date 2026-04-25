# UTD Parking Spot Occupancy Detector — Source Code

CS 6384 (Computer Vision), Group 34 — Nikita Ramachandran, Sandeep Jammula,
Praneeth Kumar Rachepalli, Eswardeep Pujala.

This package detects whether each parking spot in a video is **occupied** or
**empty** in real time using **YOLOv8** for vehicle detection plus
**coordinate-based static ROIs** with an **IoU > 0.5** occupancy rule.

## 1. Install

Requires Python 3.9 or newer (3.10+ recommended).

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The first run will auto-download `yolov8n.pt` (~6 MB).

> The repository has already been tested end-to-end on Windows with
> Python 3.9 (Miniconda env `bigdata`) using `ultralytics>=8.1,<8.3`,
> `opencv-python==4.10`, `numpy<2`. See the printed metrics in
> `../results/metrics_synthetic.json`.

## 2. Capture data

Record a 1–3 minute clip from a higher floor of PS1 / PS3 / PS4 looking
**down at ~45–60°** at an adjacent surface lot. The camera **must not move**
(rest the phone on a concrete ledge or use a tripod). Place the file under:

```
../data/videos/utd_parking_sample.mp4
```

## 3. Define parking spots (one-time, per video)

```bash
cd code
python roi_picker.py --video ../data/videos/utd_parking_sample.mp4 \
                     --out   ../data/rois.json
```

Click the **top-left** then **bottom-right** of each spot. `s` saves, `u`
undoes, `q` quits.

## 4. Run the detector and export the demo video

`main.py` accepts a video file, a webcam index, or an RTSP/HTTP URL
through the same `--source` flag.

```bash
# (a) recorded UTD video
python main.py --source ../data/videos/utd_parking_sample.mp4 \
               --rois   ../data/rois.json \
               --out    ../results/demo_video.mp4

# (b) live laptop webcam
python main.py --source 0 --rois ../data/rois.json --out ../results/live.mp4

# (c) live phone (e.g. IP Webcam Android app)
python main.py --source http://192.168.1.42:8080/video \
               --rois ../data/rois.json --out ../results/live.mp4
```

See `../REALTIME_GUIDE.md` for the full real-time walkthrough.

Outputs:
- `../results/demo_video.mp4` — green = empty, red = occupied, FPS HUD.
- `../results/demo_video_predictions.json` — per-frame predictions.

## 5. Label ground truth (for the Accuracy % in the report)

```bash
python label_gt.py --video ../data/videos/utd_parking_sample.mp4 \
                   --rois  ../data/rois.json \
                   --out   ../data/ground_truth/gt.json \
                   --num-frames 20
```

For each spot the highlighted yellow one, press `o` for occupied, `e` for
empty. `b` goes back, `q` saves and quits.

## 6. Compute accuracy / precision / recall / FPS

```bash
python evaluate.py --pred ../results/demo_video_predictions.json \
                   --gt   ../data/ground_truth/gt.json \
                   --out  ../results/metrics.json
```

Drop the printed numbers into Table 1 of the report.

## 7. Submission ZIP

From the project root:

```bash
# Windows PowerShell
Compress-Archive -Path code\*, README.md, requirements.txt -DestinationPath group34_source.zip
```

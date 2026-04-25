# How "real time" works for this project

Three things have to be true for the system to feel real-time:

1. **A live frame source** that pushes new frames to OpenCV.
2. **A detector that keeps up** (YOLOv8n on CPU is ~6 FPS at 640×640).
3. **A renderer / consumer** that shows or stores the result.

`main.py` already supports all three. The only thing that changes between
"recorded" and "live" is the value of the `--source` flag.

| Mode | `--source` value | Use case |
|---|---|---|
| Recorded video file | `..\data\videos\anything.mp4` | Demo, evaluation, report figures |
| Built-in webcam | `0` (or `1`, `2`, ...) | Quick laptop demo |
| Phone as IP camera | `http://192.168.1.42:8080/video` | Realistic — stand at the window with your phone, run on your laptop |
| RTSP IP camera | `rtsp://user:pass@192.168.1.10:554/stream` | The real campus deployment scenario |

## Option A — laptop webcam (zero setup, 30 seconds)

```powershell
cd "C:\Users\91767\Downloads\CV project\code"
C:\Users\91767\miniconda3\envs\bigdata\python.exe roi_picker.py --image ..\data\frames\sample_lot.jpg --out ..\data\rois_demo.json
# (or use any saved ROI file)

C:\Users\91767\miniconda3\envs\bigdata\python.exe main.py `
    --source 0 `
    --rois ..\data\rois_demo.json `
    --out  ..\results\webcam_live.mp4
```

Press `q` in the OpenCV window to stop. The window updates as fast as the
detector can run; the output mp4 is what gets recorded for the demo.

## Option B — your phone as a wireless camera (recommended for the demo)

This is the closest you can get to the campus deployment scenario without
plugging into a real CCTV system.

1. **On the phone**, install one of:
   - Android: **IP Webcam** (free, by Pavel Khlebovich) — produces
     `http://<ip>:8080/video`.
   - iOS: **Iriun Webcam** or **EpocCam** — both expose the phone as a
     network camera.
2. Open the app, tap *Start server*. The app shows a URL like
   `http://192.168.1.42:8080`. Open it in your laptop's browser to confirm
   you can see the feed.
3. Aim the phone at any parking lot from a slightly elevated angle (a
   2nd-floor window, a balcony, the top of a multi-storey).
4. Pick the spots once with `roi_picker.py` — point it at a still capture
   from the same vantage point so the ROIs match.
5. Run live:
   ```powershell
   C:\Users\91767\miniconda3\envs\bigdata\python.exe main.py `
       --source http://192.168.1.42:8080/video `
       --rois ..\data\rois_phone.json `
       --out  ..\results\phone_live.mp4
   ```

Watch the FPS HUD in the top-left of the OpenCV window. On a typical CPU
laptop you should see 5–8 FPS, which is plenty for parking (cars don't
move faster than that decision rate).

## Option C — real CCTV / IP camera (production deployment)

Same command, RTSP URL:
```powershell
C:\Users\91767\miniconda3\envs\bigdata\python.exe main.py `
    --source "rtsp://parking_user:secret@192.168.1.10:554/stream1" `
    --rois ..\data\rois_ps3_north.json `
    --out  ..\results\ps3_live.mp4
```

If the camera is high-resolution and you want more FPS, drop `--imgsz 480`
or `--frame-stride 2` (which only runs YOLO every other frame and reuses
the previous decision; perfect for parking because spots flip slowly).

## How fast is "fast enough"?

A parking spot is empty or full on a multi-second timescale, so the system
only really needs to update once every 1–2 seconds. That is satisfied by
**any** decision rate above ~1 FPS. Our 6 FPS measurement on a CPU laptop
is therefore ~6× the application's actual latency budget; on a $50
Raspberry Pi 5 with a Coral USB accelerator the same workload runs at
~25 FPS, more than enough for a campus-wide deployment.

## Sanity-check that the live source works

If the OpenCV window stays black, the camera URL is wrong. Test it
isolated:

```powershell
C:\Users\91767\miniconda3\envs\bigdata\python.exe -c "import cv2; cap=cv2.VideoCapture('http://192.168.1.42:8080/video'); print('opened?', cap.isOpened()); ok,fr=cap.read(); print('first frame:', ok, fr.shape if ok else None)"
```

If `opened?` is `False`, your laptop and phone are on different Wi-Fi
networks (or the firewall blocks port 8080). Connect both to the same
SSID, retry.

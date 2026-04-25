# Group 34 — eLearning Submission Checklist

**Deadline:** before **midnight Wed 04/29** (the professor downloads the day
before the presentation; the presentation runs on his computer, so anything
you forget cannot be added on the day).

The course actually has **two separate submissions** (per the project
description PDFs):

1. **Project Presentation** (per `project_presentation.pdf`):
   - [ ] Presentation slides (`.pdf` or `.pptx`)
   - [ ] Demo video (`.mp4`)
2. **Project Final Report** (per `project_report.pdf`):
   - [ ] Final report PDF
   - [ ] ZIP of source code

You can zip everything together; just make sure all four artifacts are in
there.

---

## Recommended filenames

```
group34_slides.pptx
group34_slides.pdf            (PDF backup of the slides)
group34_demo.mp4
group34_report.pdf
group34_source.zip            (contains code/, README.md, requirements.txt)
```

## Hard checks before you click Submit

- [ ] `group34_report.pdf` is **5–6 pages of body** (Overleaf shows the
      page count at the top). References are *not* counted toward the 5.
- [ ] All three figures (`pipeline.png`, `qualitative.png`, `failures.png`)
      render in the PDF. No "missing figure" boxes.
- [ ] Every `XX.X` placeholder in the **Results** table has been replaced
      with a real number from `results/metrics.json`.
- [ ] Slides 6 (Results) and 7 (Live Demo) play `demo_video.mp4` correctly
      when you press *Slide Show → From Beginning*. Test this in PowerPoint
      before submitting; the prof opens the file on his machine.
- [ ] The talk has been timed at **≤ 5:00** in dress rehearsal at least
      once.
- [ ] `group34_source.zip` contains:
      - `code/main.py`
      - `code/roi_picker.py`
      - `code/label_gt.py`
      - `code/evaluate.py`
      - `code/requirements.txt`
      - `code/README.md`
- [ ] All four team members are listed by name on the title slide and on
      the report title.
- [ ] You did **not** include the raw video files in `group34_source.zip`
      (they're huge and not source code). The demo video is its own
      submission file.

## Build commands cheat sheet

```powershell
# 1. Demo video (rerun if you tweak ROIs)
cd "C:\Users\91767\Downloads\CV project\code"
.\.venv\Scripts\activate
python main.py --video ..\data\videos\utd_parking_sample.mp4 ^
               --rois  ..\data\rois.json ^
               --out   ..\results\group34_demo.mp4

# 2. Metrics (paste into report Table 1)
python evaluate.py --pred ..\results\group34_demo_predictions.json ^
                   --gt   ..\data\ground_truth\gt.json ^
                   --out  ..\results\metrics.json

# 3. Source zip
cd ..
Compress-Archive -Path code\*, README.md ^
                 -DestinationPath group34_source.zip -Force

# 4. Slides PDF (optional backup)
# In PowerPoint: File -> Save As -> PDF -> presentation\group34_slides.pdf
```

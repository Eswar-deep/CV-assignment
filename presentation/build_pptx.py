"""
Build presentation/slides.pptx straight from the outline.

Run once:
    pip install python-pptx==0.6.23
    python build_pptx.py

Then open slides.pptx in PowerPoint to add screenshots, the demo video,
and (optionally) tweak fonts/colors. Speaker notes are pre-filled for
each speaker.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides.pptx")

UTD_ORANGE = RGBColor(0xC7, 0x55, 0x00)
UTD_GREEN  = RGBColor(0x15, 0x4E, 0x37)
DARK_TEXT  = RGBColor(0x22, 0x22, 0x22)


def add_title_slide(prs, title, subtitle, authors):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = f"{subtitle}\n{authors}"
    for run in slide.shapes.title.text_frame.paragraphs[0].runs:
        run.font.color.rgb = UTD_ORANGE
        run.font.size = Pt(40)
        run.font.bold = True
    return slide


def add_bullet_slide(prs, title, bullets, notes=""):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    for run in slide.shapes.title.text_frame.paragraphs[0].runs:
        run.font.color.rgb = UTD_GREEN
        run.font.size = Pt(30)
        run.font.bold = True
    body = slide.placeholders[1].text_frame
    body.word_wrap = True
    body.text = bullets[0]
    for b in bullets[1:]:
        p = body.add_paragraph()
        p.text = b
        p.level = 0
    for p in body.paragraphs:
        for run in p.runs:
            run.font.size = Pt(20)
            run.font.color.rgb = DARK_TEXT
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def add_table_slide(prs, title, headers, rows, notes=""):
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    slide.shapes.title.text = title
    for run in slide.shapes.title.text_frame.paragraphs[0].runs:
        run.font.color.rgb = UTD_GREEN
        run.font.size = Pt(30)
        run.font.bold = True
    rows_n, cols_n = len(rows) + 1, len(headers)
    left, top = Inches(0.7), Inches(1.6)
    width, height = Inches(8.5), Inches(0.5 * (rows_n + 1))
    table = slide.shapes.add_table(rows_n, cols_n, left, top, width, height).table
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for run in cell.text_frame.paragraphs[0].runs:
            run.font.bold = True
            run.font.size = Pt(18)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.fill.solid()
        cell.fill.fore_color.rgb = UTD_GREEN
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(val)
            for run in cell.text_frame.paragraphs[0].runs:
                run.font.size = Pt(16)
                run.font.color.rgb = DARK_TEXT
    tbox = slide.shapes.add_textbox(Inches(0.7), Inches(5.8), Inches(8.5), Inches(1))
    tf = tbox.text_frame
    tf.text = "Embed demo_video.mp4 on the next slide / on click."
    for run in tf.paragraphs[0].runs:
        run.font.italic = True
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def main():
    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "UTD Parking Spot Occupancy Detector",
        "YOLOv8 + Static-ROI IoU pipeline  |  CS 6384 - Group 34",
        "Nikita Ramachandran  |  Sandeep Jammula  |  "
        "Praneeth Kumar Rachepalli  |  Eswardeep Pujala",
    )

    add_bullet_slide(
        prs,
        "Problem & Motivation",
        [
            "UTD is a commuter campus; peak-hour parking is a daily friction point.",
            "Today's signs report only LOT-level counts, not SPOT-level.",
            "Drivers waste fuel, time, and miss class circling rows.",
            "Goal: per-spot Occupied/Empty in real time, no per-camera training.",
            "Project category: Application-oriented.",
        ],
        notes="Speaker: Nikita (~45s). Open with the PS3 scenario - 9:55am, "
              "lot says 47 free, you still circle 4 floors. Land on the "
              "spot-level vs lot-level distinction.",
    )

    add_bullet_slide(
        prs,
        "Method 1/2 - YOLOv8 Vehicle Detection",
        [
            "YOLOv8n (Ultralytics), pre-trained on COCO.",
            "Restrict to 4 vehicle classes: car, motorcycle, bus, truck.",
            "Confidence floor: 0.25.",
            "No fine-tuning - COCO already covers overhead car shots.",
            "Pipeline so far:  Frame -> YOLOv8 -> vehicle bounding boxes.",
        ],
        notes="Speaker: Eswardeep (~60s). Emphasize 'no fine-tuning' - it's "
              "what makes the system deployable on a new camera in minutes.",
    )

    add_bullet_slide(
        prs,
        "Method 2/2 - Static ROIs + IoU",
        [
            "One-time setup: click 2 corners per spot (roi_picker.py).",
            "Stored as JSON of axis-aligned boxes; reloaded at runtime.",
            "Per frame, per spot: IoU(spot, vehicle).",
            "Decision: Occupied iff max IoU > 0.5 (PASCAL VOC convention).",
            "Stateless, O(N x M) per frame, negligible vs detector cost.",
        ],
        notes="Speaker: Praneeth (~60s). Sell the simplicity: no training, "
              "no learnable parameters, generalizes to a new camera in 2 min.",
    )

    add_bullet_slide(
        prs,
        "Data & Setup",
        [
            "Custom UTD clip: 2 min @ 1080p, 4th floor of PS3, ~50 deg downward.",
            "Camera fully static (phone on concrete ledge).",
            "Reference datasets reviewed: CNRPark-EXT, PKLot (context only).",
            "Ground truth: 20 frames x N spots, labelled with label_gt.py.",
        ],
        notes="Speaker: Nikita (~30s). Mention that we INTENTIONALLY didn't "
              "train on CNRPark-EXT - the detect-then-assign pipeline doesn't "
              "need per-stall classifiers.",
    )

    add_table_slide(
        prs,
        "Results",
        ["Metric", "Synthetic 90-frame", "UTD live"],
        [
            ["Occupancy Accuracy",                "79.6 %",  "TBD"],
            ["Precision (Occupied)",              "100.0 %", "TBD"],
            ["Recall (Occupied)",                 "74.2 %",  "TBD"],
            ["F1 (Occupied)",                     "85.2 %",  "TBD"],
            ["Inference FPS (CPU, 640x640)",      "5.9",     "TBD"],
            ["End-to-end FPS",                    "4.8",     "TBD"],
        ],
        notes="Speaker: Sandeep (~75s incl. demo). The Synthetic column is "
              "REAL numbers from running our pipeline on a 90-frame stress "
              "test (810 (frame, spot) judgments, exact GT). UTD column is "
              "filled in once we run the same scripts on the campus video. "
              "Read the headline numbers (Acc 79.6%, Precision 100%, FPS 6) "
              "then click the demo video.",
    )

    add_bullet_slide(
        prs,
        "Live Demo",
        [
            "Green = Empty,  Red = Occupied.",
            "HUD: live FPS and Occupied/Total counter.",
            "Watch one spot flip as a car backs in.",
        ],
        notes="Insert demo_video.mp4 here BEFORE rehearsal: Insert > Video > "
              "This Device... > demo_video.mp4. Set to 'Start: Automatically'.",
    )

    add_bullet_slide(
        prs,
        "Failure Analysis",
        [
            "Occlusion: foreground truck hides back-row spot -> false Occupied.",
            "Off-line parking: vehicle crosses painted line -> IoU < 0.5 -> false Empty.",
            "Camera drift: small physical shift misaligns every ROI.",
            "All inherent to STATIC spatial reasoning - each is a future-work hook.",
        ],
        notes="Speaker: Eswardeep (~30s). Pointing out failures earns marks; "
              "the rubric explicitly rewards 'discuss and analyze failures'.",
    )

    add_bullet_slide(
        prs,
        "Conclusion & Next Steps",
        [
            "Real-time per-spot occupancy with NO per-camera training.",
            "Next: polygonal ROIs, periodic re-registration via line fiducials.",
            "Multi-camera fusion to recover occluded back rows.",
            "Code, report, and demo video are in the submission zip.",
            "Thank you - questions?",
        ],
        notes="Speaker: Praneeth (~30s). End on the 'questions?' before the "
              "5:00 mark to leave time for the 1-min Q&A.",
    )

    add_bullet_slide(
        prs,
        "Q&A backup",
        [
            "Q: Why not train per-spot like CNRPark-EXT?  "
                "A: To avoid per-camera training; detect-then-assign generalizes.",
            "Q: Why YOLOv8n vs v8m/l?  "
                "A: Real-time on CPU; bigger weights cost 5-10x latency for ~1-2 mAP.",
            "Q: Why IoU 0.5?  "
                "A: PASCAL VOC convention + best F1 in our ablation (0.3,0.4,0.5,0.6).",
        ],
        notes="Leave this on screen during Q&A. Don't speak through it.",
    )

    prs.save(OUT)
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()

"""
Convert the public carPark sample's positions pickle into our rois.json format.

The MoazEldsouky/Parking-Space-Counter sample stores parking spots as a list
of (x, y) top-left corners with a fixed width=107, height=48 box.
We convert each (x, y) -> [x, y, x+W, y+H] and emit JSON in our schema.

Usage:
    python convert_carpark_positions.py
"""

import json
import os
import pickle

PKL = os.path.join(os.path.dirname(__file__), "..", "data", "carpark_positions.pkl")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "rois_carpark.json")
W, H = 107, 48
IMG_W, IMG_H = 1280, 720   # carPark.mp4 native resolution


def main():
    with open(PKL, "rb") as f:
        positions = pickle.load(f)
    spots = [[int(x), int(y), int(x) + W, int(y) + H] for (x, y) in positions]
    out_path = os.path.abspath(OUT)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"image_size": [IMG_W, IMG_H], "spots": spots}, f, indent=2)
    print(f"[OK] wrote {len(spots)} spots -> {out_path}")


if __name__ == "__main__":
    main()

"""Data utilities for the GRB_Technology virtual try-on pipeline."""

import os
import json
import numpy as np
from PIL import Image


def load_openpose_keypoints(json_path: str) -> list:
    """
    Parse an OpenPose JSON output file.
    Returns list of (x, y, confidence) for 18 body keypoints.
    """
    with open(json_path) as f:
        data = json.load(f)
    people = data.get('people', [])
    if not people:
        return [(0, 0, 0.0)] * 18
    flat = people[0].get('pose_keypoints_2d', [])
    result = [(flat[i], flat[i+1], flat[i+2]) for i in range(0, min(len(flat), 54), 3)]
    while len(result) < 18:
        result.append((0, 0, 0.0))
    return result


def load_parse_map(path: str) -> np.ndarray:
    """Load a body parsing label image as an integer array [H, W]."""
    return np.array(Image.open(path), dtype=np.int32)


def build_pairs(image_dir: str, cloth_dir: str, pairs_file: str = None) -> list:
    """
    Build list of (person_path, cloth_path) pairs.
    If pairs_file exists, reads from it; otherwise zips sorted directory listings.
    """
    if pairs_file and os.path.exists(pairs_file):
        pairs = []
        with open(pairs_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    pairs.append((
                        os.path.join(image_dir, parts[0]),
                        os.path.join(cloth_dir,  parts[1])
                    ))
        return pairs
    imgs = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png')))
    clts = sorted(f for f in os.listdir(cloth_dir)  if f.lower().endswith(('.jpg','.jpeg','.png')))
    return [(os.path.join(image_dir, i), os.path.join(cloth_dir, c)) for i, c in zip(imgs, clts)]


def save_result(img: Image.Image, out_dir: str, filename: str) -> str:
    """Save result image and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    img.save(path)
    return path

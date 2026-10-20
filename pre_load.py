"""Pre-extract CvT multi-stage features + rain/blur priors for each split.

Usage
-----
    python pre_load.py                    # process all splits
    python pre_load.py --split val        # process one split
"""

import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from main import CvTFeatureExtractor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGE_DIR = "/home/jackyang/WMJ/derain_datasets/derain1"
SPLITS = {
    "train": ("train.csv",
              "preload_derain1_scene/train"),
    "val":   ("val.csv",
              "preload_derain1_scene/val"),
    "test":  ("test.csv",
              "preload_derain1_scene/test"),
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_split(extractor, csv_file, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    print(f"[{output_dir}] {len(df)} samples")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(output_dir)):
        img_path = os.path.join(image_dir, row["picture"])
        if not os.path.exists(img_path):
            print(f"  skip (missing): {img_path}")
            continue

        try:
            with torch.no_grad():
                feats = extractor.extract_features(img_path)
                rain = extractor.extract_rde(img_path, target_size=(24, 24))
                blur = extractor.extract_sde(img_path, target_size=(24, 24))
        except Exception as e:
            print(f"  skip ({e}): {img_path}")
            continue

        data = {
            "f0": feats["stage0"]["feature_map"].cpu(),
            "f1": feats["stage1"]["feature_map"].cpu(),
            "f2": feats["stage2"]["feature_map"].cpu(),
            "cls": feats["stage2"]["cls_token"].cpu(),
            "rain": rain.cpu(),
            "blur": blur.cpu(),
            "mos": row["mos"],
            "image_name": row["picture"],
        }
        # Zero-padded index keeps glob order deterministic in DerainIQADataset
        torch.save(data, os.path.join(output_dir, f"{idx:05d}.pt"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all",
                        choices=["all", *SPLITS.keys()])
    args = parser.parse_args()

    splits = SPLITS.keys() if args.split == "all" else [args.split]
    extractor = CvTFeatureExtractor()

    for name in splits:
        csv_file, out_dir = SPLITS[name]
        print(f"\n=== {name} ===")
        preprocess_split(extractor, csv_file, IMAGE_DIR, out_dir)


if __name__ == "__main__":
    main()

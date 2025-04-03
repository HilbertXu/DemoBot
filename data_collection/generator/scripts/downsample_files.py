"""
File for 
"""


import shutil
from pathlib import Path
import numpy as np
import glob
import os
import argparse

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--keep_interval", type=int, default=4)
    args = parser.parse_args()
    return args



# Configuration
args = parse_args()
image_folders = [
    "rgb",
    "depth",
    "processed/sam/object/images_masks",
    "processed/sam/right/images_masks",
    "processed/sam/target/images_masks"
]

# Step-1 downsample images
for folder in image_folders:
    print(f"📂 Processing folder: {folder}")
    image_paths = sorted(glob.glob(f"{args.data_dir}/{args.seq_name}/{folder}/*.png"))
    image_paths = [Path(f) for f in image_paths]
    selected_images = set(img for i, img in enumerate(image_paths) if i % args.keep_interval == 0)

    # Delete unselected images
    deleted_count = 0
    for img in image_paths:
        if img not in selected_images:
            img.unlink()
            deleted_count += 1

    print(f"🗑️ Deleted {deleted_count} images")
    print(f"✅ Kept {len(selected_images)} images")

# Absolute path to the .npy file
npy_file = Path('/home/advr/projects/DemoBot/data/realsense/hammer_head_repose_fps30/processed/object/hammer_head_colored_pose_cam.npy')
npy_fs = glob.glob(f"{args.data_dir}/{args.seq_name}/processed/object/*.npy")

# Step 2: Downsample and overwrite the .npy file
for npy_file in npy_fs:
    try:

        arr = np.load(npy_file)
        downsampled = arr[::args.keep_interval]
        np.save(npy_file, downsampled)
        print(f"✅ Resized and saved .npy: {npy_file}")
    except Exception as e:
        print(f"⚠️ Failed to process .npy file {npy_file}: {e}")

print("🏁 All done.")

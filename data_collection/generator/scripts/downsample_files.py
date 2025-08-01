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
    parser.add_argument("--use_selected_keyframe", action='store_true', default=False)
    
    args = parser.parse_args()
    return args



# Configuration
args = parse_args()
image_folders = [
    "rgb",
    "depth",
    "processed/sam/object/images_masks",
    "processed/sam/target/images_masks",
]

keyframe_folders = "kf_rgb"
kf_paths = sorted(glob.glob(f"{args.data_dir}/{args.seq_name}/{keyframe_folders}/*.png"))
kf_paths = [Path(f) for f in kf_paths]
kf_names = sorted([p.name for p in kf_paths])

# full image paths
image_paths = sorted(glob.glob(f"{args.data_dir}/{args.seq_name}/{image_folders[0]}/*.png"))
image_paths = [Path(f) for f in image_paths]

selected_images = set(img for i, img in enumerate(image_paths) if i % args.keep_interval == 0)
all_image_names = sorted([p.name for p in image_paths])


selected_image_names = set(sorted([p.name for p in selected_images.union(set(kf_paths))]))
kf_indices = np.asarray(sorted([all_image_names.index(n) for n in kf_names]))
selected_image_indices = np.asarray([all_image_names.index(n) for n in sorted(selected_image_names)])


# Step-1 downsample images
for folder in image_folders:
    print(f"📂 Processing folder: {folder}")
    image_paths = sorted(glob.glob(f"{args.data_dir}/{args.seq_name}/{folder}/*.png"))
    image_paths = [Path(f) for f in image_paths]
    
    # Delete unselected images
    deleted_count = selected_image_indices.shape[0]
    kept_count = 0
    kept_indices = []
    for idx, img in enumerate(image_paths):
        if idx not in selected_image_indices:
            img.unlink()
            deleted_count += 1
        else:
            kept_count += 1
            kept_indices.append(idx)

    print(f"🗑️ Deleted {deleted_count} images")
    print(f"✅ Kept {kept_count} images")

# Absolute path to the .npy file
npy_fs = glob.glob(f"{args.data_dir}/{args.seq_name}/processed/object/*.npy")

# Step 2: Downsample and overwrite the .npy file
for npy_file in npy_fs:
    try:
        arr = np.load(npy_file)
        downsampled = arr[selected_image_indices]
        np.save(npy_file, downsampled)
        print(f"✅ Resized to {downsampled.shape[0]} and saved .npy: {npy_file}")
    except Exception as e:
        print(f"⚠️ Failed to process .npy file {npy_file}: {e}")

print("🏁 All done.")

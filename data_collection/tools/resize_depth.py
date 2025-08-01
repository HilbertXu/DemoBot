import cv2
import numpy as np
from glob import glob
import os

root_dir = "/home/advr/projects/DemoBot/data/iphone/hammer_head_repose_fps30"
output_dir = "/home/advr/projects/DemoBot/data/iphone/hammer_head_repose_fps30/depth_npy"
raw_depth_fs = sorted(glob(f"{root_dir}/depth_resize/*.png"))

os.makedirs(output_dir, exist_ok=True)

for f in raw_depth_fs:
    data = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    print(data.shape)
    np.save(
        f.replace("depth_resize", "depth_npy").replace(".png", ".npy"),
        data / 1000.0
    )
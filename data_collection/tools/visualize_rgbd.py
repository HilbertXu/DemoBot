import matplotlib.pyplot as plt
import numpy as np
import cv2  # for image reading
from glob import glob
import os

# Replace these paths with your actual file paths
rgb_image_path = '/home/advr/projects/DemoBot/data/iphone/hammer_handle_repose_fps15/rgb/'
depth_image_path = '/home/advr/projects/DemoBot/data/iphone/hammer_handle_repose_fps15/depth/'
output_path = '/home/advr/projects/DemoBot/data/iphone/hammer_handle_repose_fps15/rgbd/'
os.makedirs(output_path, exist_ok=True)

rgb_images = sorted(glob(f"{rgb_image_path}/*.png"))

for rgb_f in rgb_images:
    depth_f = rgb_f.replace("/rgb", "/depth")

    # Load the RGB image
    rgb = cv2.imread(rgb_f)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Load the depth image (assume it's a single channel 16-bit or 8-bit image)
    depth = cv2.imread(depth_f, cv2.IMREAD_UNCHANGED)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    im = axes[1].imshow(depth)
    axes[1].set_title('Depth Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(rgb_f.replace("/rgb", "/rgbd"))
    plt.cla()
    plt.clf()
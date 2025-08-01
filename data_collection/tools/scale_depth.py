import os
import cv2
import numpy as np

# Input and output folder paths
input_folder = '/home/advr/projects/DemoBot/data/iphone/hammer_head_repose/depth'
output_folder = '/home/advr/projects/DemoBot/data/iphone/hammer_head_repose/depth_scaled'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.tiff', '.tif')):  # adjust if needed
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load depth image (preserve original depth)
        depth = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        # Multiply by 1000 and clip to valid range for 16-bit
        depth_scaled = (depth).astype(np.uint16)

        # Save scaled depth image
        cv2.imwrite(output_path, depth_scaled)

        print(f"Processed {filename}")

print("All images processed.")
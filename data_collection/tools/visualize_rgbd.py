import matplotlib.pyplot as plt
import numpy as np
import cv2  # for image reading

# Replace these paths with your actual file paths
rgb_image_path = '/home/advr/projects/DemoBot/data/iphone/hammer_assembly/images/1743446052796.png'
depth_image_path = '/home/advr/projects/DemoBot/data/iphone/hammer_assembly/depth_scaled/1743446052796.png'

# Load the RGB image
rgb = cv2.imread(rgb_image_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

# Load the depth image (assume it's a single channel 16-bit or 8-bit image)
depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

print(np.mean(depth), np.max(depth), np.min(depth))


# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(rgb)
axes[0].set_title('RGB Image')
axes[0].axis('off')

im = axes[1].imshow(depth)
axes[1].set_title('Depth Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
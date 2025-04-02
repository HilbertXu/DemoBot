import cv2
import cv2.aruco as aruco

# Define ArUco dictionary (4x4 grid with 50 possible markers)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# Define Marker ID (must be between 0 and 49 for this dictionary)
marker_id = 2  # Change this to generate different markers

# Define Marker Size (in pixels)
marker_size = 200  # 200x200 pixels

# Generate ArUco marker
marker_image = aruco.drawMarker(aruco_dict, marker_id, marker_size)


import matplotlib.pyplot as plt

# Define marker size in cm
marker_real_size = 10  # cm

# Convert cm to inches (1 cm = 0.3937 inches)
marker_inches = marker_real_size * 0.3937

# Plot and save the ArUco marker with correct size
plt.figure(figsize=(marker_inches, marker_inches), dpi=200)
plt.imshow(marker_image, cmap='gray')
plt.axis('off')
plt.savefig(f"aruco_marker_print_{marker_id}.png", dpi=200, bbox_inches='tight')
plt.show()

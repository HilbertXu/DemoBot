'''
Date: 25 Mar 2025
Author: Yucheng Xu
Description: updated script for both recording RGB-D data and calibrate camera

'''

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import json
import time
import os
import argparse
import shutil

def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser(description="ROS Image Saver with Synchronization")

    # Add arguments
    parser.add_argument("--output_dir", type=str, help="path to the output folder")
    parser.add_argument("--calibrate_camera", action='store_true', help="whether to calibrate camera extrinsic")
    parser.add_argument("--calibrate_camera_vis", action='store_true', help="whether to visualize the marker")
    
    return parser.parse_args()

args = parse_args()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Get the absolute path to the subfolder
subfolder_depth = f"{args.output_dir}/depth"
subfolder_rgb = f"{args.output_dir}/rgb"

print(subfolder_depth, subfolder_rgb)

# Create all 

RecordStream = False

# setting up output folder
if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(f'{args.output_dir}/processed/sam/object/images_masks', exist_ok=True)
os.makedirs(f'{args.output_dir}/processed/sam/right/images_masks', exist_ok=True)
os.makedirs(f'{args.output_dir}/processed/sam/left/images_masks', exist_ok=True)
os.makedirs(subfolder_depth, exist_ok=True)
os.makedirs(subfolder_rgb, exist_ok=True)
os.symlink(f'./rgb', f'{args.output_dir}/images')



marker_size = 0.072 # Marker side length in meters
# Load ArUco dictionary and define detector parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # Using 4x4 marker
parameters = aruco.DetectorParameters_create()
obj_points = np.array([
    [-marker_size / 2,  marker_size / 2,  0],  # Top-left
    [ marker_size / 2,  marker_size / 2,  0],  # Top-right
    [ marker_size / 2, -marker_size / 2,  0],  # Bottom-right
    [-marker_size / 2, -marker_size / 2,  0]   # Bottom-left
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Streaming loop
print("Camera on, press 'e' to calibrate camera, press 'space' to start recording")
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()


        # Get instrinsics from aligned_depth_frame
        intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image)
        )  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            grey_color,
            color_image,
        )

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
        cv2.imshow("Align Example", images)

        key = cv2.waitKey(1)
        
        if key == ord('e'):
            intrinsics_matrix = np.asarray([
                    [intrinsics.fx, 0.0, intrinsics.ppx],
                    [0.0, intrinsics.fy, intrinsics.ppy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
            corners, ids, _ = aruco.detectMarkers(color_image, aruco_dict, parameters=parameters)
            retval, rvec, tvec = cv2.solvePnP(obj_points, corners[0], intrinsics_matrix, dist_coeffs)

            # Convert rotation vector to a rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            np.savez(
                f'{args.output_dir}/camera_extrinsic.npz',
                R=R, 
                T=tvec
            )
            print("Rotation Matrix (R):\n", R)
            print("Translation Vector (T):\n", tvec)
            
            if args.calibrate_camera_vis:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, intrinsics_matrix, dist_coeffs)
                image = color_image.copy()
                # Draw XYZ axes on the marker
                for i in range(len(ids)):
                    cv2.aruco.drawAxis(image, intrinsics_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size)

                cv2.imwrite(f'{args.output_dir}/calibrate.png', image)
                
        # Start saving the frames if space is pressed once until it is pressed again
        elif key & 0xFF == ord(" "):
            if not RecordStream:
                time.sleep(0.2)
                RecordStream = True

                with open(f"{args.output_dir}/cam_K.txt", "w") as f:
                    f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
                    f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
                    f.write(f"{0.0} {0.0} {1.0}\n")
                
                intrinsics_matrix = np.asarray([
                    [intrinsics.fx, 0.0, intrinsics.ppx],
                    [0.0, intrinsics.fy, intrinsics.ppy],
                    [0.0, 0.0, 1.0]
                ])
                
                np.save(f"{args.output_dir}/cam_K.npy", intrinsics_matrix)
                
                print("Recording started")
            else:
                RecordStream = False
                print("Recording stopped")
        
        

        if RecordStream:
            framename = int(round(time.time() * 1000))

            # Define the path to the image file within the subfolder
            image_path_depth = os.path.join(subfolder_depth, f"{framename}.png")
            image_path_rgb = os.path.join(subfolder_rgb, f"{framename}.png")
            
            cv2.imwrite(image_path_depth, depth_image)
            cv2.imwrite(image_path_rgb, color_image)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:

            cv2.destroyAllWindows()

            break
finally:
    
    

    pipeline.stop()
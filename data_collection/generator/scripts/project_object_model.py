import argparse
import sys
import os
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# @TODO keep only the points that fall into the object mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="sequence name")
    parser.add_argument("--seq_name", type=str, help="sequence name")
    parser.add_argument("--no_vis", default=False, action="store_true")
    parser.add_argument("--right_object_mesh", type=str, default=None, help="sequence name")
    parser.add_argument("--left_object_mesh", type=str, default=None, help="sequence name")

    args = parser.parse_args()

    return args


def project_3d_to_2d(points_3d, object_to_camera, intrinsic):
    """
    Projects 3D object points into the image plane using object pose and camera intrinsics.
    
    Args:
        points_3d (numpy array): (N, 3) array of 3D points in object space.
        object_to_camera (numpy array): (4, 4) transformation matrix (object to camera).
        intrinsic (numpy array): (3, 3) camera intrinsic matrix.

    Returns:
        points_2d (numpy array): (N, 2) projected 2D pixel coordinates.
    """

    # Convert 3D points to homogeneous coordinates (add a 1 in the last column)
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # (N, 4)

    # Transform points from object space to camera space
    points_camera = (object_to_camera @ points_homogeneous.T).T  # (N, 4)

    # Normalize homogeneous coordinates (divide by the last column)
    points_camera = points_camera[:, :3] / points_camera[:, 3:]

    # Project points into image plane: 2D = K * 3D
    points_2d_homogeneous = (intrinsic @ points_camera.T).T  # (N, 3)

    # Convert from homogeneous to 2D pixel coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    return points_2d


def process_data(object_mesh_f, images, out_image_dir):
    os.makedirs(out_image_dir, exist_ok=True)
    
    if object_mesh_f is None:
        return np.zeros((len(images), 2))
    else:
        object_name = object_mesh_f.split("/")[-1].split(".")[0]
        mesh = trimesh.load(object_mesh_f, process=False)
        object_points_3d = np.array(mesh.sample(500))  # Object points in object space
        
        num_frames = len(images)
        pose_f = f"{args.data_dir}/{args.seq_name}/processed/object/{object_name}_pose_cam.npy"
        assert os.path.exists(pose_f), f"The poses of {object_name} is missing, please check the file path"
        poses = np.load(f"{args.data_dir}/{args.seq_name}/processed/object/{object_name}_pose_cam.npy")
        
        assert poses.shape[0] == num_frames
        projected_2d = []
        for i in tqdm(range(num_frames)):
            image_f = images[i]
            out_f = f"{out_image_dir}/{image_f.split('/')[-1]}"
            # Example: Object pose in camera coordinates (4x4 transformation matrix)
            object_to_camera = poses[i, :, :]
            # Project 3D points to 2D image plane
            points_2d = project_3d_to_2d(object_points_3d.copy(), object_to_camera, K)

            # Display projected points on an image
            image = cv2.imread(images[i])
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.scatter(points_2d[:, 0], points_2d[:, 1], color="red", s=5)  # Overlay points
            plt.savefig(out_f)
            plt.cla()
            plt.clf()
            
            projected_2d.append(points_2d)
        
        projected_2d = np.stack(projected_2d, axis=0)
    
        return projected_2d

if __name__ == "__main__":
    args = parse_args()
    
    out_dir = f"{args.data_dir}/{args.seq_name}/processed/object"
    os.makedirs(out_dir, exist_ok=True)

    # Example: Camera intrinsic matrix
    K = intrinsic = np.load(f"{args.data_dir}/{args.seq_name}/cam_K.npy")
    
    out_image_dir = f"{args.data_dir}/{args.seq_name}/processed/object/projected_2d"
    images = sorted(glob(f"{args.data_dir}/{args.seq_name}/images/*"))
    keypoints = {}
    if (args.right_object_mesh is not None) and (args.left_object_mesh is not None):
        keypoints_right = process_data(args.right_object_mesh, images, f"{out_image_dir}_right")
        keypoints_left = process_data(args.left_object_mesh, images, f"{out_image_dir}_left")
        keypoints = {
            'right': keypoints_right,
            'left': keypoints_left
        }
    elif (args.right_object_mesh is not None):
        keypoints = process_data(args.right_object_mesh, images, f"{out_image_dir}_right")
        keypoints = {
            'object': keypoints
        }
    else:
        keypoints = process_data(args.left_object_mesh, images, f"{out_image_dir}_left")
        keypoints = {
            'object': keypoints
        }
    

    np.savez(
        f"{out_dir}/keypoints.npz",
        **keypoints
    )

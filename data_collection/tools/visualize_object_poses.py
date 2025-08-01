import os
import trimesh
import argparse
import numpy as np
from copy import deepcopy
import pickle
from scipy.spatial.transform import Rotation as R

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--asset_dir", type=str, default="")
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--right_object_mesh", type=str, default=None)
    parser.add_argument("--left_object_mesh", type=str, default=None)
    args = parser.parse_args()
    
    return args


def convert_to_matrix(pose):
    trans = pose[:3]
    quat = pose[3:]
    matrix = np.eye(4)
    rot_mat = R.from_quat(quat).as_matrix()
    matrix[:3, :3] = rot_mat
    matrix[:3, 3] = trans 

    return matrix


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    asset_dir = args.asset_dir
    seq_name = args.seq_name

    right_object_name = args.right_object_mesh.split("/")[-1][:-4]
    left_object_name = args.left_object_mesh.split("/")[-1][:-4]
    right_object_mesh = trimesh.load(f"{asset_dir}/{args.right_object_mesh}")
    left_object_mesh = trimesh.load(f"{asset_dir}/{args.left_object_mesh}")

    all_images = sorted(os.listdir(f"{args.data_dir}/{args.seq_name}/rgb"))
    keyframes = sorted(os.listdir(f"{args.data_dir}/{args.seq_name}/kf_rgb"))
    keyframe_indices = np.asarray([all_images.index(n) for n in keyframes])
    num_keyframes = len(keyframe_indices)

    # trajectory = pickle.load(open(f"{data_dir}/{seq_name}/trajectory_franka_allegro.pkl", "rb"))['action_chunks']
    # print(trajectory.keys())
    # chunk_ids = ['chunk_1', 'chunk_2', 'chunk_3']
    # refined_keyframe_right_object_poses = []
    # refined_keyframe_left_object_poses = []
    # for chunk_id in chunk_ids:
    #     right_pose = trajectory[chunk_id]['goal_object_pose.right']
    #     left_pose = trajectory[chunk_id]['goal_object_pose.left']
    #     refined_keyframe_right_object_poses.append(convert_to_matrix(right_pose))
    #     refined_keyframe_left_object_poses.append(convert_to_matrix(left_pose))
    

    

    refined_keyframe_right_object_poses = np.load(f"{data_dir}/{seq_name}/processed/object/{right_object_name}_pose_cam.refine.npy")[keyframe_indices, :, :]
    refined_keyframe_left_object_poses = np.load(f"{data_dir}/{seq_name}/processed/object/{left_object_name}_pose_cam.refine.npy")[keyframe_indices, :, :]

    init_keyframe_right_object_poses = np.load(f"{data_dir}/{seq_name}/processed/object/{right_object_name}_pose_cam.npy")[keyframe_indices, :, :]
    init_keyframe_left_object_poses = np.load(f"{data_dir}/{seq_name}/processed/object/{left_object_name}_pose_cam.npy")[keyframe_indices, :, :]

    for i in range(num_keyframes):
        refine_right_mesh = deepcopy(right_object_mesh)
        refine_left_mesh = deepcopy(left_object_mesh)

        refine_right_mesh.apply_transform(refined_keyframe_right_object_poses[i])
        refine_left_mesh.apply_transform(refined_keyframe_left_object_poses[i])


        scene = trimesh.Scene()
        scene.add_geometry(refine_right_mesh, node_name="peg_refined")
        scene.add_geometry(refine_left_mesh, node_name="hole_refined")

        init_right_mesh = deepcopy(right_object_mesh)
        init_right_mesh.visual.face_colors = [255, 0, 0, 255]
        init_left_mesh = deepcopy(left_object_mesh)

        init_right_mesh.apply_transform(init_keyframe_right_object_poses[i])
        init_left_mesh.apply_transform(init_keyframe_left_object_poses[i])


        scene.add_geometry(init_right_mesh, node_name="peg_init")
        scene.add_geometry(init_left_mesh, node_name="hole_init")

        scene.show()



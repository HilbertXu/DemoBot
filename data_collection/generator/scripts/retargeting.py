import os
import sys
sys.path.append('./')
import torch
import numpy as np
import tempfile
from pathlib import Path
import sapien

# Dex retargeting
from pytransform3d import rotations
from scipy.spatial.transform import Rotation
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

from mano_layer import MANOLayer

import argparse




def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser(description="Retargeting from MANO hamd to robot hand")

    # Add arguments
    parser.add_argument("--mano_root", type=str, help="path to the mano models")
    parser.add_argument("--assets_root", type=str, help="path to the asset folder")
    parser.add_argument("--scene_dir", type=str, help="path to the scene directory")
    parser.add_argument("--config", type=str, help="path to the retargeting config")
    parser.add_argument("--hand_base", type=str, help="name of the base link of the robot hand")    

    return parser.parse_args()



class AllegroHandRetargeting(object):
    def __init__(
        self,
        config,
        hand_base,
        mano_root):
        
        override = dict(add_dummy_free_joint=True)
        retarget_config = RetargetingConfig.load_from_file(config, override=override)
        self.retargeting = retarget_config.build()
        self.retargeting_joint_names = self.retargeting.joint_names
        
        # Scene
        self.scene = sapien.Scene()
        loader = self.scene.create_urdf_loader()
        
        # Build robot
        urdf_path = Path(retarget_config.urdf_path)
        if "glb" not in urdf_path.stem:
            urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
        robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
        urdf_name = urdf_path.name
        temp_path = str(urdf_path).replace('_glb', '_retargeting')
        robot_urdf.write_xml_file(temp_path)

        robot = loader.load(temp_path)
        sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
        self.retarget2sim = np.array([self.retargeting.joint_names.index(n) for n in sapien_joint_names]).astype(int)
        print(self.retargeting.joint_names)
        print(sapien_joint_names)
        
        self.hand_type = HandType.right
        self.hand_base = hand_base
        self.mano_root = mano_root

    def _compute_hand_geometry(self, p, t, use_camera_frame=False):
        p = torch.from_numpy(p.astype(np.float32))
        t = torch.from_numpy(t.astype(np.float32))
        vertex, joint = self.mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]
        if not use_camera_frame:
            joint_ones = np.ones((joint.shape[0], 1))
            joint_homo = np.hstack([joint, joint_ones])
            joint_world = (self.camera_mat @ joint_homo.T).T[:, :3]
            
            vertex_ones = np.ones((vertex.shape[0], 1))
            vertex_homo = np.hstack([vertex, vertex_ones])
            vertex_world = (self.camera_mat @ vertex_homo.T).T[:, :3]
            
            return vertex_world, joint_world
        else:
            return vertex, joint

    def _global_hand_orientation_to_world(self, rot):
        # Step 1: Convert axis-angle to rotation matrix
        R_mano_cam = Rotation.from_rotvec(rot).as_matrix()
        
        print(R_mano_cam.shape, self.camera_mat.shape)
        
        # Step 2: Transform the rotation to the world frame
        R_mano_world = self.camera_mat[:3, :3] @ R_mano_cam  # Matrix multiplication
        
        # Step 3: Convert back to axis-angle representation
        quaternion_world = Rotation.from_matrix(R_mano_world).as_quat()
        
        return quaternion_world
    
    def _convert_transform_sequence(self, transform_list):
        """
        Convert a sequence of 4x4 transformation matrices to translations and temporally consistent quaternions.
        
        Parameters:
            transform_list (list or np.ndarray): Sequence of 4x4 numpy arrays.
        
        Returns:
            translations (list of np.ndarray): List of translation vectors.
            quaternions (list of np.ndarray): List of quaternions in [x, y, z, w] format.
        """
        translations = []
        quaternions = []
        
        for i, T in enumerate(transform_list):
            # Extract translation vector from last column
            translation = T[:3, 3]
            
            # Extract rotation matrix (upper left 3x3) and convert to quaternion
            R_mat = T[:3, :3]
            q = Rotation.from_matrix(R_mat).as_quat()  # default order [x, y, z, w]
            
            # Enforce temporal consistency: ensure the dot product with the previous quaternion is positive.
            if i > 0:
                if np.dot(q, quaternions[-1]) < 0:
                    q = -q
            
            translations.append(translation)
            quaternions.append(q)
        
        return translations, quaternions
    
    def retarget(self, assets_root, scene_dir, camera_f):
        # Load camera extrinsic matrix
        camera_pose = np.load(camera_f)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = camera_pose['R']
        extrinsic_matrix[:3, 3] = camera_pose['T'].reshape(3)
        self.camera_mat = np.linalg.inv(extrinsic_matrix)

        # Load processed MANO hand
        mano_f = f'{scene_dir}/processed/hold_fit.aligned.npy'
        mano_data = np.load(mano_f, allow_pickle=True).item()
        mano_hand = mano_data['right']
        
        # hand_shape = mano_hand['hand_beta'] # estimated hand shape
        hand_shape = np.load(f'{assets_root}/hand.npy') # optimized hand shape, has similar finger length to allegro hand
        hand_rot = mano_hand['hand_rot']
        hand_pose = mano_hand['hand_pose']
        hand_trans = mano_hand['hand_transl']
        
        num_frames = hand_pose.shape[0]
        
        self.mano_layer = MANOLayer(
            mano_root=self.mano_root, 
            side='right', 
            betas=hand_shape.astype(np.float32)
        )
        
        # Warm start
        wrist_quat = self._global_hand_orientation_to_world(hand_rot[0, :])
        p = np.concatenate([hand_rot[0, :], hand_pose[0,:]], axis=0)
        t = hand_trans[0, :]
        vertex, joint = self._compute_hand_geometry(p, t)
        # self.retargeting.warm_start(
        #     joint[0, :],
        #     wrist_quat,
        #     hand_type=self.hand_type,
        #     is_mano_convention=True,
        # )
        results = []
        transformations = []
        for i in range(num_frames):
            p = np.concatenate([hand_rot[i, :], hand_pose[i,:]], axis=0)
            t = hand_trans[i, :]
            vertex, joint = self._compute_hand_geometry(p, t)
            
            retargeting_type = self.retargeting.optimizer.retargeting_type
            indices = self.retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint[task_indices, :] - joint[origin_indices, :]
            
            # qpos = self.retargeting.retarget(ref_value)[self.retarget2sim]
            qpos = self.retargeting.retarget(ref_value)
            
            self.retargeting.optimizer.robot.compute_forward_kinematics(qpos)
            hand_base_pose = self.retargeting.optimizer.robot.get_link_pose(
                self.retargeting.optimizer.robot.get_link_index(self.hand_base)
            )
            transformations.append(hand_base_pose)
            
            results.append(qpos)
        translations, quaternions = self._convert_transform_sequence(transformations)
        
        results = np.stack(results, axis=0)
        translations = np.stack(translations, axis=0)
        quaternions = np.stack(quaternions, axis=0)
        np.savez(
            f'{scene_dir}/retarget.npz', 
            qpos=results,
            base_transl=translations,
            base_quat=quaternions
        )
            
        # print(qpos.shape, self.retargeting_joint_names)



args = parse_args()
retargeting = AllegroHandRetargeting(
    mano_root=args.mano_root,
    config=args.config,
    hand_base=args.hand_base
)

retargeting.retarget(
    assets_root=args.assets_root,
    scene_dir=args.scene_dir,
    camera_f=f'{args.scene_dir}/camera_extrinsic.npz'
)



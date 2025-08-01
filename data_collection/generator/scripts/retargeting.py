import os
import sys
sys.path.append('./')
import torch
import numpy as np
import tempfile
from pathlib import Path

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
    parser.add_argument("--right_config", type=str, default=None, help="path to the retargeting config for right hand")
    parser.add_argument("--left_config", type=str, default=None, help="path to the retargeting config for left hand")
    parser.add_argument("--hand_base", type=str, help="name of the base link of the robot hand")   
    return parser.parse_args()



class AllegroHandRetargeting(object):
    def __init__(
        self,
        hand_base,
        mano_root,
        assets_root,
        scene_dir,
        right_config=None,
        left_config=None,
        ):
        
        if right_config is not None:
            self.retarget_right_hand = True
            override = dict(add_dummy_free_joint=True)
            right_retarget_config = RetargetingConfig.load_from_file(right_config, override=override)
            self.right_retargeting = right_retarget_config.build()
            self.right_retargeting_joint_names = self.right_retargeting.joint_names
            
            self.robot_name = right_retarget_config.robot_name
            print(f"Retargeting from right MANO hand to {self.robot_name}")
            
            # Build robot
            right_urdf_path = Path(right_retarget_config.urdf_path)
            if "glb" not in right_urdf_path.stem:
                right_urdf_path = right_urdf_path.with_stem(right_urdf_path.stem + "_glb")
            right_robot_urdf = urdf.URDF.load(str(right_urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            temp_path = str(right_urdf_path).replace('_glb', '_retargeting')
            right_robot_urdf.write_xml_file(temp_path)
        else:
            self.retarget_right_hand = False
        
        
        if left_config is not None:
            self.retarget_left_hand = True
            
            override = dict(add_dummy_free_joint=True)
            left_retarget_config = RetargetingConfig.load_from_file(left_config, override=override)
            self.left_retargeting = left_retarget_config.build()
            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            
            self.robot_name = left_retarget_config.robot_name
            print(f"Retargeting from left MANO hand to {self.robot_name}")
            
            # Build robot
            left_urdf_path = Path(left_retarget_config.urdf_path)
            if "glb" not in left_urdf_path.stem:
                left_urdf_path = left_urdf_path.with_stem(left_urdf_path.stem + "_glb")
            left_robot_urdf = urdf.URDF.load(str(left_urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            temp_path = str(left_urdf_path).replace('_glb', '_retargeting')
            left_robot_urdf.write_xml_file(temp_path)
        else:
            self.retarget_left_hand = False
        
        
        # Load camera extrinsic matrix
        camera_f = f'{scene_dir}/camera_extrinsic.npz'
        camera_pose = np.load(camera_f)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = camera_pose['R']
        extrinsic_matrix[:3, 3] = camera_pose['T'].reshape(3)
        self.camera_mat = np.linalg.inv(extrinsic_matrix)

        # Load processed MANO hand
        mano_f = f'{scene_dir}/processed/hold_fit.aligned.npy'
        self.mano_data = np.load(mano_f, allow_pickle=True).item()

        if self.robot_name == 'leap_hand':
            
            sim_robot_joint_names = ['dummy_x_translation_joint', 'dummy_y_translation_joint', 'dummy_z_translation_joint', 
                                     'dummy_x_rotation_joint', 'dummy_y_rotation_joint', 'dummy_z_rotation_joint',
                                     '1', '12', '5', '9', '0', 
                                     '13', '4', '8', '2', '14', 
                                     '6', '10', '3', '15', '7', '11']
        elif self.robot_name == 'allegro':
            sim_robot_joint_names = ['dummy_x_translation_joint', 'dummy_y_translation_joint', 'dummy_z_translation_joint', 
                                     'dummy_x_rotation_joint', 'dummy_y_rotation_joint', 'dummy_z_rotation_joint', 
                                     'joint_0.0', 'joint_4.0', 'joint_8.0', 'joint_12.0', 
                                     'joint_1.0', 'joint_5.0', 'joint_9.0', 'joint_13.0', 
                                     'joint_2.0', 'joint_6.0', 'joint_10.0', 'joint_14.0', 
                                     'joint_3.0', 'joint_7.0', 'joint_11.0', 'joint_15.0']
        elif self.robot_name == 'panda_gripper':
            sim_robot_joint_names = ['dummy_x_translation_joint', 'dummy_y_translation_joint', 'dummy_z_translation_joint', 
                                     'dummy_x_rotation_joint', 'dummy_y_rotation_joint', 'dummy_z_rotation_joint', 
                                     'panda_finger_joint1', 'panda_finger_joint2']
        else:
            raise ValueError(f"Robot {self.robot_name} is not supported. Valid: allegro, leap_hand, panda_gripper")
        
        self.scene_dir = scene_dir
        self.hand_base = hand_base
        self.mano_root = mano_root
        self.hand_shape = np.load(f'{assets_root}/hand.npy') # optimized hand shape, has similar finger length to allegro hand
        self.hand_type_mapping = {
            'right': HandType.right,
            'left': HandType.left
        }
        self.retarget_cls_mapping = {
            'right': (self.right_retargeting, np.array([self.right_retargeting.joint_names.index(n) for n in sim_robot_joint_names]).astype(int)) if right_config is not None else None,
            'left': (self.left_retargeting, np.array([self.left_retargeting.joint_names.index(n) for n in sim_robot_joint_names]).astype(int)) if left_config is not None else None
        }
        

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
    
    def retarget(self, mano_hand, hand_type):
        hand_rot = mano_hand['hand_rot']
        hand_pose = mano_hand['hand_pose']
        hand_trans = mano_hand['hand_transl']
        
        num_frames = hand_pose.shape[0]
        
        self.mano_layer = MANOLayer(
            mano_root=self.mano_root, 
            side=hand_type, 
            betas=self.hand_shape.astype(np.float32)
        )
        
        retargeting, retarget2sim = self.retarget_cls_mapping[hand_type]
        
        # Warm start
        wrist_quat = self._global_hand_orientation_to_world(hand_rot[0, :])
        p = np.concatenate([hand_rot[0, :], hand_pose[0,:]], axis=0)
        t = hand_trans[0, :]
        vertex, joint = self._compute_hand_geometry(p, t)
        retargeting.warm_start(
            joint[0, :],
            wrist_quat,
            hand_type=self.hand_type_mapping[hand_type],
            is_mano_convention=True,
        )
        results = []
        transformations = []
        for i in range(num_frames):
            p = np.concatenate([hand_rot[i, :], hand_pose[i,:]], axis=0)
            t = hand_trans[i, :]
            vertex, joint = self._compute_hand_geometry(p, t)
            
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint[task_indices, :] - joint[origin_indices, :]
            
            qpos = retargeting.retarget(ref_value)[retarget2sim]
            
            retargeting.optimizer.robot.compute_forward_kinematics(qpos)
            hand_base_pose = retargeting.optimizer.robot.get_link_pose(
                retargeting.optimizer.robot.get_link_index(self.hand_base)
            )
            transformations.append(hand_base_pose)
            
            results.append(qpos)
        translations, quaternions = self._convert_transform_sequence(transformations)
        
        results = np.stack(results, axis=0)
        translations = np.stack(translations, axis=0)
        quaternions = np.stack(quaternions, axis=0)
        
        return {
            'qpos': results,
            'base_transl': translations,
            'base_quat': quaternions
        }
        
        
    
    def run(self):
        retarget = {}
        if self.retarget_left_hand:
            retarget['left'] = self.retarget(
                self.mano_data['left'], hand_type='left'
            )
        else:
            retarget['left'] = None
        if self.retarget_right_hand:
            retarget['right'] = self.retarget(
                self.mano_data['right'], hand_type='right'
            )
        else:
            retarget['right'] = None
        
        np.savez(
            f'{self.scene_dir}/retarget.npz',
            left=retarget['left'],
            right=retarget['right']
        )



args = parse_args()
retargeting = AllegroHandRetargeting(
    hand_base=args.hand_base,
    mano_root=args.mano_root,
    assets_root=args.assets_root,
    scene_dir=args.scene_dir,
    right_config=args.right_config,
    left_config=args.left_config,
)

retargeting.run()




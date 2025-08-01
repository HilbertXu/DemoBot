# Date: 14 June 2025
# Author: Yucheng Xu
# Code for refining the object poses in the selected keyframe

import os

import copy
import torch
import torch.nn.functional as F
import os.path as op
import numpy as np
import sys
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path = [".", ".."] + sys.path

from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import axis_angle_to_matrix


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--asset_dir", type=str, default="")
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--right_object_keypoints", type=str, default=None)
    parser.add_argument("--left_object_keypoints", type=str, default=None)
    parser.add_argument("--num_frames_to_refine", type=int, default=0)
    parser.add_argument("--task", type=str, default='insert')
    parser.add_argument("--mode", type=str, default='ro')
    args = parser.parse_args()
    args = edict(vars(args))
    
    return args


def prepare_data(pose, kpts, requires_grad=False):
    rot_mat = pose[:3, :3] # [3, 3]
    trans = pose[:3, 3]
    kpts = torch.tensor(kpts, dtype=torch.float32).cuda()

    axis_angs = matrix_to_axis_angle(torch.tensor(rot_mat, dtype=torch.float32)).view(1, 3).cuda()
    trans = torch.tensor(trans, dtype=torch.float32).cuda().view(1, 3).cuda()

    if requires_grad:
        return axis_angs.requires_grad_(), trans.requires_grad_(), kpts
    else:
        return axis_angs, trans, kpts

def apply_pose(rot, trans, kpts):
    # rot [num_kpts, 3, 3] 
    # trans [num_kpts, 3]
    # kpts [num_kpts, 3]
    rot_mat = axis_angle_to_matrix(rot)
    kpts_w = torch.bmm(rot_mat.repeat(3, 1, 1), kpts.unsqueeze(-1))
    kpts_w = kpts_w.squeeze() + trans


    return kpts_w    


def get_dir_vector(kpts):
    axis = kpts[0, :] - kpts[2, :]
    return axis / torch.norm(axis, dim=-1, keepdim=True)

def get_tip_pos(kpts, index):
    return kpts[index, :]


def calculate_loss(a_axis, b_axis, a_pos, b_pos, a_start, b_start, b_end, use_pos_loss):
    rot_loss = 1 - torch.abs(F.cosine_similarity(a_axis, b_axis, dim=-1))
    vec1 = b_start - a_start
    vec2 = b_end - a_start
    dist1 = torch.norm(torch.cross(vec1, a_axis), dim=-1)
    dist2 = torch.norm(torch.cross(vec2, a_axis), dim=-1)
    colinear_loss = dist1**2 + dist2**2

    pos_loss = F.smooth_l1_loss(a_pos, b_pos)

    if use_pos_loss:
        loss = rot_loss + colinear_loss + pos_loss
    else:
        loss = rot_loss + colinear_loss

    return loss


def calculate_reg(rot1, rot2, trans1, trans2):
    return F.mse_loss(rot1, rot2) + F.mse_loss(trans1, trans2) 



if __name__ == "__main__":
    args = parse_args()

    # read keyframe indices
    all_images = sorted(os.listdir(f"{args.data_dir}/{args.seq_name}/rgb"))
    keyframes = sorted(os.listdir(f"{args.data_dir}/{args.seq_name}/kf_rgb"))[1:] # usually the first frame is lift-up keyframe, not need to refine
    keyframe_indices = np.asarray([all_images.index(n) for n in keyframes])
    num_keyframes = len(keyframe_indices)
    print(keyframe_indices)

    # read init object poses
    right_object_name = args.right_object_keypoints.split("/")[-1][:-4]
    left_object_name = args.left_object_keypoints.split("/")[-1][:-4]
    init_right_object_poses = np.load(f"{args.data_dir}/{args.seq_name}/processed/object/{right_object_name}_pose_cam.npy")
    init_left_object_poses = np.load(f"{args.data_dir}/{args.seq_name}/processed/object/{left_object_name}_pose_cam.npy")

    refine_right_object_poses = copy.deepcopy(init_right_object_poses)
    refine_left_object_poses = copy.deepcopy(init_left_object_poses)

    init_right_object_pose_kf = copy.deepcopy(init_right_object_poses)[keyframe_indices, :, :]
    init_left_object_pose_kf = copy.deepcopy(init_left_object_poses)[keyframe_indices, :, :]

    # read object keypoints
    right_object_kpts = np.load(f"{args.asset_dir}/{args.right_object_keypoints}")['insert_keypoints'][:]
    left_object_kpts = np.load(f"{args.asset_dir}/{args.left_object_keypoints}")['insert_keypoints'][:]

    # optimization with pytorch
    for idx in range(num_keyframes-args.num_frames_to_refine, num_keyframes):
        print(f"optimizing keyframe: {keyframes[idx]}")
        if args.mode == 'ro':
            rot_to_opt, trans_to_opt, kpts_to_opt = prepare_data(
                init_right_object_pose_kf[idx], 
                right_object_kpts,
                requires_grad=True
            )
            rot_tgt, trans_tgt, kpts_tgt = prepare_data(
                init_left_object_pose_kf[idx],
                left_object_kpts,
                requires_grad=False
            )
        elif args.mode == 'lo':
            rot_to_opt, trans_to_opt, kpts_to_opt = prepare_data(
                init_left_object_pose_kf[idx],
                left_object_kpts,
                requires_grad=True
            )
            rot_tgt, trans_tgt, kpts_tgt = prepare_data(
                init_right_object_pose_kf[idx], 
                right_object_kpts,
                requires_grad=False
            )
        init_rot = rot_to_opt.clone().detach()
        init_trans = trans_to_opt.clone().detach()

        print(init_rot.shape, init_trans.shape, rot_to_opt.shape, trans_to_opt.shape)

        optimizer = torch.optim.Adam([rot_to_opt, trans_to_opt], lr=3e-4)

        pbar = tqdm(range(10000))
        min_loss = 999
        early_stop_tol = 500
        early_stop_counter = 0
        for step in pbar:
            optimizer.zero_grad()

            kpts_to_opt_w = apply_pose(rot_to_opt, trans_to_opt, kpts_to_opt)
            kpts_tgt_w = apply_pose(rot_tgt, trans_tgt, kpts_tgt)

            axis_to_opt = get_dir_vector(kpts_to_opt_w)
            axis_tgt = get_dir_vector(kpts_tgt_w)

            tip_pos_to_opt = get_tip_pos(kpts_to_opt_w, index=0)
            tip_pos_tgt = get_tip_pos(kpts_tgt_w, index=2)

            # a_axis, b_axis, a_pos, b_pos, a_start, b_start, b_end, use_pos_loss

            loss = calculate_loss(
                a_axis=axis_to_opt, 
                b_axis=axis_tgt,
                a_pos=tip_pos_to_opt, 
                b_pos=tip_pos_tgt,
                a_start=tip_pos_to_opt,
                b_start=kpts_tgt_w[0, :],
                b_end=kpts_tgt_w[2, :],
                use_pos_loss=(idx==num_keyframes-1)
            )
            reg = calculate_reg(
                rot_to_opt, init_rot,
                trans_to_opt, init_trans
            )

            total_loss = loss + 0.1 * reg
            total_loss.backward()
            if step % 500 == 0:
                print(f"Step {step} | Loss: {total_loss.item():.4f}")

            optimizer.step()
            
            if total_loss.item() < min_loss:
                min_loss = total_loss.item()
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= early_stop_tol:
                print(f"Early stop at step: {step}, min loss: {min_loss:.3f}")
                break
                
        
        mat = np.eye(4)
        rot_mat = axis_angle_to_matrix(rot_to_opt).squeeze().detach().cpu().numpy()
        trans = trans_to_opt.squeeze().detach().cpu().numpy()
        mat[:3, :3] = rot_mat
        mat[:3, 3] = trans
        if args.mode == 'ro':
            refine_right_object_poses[keyframe_indices[idx], :, :] = mat
            np.save(f"{args.data_dir}/{args.seq_name}/processed/object/{right_object_name}_pose_cam.refine.npy", refine_right_object_poses)
            np.save(f"{args.data_dir}/{args.seq_name}/processed/object/{left_object_name}_pose_cam.refine.npy", refine_left_object_poses)
        elif args.mode == 'lo':
            refine_left_object_poses[keyframe_indices[idx], :, :] = mat
            np.save(f"{args.data_dir}/{args.seq_name}/processed/object/{right_object_name}_pose_cam.refine.npy", refine_right_object_poses)
            np.save(f"{args.data_dir}/{args.seq_name}/processed/object/{left_object_name}_pose_cam.refine.npy", refine_left_object_poses)
        

        


            




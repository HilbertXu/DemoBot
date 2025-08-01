from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO 
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt

LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)


def visualize_hand_vertices(image, pts_right, pts_left, point_size=1, save_f=None):
    """
    image: np.array of shape [H, W, 3] or [H, W]
    vertices_2d: np.array of shape [N, 2], values in pixel coordinates
    """
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.scatter(pts_right[:, 0], pts_right[:, 1], s=point_size**2, c='red')
    plt.scatter(pts_left[:, 0], pts_left[:, 1], s=point_size**2, c='blue')
    plt.axis('off')
    plt.savefig(save_f, bbox_inches='tight', dpi=150)
    
    plt.close(fig)
    

 # Adopted from Hamer
def to_xy_batch(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 3
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
    zz = x_homo[:, :, 2:3]
    
    
    x = x_homo[:, :, :2] / zz
    return x

def project2d_batch(K, pts_cam):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    pts2d_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)
    return pts2d

# reform the outputs and save them
def reform_pred_list(pred_list, im_paths):
    # For compitable with MANO hand
    j2d_mapping = np.asarray([
        0, 5, 6, 7, 
        9, 10, 11,
        17, 18, 19,
        13, 14, 15, 
        1, 2, 3, 4, 
        8, 12, 16, 20
    ], dtype=int)
    verts_r = np.zeros((len(im_paths), 778, 3))*np.nan
    verts_l = np.copy(verts_r)
    global_orient_r = np.zeros((len(im_paths), 1, 3, 3))*np.nan
    hand_pose_r = np.zeros((len(im_paths), 15, 3, 3))*np.nan
    betas_r = np.zeros((len(im_paths), 10))*np.nan
    
    joints_r = np.zeros((len(im_paths), 21, 3))*np.nan
    joints_l = np.copy(joints_r)
    global_orient_l = np.zeros((len(im_paths), 1, 3, 3))*np.nan
    hand_pose_l = np.zeros((len(im_paths), 15, 3, 3))*np.nan
    betas_l = np.zeros((len(im_paths), 10))*np.nan


    for pred_dict in pred_list:
        is_right = bool(pred_dict['is_right'])

        v3d_cam = pred_dict['verts']  + pred_dict['cam_t.full'][None, :]
        j3d_cam = pred_dict['jts']  + pred_dict['cam_t.full'][None, :]

        idx = im_paths.index(pred_dict['img_path'])

        if is_right:
            verts_r[idx] = v3d_cam
            joints_r[idx] = j3d_cam
            global_orient_r[idx] = pred_dict['global_orient']
            betas_r[idx] = pred_dict['betas']
            hand_pose_r[idx] = pred_dict['hand_pose']
        else:
            verts_l[idx] = v3d_cam
            joints_l[idx] = j3d_cam
            global_orient_l[idx] = pred_dict['global_orient']
            betas_l[idx] = pred_dict['betas']
            hand_pose_l[idx] = pred_dict['hand_pose']

    verts_r = verts_r.astype(np.float32)
    verts_l = verts_l.astype(np.float32)
    joints_r = joints_r.astype(np.float32)
    joints_l = joints_l.astype(np.float32)
    
    global_orient_l = global_orient_l.astype(np.float32)
    betas_l = betas_l.astype(np.float32)
    hand_pose_l = hand_pose_l.astype(np.float32)
    
    global_orient_r = global_orient_r.astype(np.float32)
    betas_r = betas_r.astype(np.float32)
    hand_pose_r = hand_pose_r.astype(np.float32)
    
    
    K = torch.FloatTensor(pred_list[0]['K'])
    joints_r = torch.FloatTensor(joints_r)
    joints_l = torch.FloatTensor(joints_l)
    verts_r = torch.FloatTensor(verts_r)
    verts_l = torch.FloatTensor(verts_l)
    
    v2d_r = project2d_batch(K[None, :, :].repeat(verts_r.shape[0], 1, 1), verts_r).numpy()
    v2d_l = project2d_batch(K[None, :, :].repeat(verts_l.shape[0], 1, 1), verts_l).numpy()
    j2d_r = project2d_batch(K[None, :, :].repeat(joints_r.shape[0], 1, 1), joints_r).numpy()
    j2d_l = project2d_batch(K[None, :, :].repeat(joints_l.shape[0], 1, 1), joints_l).numpy()    
    

    results_3d = {}
    results_3d['v3d.right'] = verts_r
    results_3d['v3d.left'] = verts_l
    results_3d['j3d.right'] = joints_r
    results_3d['j3d.left'] = joints_l
    results_3d['im_paths'] = im_paths
    results_3d['K'] = pred_list[0]['K']
    
    results_2d = {}
    results_2d['v2d.right'] = v2d_r
    results_2d['v2d.left'] = v2d_l
    results_2d['j2d.right'] = j2d_r[:, j2d_mapping, :]
    results_2d['j2d.left'] = j2d_l[:, j2d_mapping, :]
    results_2d['im_paths'] = im_paths
    
    results_mano = {}
    results_mano['global_orient.right'] = global_orient_r
    results_mano['betas.right'] = betas_r
    results_mano['hand_pose.right'] = hand_pose_r
    results_mano['global_orient.left'] = global_orient_l
    results_mano['betas.left'] = betas_l
    results_mano['hand_pose.left'] = hand_pose_l
    
    return results_3d, results_2d, results_mano


def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--data_dir', type=str, default='./', help='Root directory of the dataset')
    parser.add_argument('--seq_name', type=str, default='images', help='name of the data folder')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')

    args = parser.parse_args()
    args.img_folder = f'{args.data_dir}/{args.seq_name}/rgb'
    
    
    v2d_out_dir = f'{args.data_dir}/{args.seq_name}/processed/hand_v2d'
    j2d_out_dir = f'{args.data_dir}/{args.seq_name}/processed/hand_j2d'
    
    os.makedirs(v2d_out_dir, exist_ok=True)
    os.makedirs(j2d_out_dir, exist_ok=True)

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output directory if it does not exist
    mesh_out_folder = f'{args.data_dir}/{args.seq_name}/processed/wilor_mesh'
    wilor_vis_out_folder = f'{args.data_dir}/{args.seq_name}/processed/wilor_vis'
    os.makedirs(mesh_out_folder, exist_ok=True)
    os.makedirs(wilor_vis_out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = sorted(glob(f'{args.img_folder}/*.png'))
    pred_list = []
    # Iterate over all images in folder
    for img_path in tqdm(img_paths):
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(img_cv2, conf = 0.15, verbose=False)[0]
        bboxes    = []
        is_right  = []
        for det in detections: 
            # Apply threshold filter to ignore low-conf hands
            if det.boxes.conf.cpu().detach().squeeze().item() < 0.15:
                continue
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        if len(bboxes) == 0:
            continue
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []
        
        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right    = batch['right'][n].cpu().numpy()
                verts[:,0]  = (2*is_right-1)*verts[:,0]
                joints[:,0] = (2*is_right-1)*joints[:,0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                pred_mano_global_orient = out['pred_mano_params']['global_orient'][n].cpu().numpy()
                pred_mano_betas = out['pred_mano_params']['betas'][n].cpu().numpy()
                pred_mano_hand_pose = out['pred_mano_params']['hand_pose'][n].cpu().numpy()
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                
                pred_dict = {
                    'img_path': img_path,
                    'verts': verts,
                    'jts': joints,
                    'cam_t.full': cam_t,
                    'global_orient': pred_mano_global_orient,
                    'betas': pred_mano_betas,
                    'hand_pose': pred_mano_hand_pose,
                    'is_right': is_right
                }
                
                fx = fy = float(scaled_focal_length.cpu().numpy())
                cx, cy = img_size[n].cpu().detach().numpy() / 2
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                pred_dict['K'] = K
                
                pred_list.append(pred_dict)
                
                
                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    tmesh.export(os.path.join(mesh_out_folder, f'{img_fn}_{n}.obj'))

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(wilor_vis_out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])
    
    
    # reform the outputs and save them 
    results_3d, results_2d, results_mano = reform_pred_list(pred_list, img_paths)
    
    for im_path, v2d_right, v2d_left in zip(results_2d['im_paths'], results_2d['v2d.right'], results_2d['v2d.left']):
        img = cv2.imread(im_path)[:, :, ::-1]
        img_name = str(im_path).split('/')[-1]
        visualize_hand_vertices(
            img, v2d_right, v2d_left, save_f=f'{v2d_out_dir}/{img_name}'
        )
        

    for im_path, j2d_right, j2d_left in zip(results_2d['im_paths'], results_2d['j2d.right'], results_2d['j2d.left']):
        img = cv2.imread(im_path)[:, :, ::-1]
        img_name = str(im_path).split('/')[-1]
        visualize_hand_vertices(
            img, j2d_right, j2d_left, save_f=f'{j2d_out_dir}/{img_name}'
        )

    np.save(
        f'{args.data_dir}/{args.seq_name}/processed/v3d.npy',
        results_3d
    )
    np.save(
        f'{args.data_dir}/{args.seq_name}/processed/j2d.full.npy',
        results_2d
    )
    
    np.save(
        f'{args.data_dir}/{args.seq_name}/processed/mano.init.npy',
        results_mano
    )

def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

if __name__ == '__main__':
    main()

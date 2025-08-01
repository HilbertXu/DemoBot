import torch
import torch.nn as nn
from common.xdict import xdict
from common.transforms import project2d_batch
from src.alignment.loss_terms import gmof
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import axis_angle_to_matrix


l1_loss = nn.L1Loss(reduction="none")


class ObjectParameters(nn.Module):
    def __init__(self, data, meta):
        super().__init__()
        # unpacking
        K = meta["K"]
        o2w_all = data["o2w_all"]
        obj_rot = matrix_to_axis_angle(o2w_all[:, :3, :3])
        obj_transl = o2w_all[:, :3, 3]
        obj_cano = data['object_cano']

        # object parameters
        obj_scale = torch.FloatTensor(np.array([1.0]))

        # self.register_parameter("obj_scale", nn.Parameter(obj_scale))
        self.register_parameter("obj_rot", nn.Parameter(obj_rot))
        self.register_parameter("obj_transl", nn.Parameter(obj_transl))

        self.register_buffer("obj_scale", nn.Parameter(obj_scale))
        self.register_buffer("obj_cano", obj_cano)

        self.K = K

        targets = xdict()
        self.targets = targets
        self.im_paths = meta["im_paths"]

    def forward(self):
        num_frames = len(self.obj_rot)
        device = self.obj_rot.device

        rot_mat = axis_angle_to_matrix(self.obj_rot)

        K = self.K[None, :, :].repeat(num_frames, 1, 1).to(device)

        obj_cano = self.obj_cano.clone() * self.obj_scale
        obj_cano = obj_cano.T[None, :, :].repeat(num_frames, 1, 1)
        
        # rotate
        pts_w = torch.bmm(rot_mat, obj_cano)
        pts_w = pts_w + self.obj_transl[:, :, None]
        pts_w = pts_w.permute(0, 2, 1)

        # divided by zero
        pts_w_results = pts_w.clone()
        pts_w_results[pts_w[:, :, 2] == 0.0] = 1e-8
        pts_w = pts_w_results

        out = xdict()
        o2d = project2d_batch(K, pts_w)

        out["j3d"] = pts_w
        out["j2d"] = o2d
        out["im_paths"] = self.im_paths
        out["K"] = self.K
        out["obj_scale"] = self.obj_scale

        o2w_all = torch.eye(4, device=device).unsqueeze(0).repeat(num_frames, 1, 1)
        o2w_all[:, :3, :3] = rot_mat
        o2w_all[:, :3, 3] = self.obj_transl
        out["o2w_all"] = o2w_all

        return out

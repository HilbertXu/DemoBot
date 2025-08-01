import torch.nn as nn
import torch

import torch
import torch.nn as nn
import sys
from src.alignment.loss_terms import gmof
import matplotlib.pyplot as plt
import cv2

sys.path = [".."] + sys.path

mse_loss = nn.MSELoss(reduction="none")
l1_loss = nn.L1Loss(reduction="none")


def loss_fn_h(preds, targets, conf):
    # op2d
    loss = 0.0
    device = preds["right.j2d"].device
    
    targets_j2d_r = targets["right.j2d.gt"].to(device)
    is_valid = ~torch.isnan(targets_j2d_r[:, 0, 0])
    loss_2d_r = gmof(
        preds["right.j2d"][is_valid] - targets_j2d_r[is_valid, :, :2],
        sigma=conf.j2d_sigma,
    ).sum(dim=-1)
    loss_2d_r = loss_2d_r.mean() * conf.j2d

    loss += loss_2d_r

    targets_j2d_l = targets["left.j2d.gt"].to(device)
    is_valid = ~torch.isnan(targets_j2d_l[:, 0, 0])
    loss_2d_l = gmof(
        preds["left.j2d"][is_valid] - targets_j2d_l[is_valid, :, :2], sigma=conf.j2d_sigma
    ).sum(dim=-1)
    loss_2d_l = loss_2d_l.mean() * conf.j2d
    loss += loss_2d_l
    loss /= 2.0
    return loss


def loss_fn_o(preds, targets, conf):
    targets_ro2d = targets["right_object.j2d.gt"]
    targets_lo2d = targets["left_object.j2d.gt"]
    r3d = targets["right.j3d"]
    l3d = targets["left.j3d"]
    ro3d = preds["right_object.j3d"]
    lo3d = preds["left_object.j3d"]

    # coarse contact
    centroid_r = r3d.mean(dim=1)
    centroid_l = l3d.mean(dim=1)
    centroid_ro = ro3d.mean(dim=1)
    centroid_lo = lo3d.mean(dim=1)
    
    loss_r = l1_loss(centroid_r, centroid_ro).mean() * conf.contact
    loss_l = l1_loss(centroid_l, centroid_lo).mean() * conf.contact
    loss = (loss_r + loss_l) / 2.0

    # 2d reprojection
    loss += (
        gmof(preds["right_object.j2d"] - targets_ro2d, sigma=conf.o2d_sigma).sum(dim=-1).mean()
        * conf.o2d
    )
    loss += (
        gmof(preds["left_object.j2d"] - targets_lo2d, sigma=conf.o2d_sigma).sum(dim=-1).mean()
        * conf.o2d
    )

    # encourage: in front of camera
    z_min = torch.clamp(-o3d[:, :, 2].mean(dim=1), min=0.0)
    if z_min.sum() > 0:
        loss_z = z_min.sum() / torch.nonzero(z_min).shape[0]
        loss += loss_z * conf.z_min
    return loss


def loss_fn_ho(preds, targets, conf):
    v3d_r = preds["right.v3d"]
    v3d_l = preds["left.v3d"]
    v3d_ro = preds["right_object.j3d"]
    v3d_lo = preds["left_object.j3d"]

    centroid_r = v3d_r.mean(dim=1)
    centroid_l = v3d_l.mean(dim=1)
    centroid_ro = v3d_ro.mean(dim=1)
    centroid_lo = v3d_lo.mean(dim=1)
    diff_r = centroid_r[:-1] - centroid_r[1:]
    diff_l = centroid_l[:-1] - centroid_l[1:]
    diff_ro = centroid_ro[:-1] - centroid_ro[1:]
    diff_lo = centroid_lo[:-1] - centroid_lo[1:]
    loss_smooth_r = mse_loss(diff_r, torch.zeros_like(diff_r).detach()).mean()
    loss_smooth_l = mse_loss(diff_l, torch.zeros_like(diff_l).detach()).mean()
    loss_smooth_ro = mse_loss(diff_ro, torch.zeros_like(diff_ro).detach()).mean()
    loss_smooth_lo = mse_loss(diff_lo, torch.zeros_like(diff_lo).detach()).mean()
    loss = loss_smooth_r + loss_smooth_l + loss_smooth_ro / 3.0 + loss_smooth_lo / 3.0
    loss = loss * 100.0
    return loss


from src.alignment.pl_module.generic_module import PLModule


class H2O2Module(PLModule):
    def __init__(self, data, args, conf):
        super().__init__(data, args, conf, loss_fn_h, loss_fn_o, loss_fn_ho)

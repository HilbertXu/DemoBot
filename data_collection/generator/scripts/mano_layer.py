# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Wrapper layer for manopth ManoLayer."""

import torch

from torch.nn import Module
from manopth.manolayer import ManoLayer


class MANOLayer(Module):
    """Wrapper layer for manopth ManoLayer."""

    def __init__(self, mano_root, side, betas, use_pca=True, ncomps=45):
        """Constructor.
        Args:
          side: MANO hand type. 'right' or 'left'.
          betas: A numpy array of shape [10] containing the betas.
        """
        super(MANOLayer, self).__init__()

        self._side = side
        self._betas = betas
        if use_pca:
            self._mano_layer = ManoLayer(flat_hand_mean=False,
                                        ncomps=ncomps,
                                        side=self._side,
                                        mano_root=mano_root,
                                        use_pca=use_pca)
        else:
            self._mano_layer = ManoLayer(flat_hand_mean=False,
                                        side=self._side,
                                        mano_root=mano_root)

        b = torch.from_numpy(self._betas).unsqueeze(0)
        f = self._mano_layer.th_faces
        self.register_buffer('b', b)
        self.register_buffer('f', f)

        v = torch.matmul(self._mano_layer.th_shapedirs, self.b.transpose(
            0, 1)).permute(2, 0, 1) + self._mano_layer.th_v_template
        r = torch.matmul(self._mano_layer.th_J_regressor[0], v)
        self.register_buffer('root_trans', r)

    def forward(self, p, t):
        """Forward function.
        Args:
          p: A tensor of shape [B, 48] containing the pose.
          t: A tensor of shape [B, 3] containing the trans.
        Returns:
          v: A tensor of shape [B, 778, 3] containing the vertices.
          j: A tensor of shape [B, 21, 3] containing the joints.
        """
        if not isinstance(p, torch.Tensor):
            p = torch.from_numpy(p)
        if not isinstance(t, torch.Tensor):
            t = torch.from_numpy(t)
        if len(p.shape) == 1:
            p = p.unsqueeze(0)
        if len(t.shape) == 1:
            t = t.unsqueeze(0)
            
        v, j = self._mano_layer(p, self.b.expand(p.size(0), -1), t)
        v /= 1000
        j /= 1000
        return v, j
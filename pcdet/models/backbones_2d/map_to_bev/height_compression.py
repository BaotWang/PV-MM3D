import torch.nn as nn
import numpy as np
from pcdet.datasets.augmentor.X_transform import X_TRANS
import torch
from tools.visual_utils.open3d_vis_utils import draw_scenes


class HeightCompression(nn.Module):
    def __init__(self, model_cfg,  voxel_size=None, point_cloud_range=None):
        super().__init__()
        self.model_cfg = model_cfg

        # NUM_BEV_FEATURES: 256
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

        self.x_trans = X_TRANS()
        self.point_cloud_range = point_cloud_range
        self.voxel_size=voxel_size

    def forward(self, batch_dict):

        #
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        # SparseConvTensor[shape=torch.Size([19621, 64])]; spatial_shape:(4,200,176) 这是 真实点云与虚拟点云混合在一起的 稀疏tensors
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # spatial_features:(bs, 64, 4, 200, 176); sparse tensor 转 dense tensor
        spatial_features = encoded_spconv_tensor.dense()
        # C 特征数:64； D:4; H:200; W:176
        N, C, D, H, W = spatial_features.shape
        # spatial_features:(bs, 64*4, 200, 176); 高度压缩 成为 BEV特征
        spatial_features = spatial_features.view(N, C * D, H, W)

        # spatial_features: (bs, 256, 200, 176)
        batch_dict['spatial_features'] = spatial_features

        return batch_dict

import torch
from tools.visual_utils.open3d_vis_utils import draw_scenes

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.model = self.model_cfg.get('MODEL',None)

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """

        # False
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        # i=0; frame_id = ''
        for i in range(rot_num):
            if i==0:
                frame_id = ''
            else:
                frame_id = str(i)

            # voxel_features:(num_voxels, 5, 8); (eg:num_voxels=76891)
            # voxel_num_points:76891
            voxel_features, voxel_num_points = batch_dict['voxels'+frame_id], batch_dict['voxel_num_points'+frame_id]
            # 对 max_points_per_voxel 的所有点求和（最多为 5个点）
            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
            #
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            # points_mean:(num_voxels,8)
            points_mean = points_mean / normalizer

            # True
            if self.model is not None:
                # True
                if self.model == 'max':
                    # time_max:(num_voxels,8)
                    time_max = voxel_features[:, :, :].max(dim=1, keepdim=False)[0]
                    #
                    points_mean[:, -1] = time_max[:, -1]

            # 存储 voxel_features:(num_voxels,8) 的特征，并使用 contiguous() 函数，使得值连续在一起
            batch_dict['voxel_features'+frame_id] = points_mean.contiguous()

            # False
            if 'mm' in batch_dict:
                voxel_features, voxel_num_points = batch_dict['voxels_mm'+frame_id], batch_dict[
                    'voxel_num_points_mm'+frame_id]
                points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
                normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
                points_mean = points_mean / normalizer

                batch_dict['voxel_features_mm'+frame_id] = points_mean.contiguous()

        return batch_dict

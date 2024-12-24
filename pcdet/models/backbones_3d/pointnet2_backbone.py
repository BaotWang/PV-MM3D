import torch
import torch.nn as nn
import numpy as np

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack
from tools.visual_utils.open3d_vis_utils import draw_scenes


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            #
            self.SA_modules.append(
                #
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out

            #
            self.FP_modules.append(
                #
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

        # 获取虚拟点云的采样模式
        self.sampling_mode = self.model_cfg.get("Sampling_mode", None)

        # 如果是 part-based
        if self.sampling_mode == 'part-based':
            self.bin_num = self.model_cfg.get("BIN_NUM", None)
            self.each_bin_filter = self.model_cfg.get("EACH_BIN_FILTER", None)
            self.sampling_type = self.model_cfg.get("SAMPLE_TYPE", None)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']

        # 求要求为 16384(每帧) 个点; 获取 虚拟点云
        points_mm = batch_dict['points_mm']

        # 对输入数据进行 初步采样
        if self.sampling_mode == 'part-based':
            # (bs*16384, 5)
            points_mm = self.input_point_discard(points_mm, every_frame_points_num=16384, bs=batch_size)[:, :5]

        else:
            # (bs*16384, 5)
            points_mm = self.process_points_mm(raw_points_mm=points_mm, batch_size=batch_size)[:, :5]

        # xyz:(bs*16384, 3)
        batch_idx, xyz, features = self.break_up_pc(points_mm)

        # 验证每个批次的大小是否一致
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()

        # xyz:(bs, 16384, 3)
        xyz = xyz.view(batch_size, -1, 3)
        # features:(bs, 1, 16384)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None

        l_xyz, l_features = [xyz], [features]

        #  ModuleList:4 ---> PointnetSAModuleMSG
        for i in range(len(self.SA_modules)):
            # i=0:(bs,4096,3); (bs,32+64,4096);  i=1:(bs,1024,3);(bs,128+128,1024)
            # i=2:(bs,256,3); (bs,256+256,256);  i=3:(bs,64,3);(bs,512+512,64)
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])

            l_xyz.append(li_xyz)
            l_features.append(li_features)

        #
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            #
            #
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])  # (B, C, N)

        # (bs, 16384, 128); (B, N, C)
        point_features = l_features[0].permute(0, 2, 1).contiguous()

        # point_mm_features:(bs*16384, 128)
        batch_dict['point_mm_features'] = point_features.view(-1, point_features.shape[-1])

        # point_mm_features:(bs*16384, 4)
        batch_dict['point_mm_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)

        return batch_dict

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    """
    process_points_mm:
        默认的虚拟点云采样策略

    """

    def process_points_mm(self, num_points=16384, raw_points_mm=None, batch_size=1):

        points_mm_list = []

        for bs_idx in range(batch_size):

            if isinstance(raw_points_mm, torch.Tensor) and raw_points_mm.is_cuda:
                points_mm_mask = raw_points_mm[:, 0] == bs_idx
                points_mm = raw_points_mm[points_mm_mask].cpu().numpy()

            if num_points < len(points_mm):
                pts_depth = np.linalg.norm(points_mm[:, 0:3], axis=1)
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                choice = []
                if num_points > len(far_idxs_choice):
                    near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                    choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                        if len(far_idxs_choice) > 0 else near_idxs_choice
                else:
                    choice = np.arange(0, len(points_mm), dtype=np.int32)
                    choice = np.random.choice(choice, num_points, replace=False)
                np.random.shuffle(choice)
                points_mm_list.append(points_mm[choice])

            else:
                choice = np.arange(0, len(points_mm), dtype=np.int32)
                if num_points > len(points_mm):
                    extra_choice = np.random.choice(choice, num_points - len(points_mm), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

                points_mm_list.append(points_mm[choice])

        return torch.tensor(np.concatenate(points_mm_list)).cuda()

    def partition(self, points, num=10, max_dis=60, rate=0.2):
        """
        partition the points into several bins.
        """

        points_list = []
        inter = max_dis / num

        all_points_num = points.shape[0]

        points_num_acc = 0

        position = num - 1

        distant_points_num_acc = 0

        for i in range(num):
            i = num - i - 1
            if i == num - 1:
                min_mask = points[:, 1] >= inter * i
                this_points = points[min_mask]

                points_num_acc += this_points.shape[0]

                sampled_sum = points_num_acc + i * this_points.shape[0]

                if sampled_sum / all_points_num < rate:
                    position = i
                    distant_points_num_acc = points_num_acc

                points_list.append(this_points)
            else:
                min_mask = points[:, 1] >= inter * i
                max_mask = points[:, 1] < inter * (i + 1)
                mask = min_mask * max_mask
                this_points = points[mask]

                points_num_acc += this_points.shape[0]

                sampled_sum = points_num_acc + i * this_points.shape[0]

                if sampled_sum / all_points_num < rate:
                    position = i
                    distant_points_num_acc = points_num_acc

                points_list.append(this_points)

        if position <= 0:
            position = 0

        return points_list, position, distant_points_num_acc

    def input_point_discard(self, points, bin_num=2, rate=0.8, every_frame_points_num=16384, bs=1):
        """
        discard points by a bin-based sampling.
        """
        retain_rate = 1 - rate
        all_frame_parts = []

        # 逐批次操作
        for bs_id in range(bs):
            cur_points = points[(points[:, :1] == bs_id).squeeze()]
            #
            parts, pos, distant_points_num_acc = self.partition(cur_points, num=bin_num, rate=retain_rate)

            # output_pts_num = int(points.shape[0] * retain_rate); 16384+1
            output_pts_num = every_frame_points_num + 1

            pts_per_bin_num = int((output_pts_num - distant_points_num_acc) / (pos + 0.0001))

            for i in range(len(parts) - pos, len(parts)):

                if parts[i].shape[0] > pts_per_bin_num:
                    rands = np.random.permutation(parts[i].shape[0])
                    parts[i] = parts[i][rands[:pts_per_bin_num]]
                else:
                    rands = np.random.permutation(pts_per_bin_num - parts[i].shape[0] + 1)
                    parts[i - 1] = parts[i - 1][rands[:pts_per_bin_num - parts[i].shape[0] + 1]]

            new_parts = []
            for j in range(len(parts)):
                part = parts[j].cpu().numpy()
                new_parts.append(part)

            all_frame_parts.append(torch.tensor(np.concatenate(new_parts)))

        return torch.cat(all_frame_parts, dim=0)


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules_stack.StackSAModuleMSG(
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules_stack.StackPointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points: (k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                    cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
                ).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](
                xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)

        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1],
                known=l_xyz[i], known_batch_cnt=l_batch_cnt[i],
                unknown_feats=l_features[i - 1], known_feats=l_features[i]
            )

        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict

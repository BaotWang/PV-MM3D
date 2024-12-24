import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, spconv_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from functools import partial
import pickle
import copy
from tools.visual_utils.open3d_vis_utils import draw_scenes

from pcdet.datasets.augmentor.X_transform import X_TRANS


class PositionalEmbedding(nn.Module):
    def __init__(self, demb=256):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=2):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        else:
            return pos_emb[:, None, :]


"""
CrossAttention():
    


"""
class CrossAttention(nn.Module):

    def __init__(self, hidden_dim, pos = True, head = 4):
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.pos_dim = 8
        self.pos = pos

        if self.pos:
            self.pos_en = PositionalEmbedding(self.pos_dim)

            self.Q_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        else:

            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.att = nn.MultiheadAttention(hidden_dim, head)
    # 这里的 inputs 与 Q_in 其实是一样的
    def forward(self, inputs, Q_in): # N,B,C
        # 160; 获取批次大小
        batch_size = inputs.shape[1]
        # 1; 获取序列长度
        seq_len = inputs.shape[0]

        # True
        if self.pos:
            pos_input = torch.from_numpy(np.arange(seq_len)+1).cuda()
            pos_input = self.pos_en(pos_input, batch_size)
            inputs_pos = torch.cat([inputs, pos_input], -1)
            pos_Q = torch.from_numpy(np.array([seq_len])).cuda()
            pos_Q = self.pos_en(pos_Q, batch_size)
            Q_in_pos = torch.cat([Q_in, pos_Q], -1)
        else:
            inputs_pos = inputs
            Q_in_pos = Q_in

        # (bs,160,256)
        Q = self.Q_linear(Q_in_pos)
        # (bs,160,256)
        K = self.K_linear(inputs_pos)
        # (bs,160,256)
        V = self.V_linear(inputs_pos)

        # 进行注意力机制的计算
        out = self.att(Q, K, V)
        # (bs,160,256)
        return out[0]


class Attention_Layer(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs): # B,K,N


        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs).permute(0, 2, 1)
        V = self.V_linear(inputs)

        alpha = torch.matmul(Q, K)

        alpha = F.softmax(alpha, dim=2)

        out = torch.matmul(alpha, V)

        out = torch.mean(out, -2)

        return out


def gen_sample_grid(rois, grid_size=7, grid_offsets=(0, 0), spatial_scale=1.):
    faked_features = rois.new_ones((grid_size, grid_size))
    N = rois.shape[0]
    dense_idx = faked_features.nonzero()  # (N, 2) [x_idx, y_idx]
    dense_idx = dense_idx.repeat(N, 1, 1).float()  # (B, 7 * 7, 2)

    local_roi_size = rois.view(N, -1)[:, 3:5]
    local_roi_grid_points = (dense_idx ) / (grid_size-1) * local_roi_size.unsqueeze(dim=1) \
                      - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 7 * 7, 2)

    ones = torch.ones_like(local_roi_grid_points[..., 0:1])
    local_roi_grid_points = torch.cat([local_roi_grid_points, ones], -1)

    global_roi_grid_points = common_utils.rotate_points_along_z(
        local_roi_grid_points.clone(), rois[:, 6]
    ).squeeze(dim=1)
    global_center = rois[:, 0:3].clone()
    global_roi_grid_points += global_center.unsqueeze(dim=1)

    x = global_roi_grid_points[..., 0:1]
    y = global_roi_grid_points[..., 1:2]

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    return x.view(grid_size**2, -1), y.view(grid_size**2, -1)


def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W        C,K,1,2   C,K,1,1

    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)# 49,K,1,1
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)

    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / W)  # normalize to between  0 and 1

    samples[:, :, :, 1] = (samples[:, :, :, 1] / H)  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1  # 49,K,1,2

    #B,C,H,W
    #B,H,W,2
    #B,C,H,W

    return torch.nn.functional.grid_sample(image, samples, align_corners=False)


"""
TEDMHead:
    


"""


class TEDMHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range=None, voxel_size=None,  num_class=1,
                 **kwargs):
        super().__init__(num_class=num_class,  model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        # self.pool_cfg_mm = model_cfg.ROI_GRID_POOL_MM
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        # LAYER_cfg_mm = self.pool_cfg_mm.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.rot_num = model_cfg.ROT_NUM

        self.x_trans_train = X_TRANS()

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        # c_out_mm = 0
        # self.roi_grid_pool_layers_mm = nn.ModuleList()
        # feat = self.pool_cfg_mm.get('FEAT_NUM', 1)
        # for src_name in self.pool_cfg_mm.FEATURES_SOURCE:
        #     mlps = LAYER_cfg_mm[src_name].MLPS
        #     for k in range(len(mlps)):
        #         mlps[k] = [input_channels[src_name]*feat] + mlps[k]
        #     pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
        #         query_ranges=LAYER_cfg_mm[src_name].QUERY_RANGES,
        #         nsamples=LAYER_cfg_mm[src_name].NSAMPLE,
        #         radii=LAYER_cfg_mm[src_name].POOL_RADIUS,
        #         mlps=mlps,
        #         pool_method=LAYER_cfg_mm[src_name].POOL_METHOD,
        #     )
        #
        #     self.roi_grid_pool_layers_mm.append(pool_layer)
        #
        #     c_out_mm += sum([x[-1] for x in mlps])

        self.shared_fc_layers = nn.ModuleList()

        for i in range(self.rot_num):
            GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layers.append(nn.Sequential(*shared_fc_list))
            break

        self.shared_fc_layers_mm = nn.ModuleList()

        # for i in range(self.rot_num):
        #     GRID_SIZE = self.model_cfg.ROI_GRID_POOL_MM.GRID_SIZE
        #     pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out_mm
        #     shared_fc_list = []
        #     for k in range(0, self.model_cfg.SHARED_FC.__len__()):
        #         shared_fc_list.extend([
        #             nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
        #             nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
        #             nn.ReLU(inplace=True)
        #         ])
        #         pre_channel = self.model_cfg.SHARED_FC[k]
        #
        #         if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
        #             shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        #     self.shared_fc_layers_mm.append(nn.Sequential(*shared_fc_list))
        #     break

        self.shared_channel = pre_channel

        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()

        for i in range(self.rot_num):
            pre_channel = self.model_cfg.SHARED_FC[-1] * 2 * 2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers.append(cls_fc_layers)

            pre_channel = self.model_cfg.SHARED_FC[-1] * 2 * 2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers.append(reg_fc_layers)
            break

        self.cls_layers_P = nn.ModuleList()
        self.reg_layers_P = nn.ModuleList()

        for i in range(self.rot_num):
            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers_P.append(cls_fc_layers)

            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers_P.append(reg_fc_layers)
            break

        self.cls_layers_PI = nn.ModuleList()
        self.reg_layers_PI = nn.ModuleList()

        for i in range(self.rot_num):
            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers_PI.append(cls_fc_layers)

            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers_PI.append(reg_fc_layers)
            break

        if self.model_cfg.get('PART', False):
            self.grid_offsets = self.model_cfg.PART.GRID_OFFSETS
            self.featmap_stride = self.model_cfg.PART.FEATMAP_STRIDE
            part_inchannel = self.model_cfg.PART.IN_CHANNEL
            self.num_parts = self.model_cfg.PART.SIZE ** 2

            self.conv_part = nn.Sequential(
                nn.Conv2d(part_inchannel, part_inchannel, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(part_inchannel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.Conv2d(part_inchannel, self.num_parts, 1, 1, padding=0, bias=False),
            )

            self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets,
                                   spatial_scale=1 / self.featmap_stride)

        self.cross_attention_layers = nn.ModuleList()
        for i in range(self.rot_num):
            this_mo = CrossAttention(self.shared_channel)
            self.cross_attention_layers.append(this_mo)
            break

        self.cross_attention_layers_mm = nn.ModuleList()
        for i in range(self.rot_num):
            this_mo = CrossAttention(self.shared_channel)
            self.cross_attention_layers_mm.append(this_mo)
            break

        self.init_weights()
        self.ious = {0: [], 1: [], 2: [], 3: []}

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.cls_layers, self.reg_layers]:
            for stage_module in module_list:
                for m in stage_module.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        for module_list in [self.cls_layers, self.reg_layers]:
            for stage_module in module_list:
                nn.init.normal_(stage_module[-1].weight, 0, 0.01)
                nn.init.constant_(stage_module[-1].bias, 0)
        for m in self.shared_fc_layers.modules():
            if isinstance(m, nn.Linear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def obtain_conf_preds(self, confi_im, anchors):

        confi = []

        for i, im in enumerate(confi_im):
            boxes = anchors[i]
            im = confi_im[i]
            if len(boxes) == 0:
                confi.append(torch.empty(0).type_as(im))
            else:
                (xs, ys) = self.gen_grid_fn(boxes)
                out = bilinear_interpolate_torch_gridsample(im, xs, ys)
                x = torch.mean(out, 0).view(-1, 1)
                confi.append(x)

        confi = torch.cat(confi)

        return confi

    def roi_part_pool(self, batch_dict, parts_feat):
        rois = batch_dict['rois_score'].clone()
        confi_preds = self.obtain_conf_preds(parts_feat, rois)

        return confi_preds

    """
    roi_grid_pool():
        该函数是一个区域卷积池化（ROI Grid Pooling）操作，
        主要用于将给定的RoIs（Region of Interests）中的点云数据进行网格化处理，
        并对每个网格进行特征池化操作
    
    """
    def roi_grid_pool(self, batch_dict, i):
        """
        rois: (B, num_rois, 7 + C)
        point_coords: (num_points, 4)  [bs_idx, x, y, z]
        point_features: (num_points, C)
        point_cls_scores: (N1 + N2 + N3 + ..., 1)
        point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        # True
        if i==0:
            rot_num_id = ''
        else:
            rot_num_id = str(i)

        # rois:(bs,160,7); 获取 rois
        rois = batch_dict['rois'].clone()

        batch_size = batch_dict['batch_size']
        # False
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        ''' get_global_grid_points_of_roi(): 得到roi的全局网格点 '''
        # roi_grid_xyz:(bs*160, 6x6x6, 3) <--- (BxN, 6x6x6, 3)
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            # (bs, 160, 7)
            rois,
            # GRID_SIZE: 6
            grid_size=self.pool_cfg.GRID_SIZE
        )

        # roi_grid_xyz: (bs*160, 6x6x6, 3) ---> (bs, 160x6x6x6, 3); 160x6x6x6 = 34560
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        ''' compute the voxel coordinates of grid points: 计算网格点的体素坐标 '''
        # (bs,34560,1)
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        # (bs,34560,1)
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        # (bs,34560,1)
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]

        # roi_grid_coords: (bs, 34560, 1+1+1)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)
        # (bs,34560,1)
        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)

        # 逐批次操作
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx

        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_batch_cnt:初始化每个批次的roi_grid数量
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        #
        pooled_features_list = []

        # 特征来源：['x_conv3', 'x_conv4']
        #
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            # NeighborVoxelSAModuleMSG
            pool_layer = self.roi_grid_pool_layers[k]
            #
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                # src_name = x_conv3 ---> cur_stride:4
                # src_name = x_conv4 ---> x_conv4: 8
                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                j=i
                while 'multi_scale_3d_features'+rot_num_id not in batch_dict:
                    j-=1
                    rot_num_id = str(j)
                # 这里 cur_sp_tensors 获取值其实不起作用，下面的操作也会重新覆盖掉
                cur_sp_tensors = batch_dict['multi_scale_3d_features'+rot_num_id][src_name]

                # False
                if with_vf_transform:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]

                else:
                    # 'x_conv3': SparseConvTensor[shape=torch.Size([80787, 64])]
                    cur_sp_tensors = batch_dict['multi_scale_3d_features'+rot_num_id][src_name]

                ''' compute voxel center xyz and batch_cnt：计算体素中心xyz和 '''
                # cur_coords:(80787, 4); 获取当前的稀疏tensor的索引位置; 4:(bs_idx,x,y,z)
                cur_coords = cur_sp_tensors.indices

                ''' get_voxel_centers(): 获取体素中心'''
                # cur_voxel_xyz:(80787, 3)
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )
                # 初始化每个批次的体素中心: [0,0]
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()

                # 逐批次计算
                for bs_idx in range(batch_size):
                    # [40609,40178]
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

                ''' get voxel2point tensor(): '''
                # v2p_ind_tensor:(bs, 21, 400, 352);用于在指定形状的张量中，根据提供的多维索引将特定点的值设置为特定值
                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                ''' compute the grid coordinates in this scale, in [batch_idx, x y z] order '''
                # (bs, 34560, 3)
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                # (bs, 34560, 1+3)
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                # (bs, 34560, 4)
                cur_roi_grid_coords = cur_roi_grid_coords.int()

                ''' voxel neighbor aggregation '''
                # pooled_features:(34560, 64)
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                # pooled_features:(160, 216, 64)
                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                #
                pooled_features_list.append(pooled_features)
        # ms_pooled_features:(160, 216, 64+64)
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    """
    roi_grid_pool_mm():
        
    
    
    """
    # def roi_grid_pool_mm(self, batch_dict, i):
    #     """
    #     rois: (B, num_rois, 7 + C)
    #     point_coords: (num_points, 4)  [bs_idx, x, y, z]
    #     point_features: (num_points, C)
    #     point_cls_scores: (N1 + N2 + N3 + ..., 1)
    #     point_part_offset: (N1 + N2 + N3 + ..., 3)
    #     """
    #
    #     if i==0:
    #         rot_num_id = ''
    #     else:
    #         rot_num_id = str(i)
    #
    #     # rois: (bs, 160, 7)
    #     rois = batch_dict['mm_rois'].clone()
    #
    #     batch_size = batch_dict['batch_size']
    #
    #     # False
    #     with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
    #
    #     ''' get_global_grid_points_of_roi(): '''
    #     # roi_grid_xyz:(160, 64, 3)
    #     roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
    #         rois, grid_size=self.pool_cfg_mm.GRID_SIZE
    #     )  # (BxN, 6x6x6, 3)
    #
    #     # roi_grid_xyz: (B, Nx6x6x6, 3)
    #     roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)
    #
    #     ''' compute the voxel coordinates of grid points '''
    #     #
    #     roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
    #     #
    #     roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
    #     #
    #     roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
    #
    #     # roi_grid_coords: (B, Nx6x6x6, 3)
    #     roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)
    #
    #     #
    #     batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
    #
    #     #
    #     for bs_idx in range(batch_size):
    #
    #         batch_idx[bs_idx, :, 0] = bs_idx
    #
    #     # roi_grid_coords: (B, Nx6x6x6, 4)
    #     # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
    #     # roi_grid_coords = roi_grid_coords.int()
    #     #
    #     roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])
    #
    #     #
    #     pooled_features_list = []
    #
    #     # ['x_conv3', 'x_conv4']
    #     for k, src_name in enumerate(self.pool_cfg_mm.FEATURES_SOURCE):
    #         #
    #         pool_layer = self.roi_grid_pool_layers_mm[k]
    #         #
    #         if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:
    #
    #             # x_conv3 ---> cur_stride:4
    #             cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
    #             # False
    #             if with_vf_transform:
    #                 cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
    #             else:
    #                 if 'multi_scale_3d_features_mm'+rot_num_id in batch_dict: # False
    #                     cur_sp_tensors = batch_dict['multi_scale_3d_features_mm'+rot_num_id][src_name]
    #                 else:  # 这里其实与上一步的真实点云中的操作是一样的
    #                     cur_sp_tensors = batch_dict['multi_scale_3d_features' + rot_num_id][src_name]
    #
    #             ''' compute voxel center xyz and batch_cnt '''
    #             #
    #             cur_coords = cur_sp_tensors.indices
    #
    #             ''' get_voxel_centers(): '''
    #             #
    #             cur_voxel_xyz = common_utils.get_voxel_centers(
    #                 cur_coords[:, 1:4],
    #                 downsample_times=cur_stride,
    #                 voxel_size=self.voxel_size,
    #                 point_cloud_range=self.point_cloud_range
    #             )
    #
    #             #
    #             cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
    #             #
    #             for bs_idx in range(batch_size):
    #                 cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
    #
    #             ''' get voxel2point tensor: generate_voxel2pinds() '''
    #             # v2p_ind_tensor:(bs, 21, 400, 352)
    #             v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)
    #
    #             ''' compute the grid coordinates in this scale, in [batch_idx, x y z] order '''
    #             # (bs, 34560, 3)
    #             cur_roi_grid_coords = roi_grid_coords // cur_stride
    #             # (bs, 34560, 1+3)
    #             cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
    #             #
    #             cur_roi_grid_coords = cur_roi_grid_coords.int()
    #
    #             ''' voxel neighbor aggregation '''
    #             #
    #             pooled_features = pool_layer(
    #                 xyz=cur_voxel_xyz.contiguous(),
    #                 xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
    #                 new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
    #                 new_xyz_batch_cnt=roi_grid_batch_cnt,
    #                 new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
    #                 features=cur_sp_tensors.features.contiguous(),
    #                 voxel2point_indices=v2p_ind_tensor
    #             )
    #
    #             # (160, 64, 64)
    #             pooled_features = pooled_features.view(
    #                 -1, self.pool_cfg_mm.GRID_SIZE ** 3,
    #                 pooled_features.shape[-1]
    #             )  # (BxN, 6x6x6, C)
    #             pooled_features_list.append(pooled_features)
    #
    #     # ms_pooled_features:(160,64,64+64)
    #     ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
    #
    #     return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def roi_x_trans(self, rois, rot_num_id, transform_param):
        while rot_num_id>=len(transform_param[0]):
            rot_num_id-=1

        batch_size = len(rois)
        rois = rois.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = rois[bt_i]
            bt_transform_param = transform_param[bt_i]
            previous_stage_param = bt_transform_param[rot_num_id-1]
            current_stage_param = bt_transform_param[rot_num_id]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': previous_stage_param})
            transed_roi = self.x_trans_train.forward_with_param({'boxes': transed_roi['boxes'],
                                                                  'transform_param': current_stage_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)

    def roi_score_trans(self, rois, rot_num_id, transform_param):
        while rot_num_id>=len(transform_param[0]):
            rot_num_id-=1

        batch_size = len(rois)
        rois = rois.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = rois[bt_i]
            bt_transform_param = transform_param[bt_i]
            previous_stage_param = bt_transform_param[0]
            current_stage_param = bt_transform_param[rot_num_id]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': current_stage_param})
            transed_roi = self.x_trans_train.forward_with_param({'boxes': transed_roi['boxes'],
                                                                  'transform_param': previous_stage_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)

    def pred_x_trans(self, preds, rot_num_id, transform_param):
        while rot_num_id>=len(transform_param[0]):
            rot_num_id-=1

        batch_size = len(preds)
        preds = preds.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = preds[bt_i]
            bt_transform_param = transform_param[bt_i]
            current_stage_param = bt_transform_param[rot_num_id]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': current_stage_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)

    """
    multi_grid_pool_aggregation():
    
    
    """
    def multi_grid_pool_aggregation(self, batch_dict, targets_dict):

        # False
        if self.model_cfg.get('PART', False):
            feat_2d = batch_dict['st_features_2d']
            parts_feat = self.conv_part(feat_2d)

        # 创建所有的 预测值、分数、rois
        all_preds = []
        all_scores = []
        all_rois = []

        # 创建所有共享特征列表
        all_shared_features = []
        #
        all_shared_features_mm = []

        # self.rot_num = 1;
        for i in range(self.rot_num):
            # rot_num_id = '0'
            rot_num_id = str(i)

            # False
            if i >= 1 and 'transform_param' in batch_dict:
                batch_dict['rois'] = self.roi_x_trans(batch_dict['rois'], i, batch_dict['transform_param'])

            # True
            if self.training:

                ''' assign_targets(): 将 512个rois 进一步细化成 160个rois'''
                #
                targets_dict = self.assign_targets(batch_dict, i)
                # 给 targets_dict 字典中赋值，强化参数
                targets_dict['aug_param'] = batch_dict['aug_param']
                # 给 targets_dict 字典中赋值，图片尺寸
                targets_dict['image_shape'] = batch_dict['image_shape']
                # 给 targets_dict 字典中赋值，校准对象
                targets_dict['calib'] = batch_dict['calib']
                # rois:(bs,160,7)；给 batch_dict 字典中 rois 更新值(512个rois 变成了 160rois)
                batch_dict['rois'] = targets_dict['rois']
                # roi_labels:(bs,160)；同 上述 rois
                batch_dict['roi_labels'] = targets_dict['roi_labels']

            # False
            if i >= 1 and 'transform_param' in batch_dict:
                batch_dict['rois_score'] = self.roi_score_trans(batch_dict['rois'], i, batch_dict['transform_param'])

            else:
                # rois_score:(bs,160,7)
                batch_dict['rois_score'] = batch_dict['rois']

            # False
            if self.model_cfg.get('PART', False):
                part_scores = self.roi_part_pool(batch_dict, parts_feat)

            # False
            if 'transform_param' in batch_dict:
                pooled_features = self.roi_grid_pool(batch_dict, i)
                pooled_features_mm = self.roi_grid_pool_mm(batch_dict, i)

            else:
                ''' roi_grid_pool():    真实点云 roi 网格池化'''
                # pooled_features:(bs*160, 216, 128); [x_conv3 + x_conv4]; 6*6*6 = 216
                pooled_features = self.roi_grid_pool(batch_dict, 0)

                ''' roi_grid_pool_mm(): 虚拟点云 roi 网格池化'''
                # pooled_features_mm:(bs*160, 64, 128); [x_conv3 + x_conv4]; 4*4*4 = 64
                # pooled_features_mm = self.roi_grid_pool_mm(batch_dict, 0)

            # pooled_features:(bs*160, 216*128); 216*128=27648  改变维度
            pooled_features = pooled_features.view(pooled_features.size(0), -1)

            ''' shared_fc_layers(): '''
            # shared_features:(bs*160, 256)
            shared_features = self.shared_fc_layers[0](pooled_features)
            # shared_features:(1, bs*160, 256)
            shared_features = shared_features.unsqueeze(0)  # 1,B,C
            #
            all_shared_features.append(shared_features)
            # 因为只有一个 shared_features，所以这里的 pre_feat 与 shared_features 是一样的
            pre_feat = torch.cat(all_shared_features, 0)

            ''' cross_attention_layers(): 这里的 pre_feat与shared_features 是一样的,所以这里的cross_attention其实就是 self-attention '''
            # cur_feat:(bs, 160, 256)
            cur_feat = self.cross_attention_layers[0](pre_feat, shared_features)
            # cur_feat:(bs, 160, 256+256)
            cur_feat = torch.cat([cur_feat, shared_features], -1)
            # cur_feat:(160, 512)
            cur_feat = cur_feat.squeeze(0)  # B, C*2

            # (bs*160, 64 * 128); 64 * 128 = 8192
            # pooled_features_mm = pooled_features_mm.view(pooled_features_mm.size(0), -1)

            ''' shared_fc_layers_mm(): '''
            # shared_features_mm:(bs*160, 256)
            # shared_features_mm = self.shared_fc_layers_mm[0](pooled_features_mm)

            # shared_features_mm:(1, bs*160, 256)
            shared_features_mm = batch_dict['shared_features_mm'].permute(2, 0, 1)

            # shared_features_mm = shared_features_mm.unsqueeze(0)  # 1,B,C
            #
            all_shared_features_mm.append(shared_features_mm)
            # pre_feat_mm:(); 因为只有一个 all_shared_features_mm，所以这里的 pre_feat_mm 与 all_shared_features_mm 是一样的
            pre_feat_mm = torch.cat(all_shared_features_mm, 0)

            ''' cross_attention_layers_mm(): 同上'''
            # cur_feat_mm:(1, bs*160, 256)
            cur_feat_mm = self.cross_attention_layers_mm[0](pre_feat_mm, shared_features_mm)
            # (1, bs*160, 256+256)
            cur_feat_mm = torch.cat([cur_feat_mm, shared_features_mm], -1)
            # (bs*160,512)
            cur_feat_mm = cur_feat_mm.squeeze(0)  # B, C*2

            # TODO 可以考虑跨模态融合，
            ''' 混合特征 '''
            # final_feat:(160, 512 + 512); 这里做的直接就是两个不同模态特征之间的融合
            final_feat = torch.cat([cur_feat_mm, cur_feat],-1)
            # rcnn_cls:(160,1); 分类预测
            rcnn_cls = self.cls_layers[0](final_feat)
            # rcnn_reg:(160,7); 回归预测
            rcnn_reg = self.reg_layers[0](final_feat)

            ''' 虚拟点云特征 '''
            # rcnn_cls_pi:(160, 1); 虚拟点云分类预测
            rcnn_cls_pi = self.cls_layers_PI[0](cur_feat_mm)
            # rcnn_reg_pi:(160, 7); 虚拟点云回归预测
            rcnn_reg_pi = self.reg_layers_PI[0](cur_feat_mm)

            ''' 真实点云特征 '''
            # rcnn_cls_p:(160, 1); 点云分类预测
            rcnn_cls_p = self.cls_layers_P[0](cur_feat)
            # rcnn_reg_p:(160, 7); 点云回归预测
            rcnn_reg_p = self.reg_layers_P[0](cur_feat)

            # False
            if self.model_cfg.get('PART', False):
                rcnn_cls = rcnn_cls+part_scores
                rcnn_cls_pi = rcnn_cls_pi+part_scores
                rcnn_cls_p = rcnn_cls_p+part_scores

            ''' generate_predicted_boxes(): '''
            # batch_cls_preds:(bs, 160, 1)
            # batch_box_preds:(bs, 160, 7)
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'],
                #
                rois=batch_dict['rois'],
                # (160,1)
                cls_preds=rcnn_cls,
                # (160,7)
                box_preds=rcnn_reg
            )

            # (bs, 160, 7)
            outs = batch_box_preds.clone()

            # False
            if 'transform_param' in batch_dict:
                outs = self.pred_x_trans(outs, i, batch_dict['transform_param'])

            #
            all_preds.append(outs)
            #
            all_scores.append(batch_cls_preds)

            # True
            if self.training:
                #
                targets_dict_pi = copy.deepcopy(targets_dict)
                #
                targets_dict_p = copy.deepcopy(targets_dict)
                # 混合点云最终的 分类预测
                targets_dict['rcnn_cls'] = rcnn_cls
                # 混合点云最终的 回归预测
                targets_dict['rcnn_reg'] = rcnn_reg
                # 虚拟点云最终的 分类预测
                targets_dict_pi['rcnn_cls'] = rcnn_cls_pi
                # 虚拟点云最终的 回归预测
                targets_dict_pi['rcnn_reg'] = rcnn_reg_pi
                # 点云最终的 分类预测
                targets_dict_p['rcnn_cls'] = rcnn_cls_p
                # 点云最终的 回归预测
                targets_dict_p['rcnn_reg'] = rcnn_reg_p

                # 上述对应的 targets_dict 字典 全部添加到 forward_ret_dict 字典中
                self.forward_ret_dict['targets_dict' + rot_num_id] = targets_dict
                #
                self.forward_ret_dict['targets_dict_pi' + rot_num_id] = targets_dict_pi
                #
                self.forward_ret_dict['targets_dict_p' + rot_num_id] = targets_dict_p

            # 最终的 生成的 batch_box_preds 和 batch_cls_preds
            batch_dict['rois'] = batch_box_preds
            #
            batch_dict['roi_scores'] = batch_cls_preds.squeeze(-1)

        # 这里没有使用 级联框架，所以这里返回的是就是其本身。
        return torch.mean(torch.stack(all_preds), 0), torch.mean(torch.stack(all_scores), 0)

    def forward(self, batch_dict):

        # False
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            self.rot_num = trans_param.shape[1]

        ''' proposal_layer():  '''
        #
        targets_dict = self.proposal_layer(
            batch_dict,
            nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )



        ''' multi_grid_pool_aggregation(): 多网格池化聚合'''
        # boxes:()
        # scores:()
        boxes, scores = self.multi_grid_pool_aggregation(batch_dict, targets_dict)

        # False
        if not self.training:
            batch_dict['batch_box_preds'] = boxes
            batch_dict['batch_cls_preds'] = scores

        return batch_dict
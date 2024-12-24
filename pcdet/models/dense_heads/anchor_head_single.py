import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
import torch
import cv2
import numpy as np

def get_layer(dim,out_dim,init = None):
    init_func = nn.init.kaiming_normal_
    layers = []
    conv = nn.Conv2d(dim, dim,
                      kernel_size=3, padding=1, bias=True)
    nn.init.normal_(conv.weight, mean=0, std=0.001)
    layers.append(conv)
    layers.append(nn.BatchNorm2d(dim))
    layers.append(nn.ReLU())
    conv2 = nn.Conv2d(dim, out_dim,
                     kernel_size=1, bias=True)

    if init is None:
        nn.init.normal_(conv2.weight, mean=0, std=0.001)
        layers.append(conv2)

    else:
        conv2.bias.data.fill_(init)
        layers.append(conv2)

    return nn.Sequential(*layers)


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.grid_size = grid_size  # [1408 1600   40]
        self.range = point_cloud_range

        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        # 分类预测
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )

        # 回归预测
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        # 方向分类预测
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    """
    get_anchor_mask():
        
    
    """
    def get_anchor_mask(self,data_dict,shape):

        stride = np.round(self.voxel_size*8.*10.)

        minx=self.range[0]
        miny=self.range[1]

        points = data_dict["points"]

        mask = torch.zeros(shape[-2],shape[-1])

        mask_large = torch.zeros(shape[-2]//10,shape[-1]//10)

        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride

        in_x = in_x.long().clamp(max=shape[-1]//10-1)
        in_y = in_y.long().clamp(max=shape[-2]//10-1)

        mask_large[in_y,in_x] = 1

        mask_large = mask_large.clone().int().detach().cpu().numpy()

        mask_large_index = np.argwhere( mask_large>0 )

        mask_large_index = mask_large_index*10

        index_list=[]

        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])

        index_list = np.concatenate(index_list,0)

        inds = torch.from_numpy(index_list).cuda().long()

        mask[inds[:,0],inds[:,1]]=1

        return mask.bool()

    def forward(self, data_dict):

        # anchor_mask:(200,176); 获取 anchors 掩码
        anchor_mask = self.get_anchor_mask(data_dict,data_dict['st_features_2d'].shape)

        new_anchors = []

        # 获取 Car 类别的平均anchor的尺寸
        for anchors in self.anchors_root:
            # 根据掩码对anchors进行过滤
            new_anchors.append(anchors[:, anchor_mask, ...])

        # 将 anchors重新赋值; anchors_root 原本是 176*200=35200 个anchors，经过过滤后变成 (eg:15700 ) 个anchors
        self.anchors = new_anchors

        # st_features_2d:(bs,256,200,176)
        st_features_2d = data_dict['st_features_2d']

        # 类别预测：cls_preds:(bs,2,200,176)； Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
        cls_preds = self.conv_cls(st_features_2d)
        # 回归预测：box_preds:(bs,14,200,176)； Conv2d(256, 14, kernel_size=(1, 1), stride=(1, 1))
        box_preds = self.conv_box(st_features_2d)

        # cls_preds:(bs, 15700, 2)；改变维度排序，并且根据掩码进行过滤
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]
        # box_preds:(bs, 15700, 14); 改变维度排序，并且根据掩码进行过滤
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]

        # 将分类预测结果保存到 forward_ret_dict 中
        self.forward_ret_dict['cls_preds'] = cls_preds
        # 将回归预测结果保存到 forward_ret_dict 中
        self.forward_ret_dict['box_preds'] = box_preds

        # True
        if self.conv_dir_cls is not None:
            # 方向类别预测: dir_cls_preds:(bs,4,200,176)； Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
            dir_cls_preds = self.conv_dir_cls(st_features_2d)
            # dir_cls_preds:(bs,15700,4)； 改变维度排序，并且根据掩码进行过滤
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]
            # 将方向类别预测结果保存到 forward_ret_dict 中
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        # True
        if self.training:
            #
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            #
            self.forward_ret_dict.update(targets_dict)

            data_dict['gt_ious'] = targets_dict['gt_ious']

        # True
        if not self.training or self.predict_boxes_when_training:

            ''' generate_predicted_boxes(): '''
            # batch_cls_preds:()
            #
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )

            #
            data_dict['batch_cls_preds'] = batch_cls_preds
            #
            data_dict['batch_box_preds'] = batch_box_preds
            # False
            data_dict['cls_preds_normalized'] = False

        # False
        if self.model_cfg.get('NMS_CONFIG', None) is not None:
            #
            self.proposal_layer(
                data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )

        return data_dict

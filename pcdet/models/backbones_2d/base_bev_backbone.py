import numpy as np
import torch
import torch.nn as nn
from tools.visual_utils.open3d_vis_utils import draw_scenes

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg,  input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features_post = c_in

    ''' 主要操作就是： 
            对上一步骤中HeightCompression得到的 BEV特征 spatial_features 进行卷积操作 '''
    def forward(self, data_dict):

        # spatial_features:(bs, 256, 200, 176)
        spatial_features = data_dict['spatial_features']

        ups = []
        x = spatial_features

        # len(self.blocks) = 2
        for i in range(len(self.blocks)):
            # i:0  ---> (bs, 64, 200, 176)
            # i:1  ---> (bs, 64*2, 100, 88)
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])

            # True
            if len(self.deblocks) > 0:
                # i:0  ---> up:(bs, 64*2, 200, 176)
                # i:1  ---> up:(bs, 64*2, 200, 176)
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        # True
        if len(ups) > 1:
            # (bs,128+128,200,176)
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        # False
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        # st_features_2d:(bs,128+128,200,176)
        data_dict['st_features_2d'] = x

        return data_dict

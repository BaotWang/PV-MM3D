import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tools.visual_utils.open3d_vis_utils import draw_scenes

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):

        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        # PosixPath('../data/kitti/training/velodyne')
        self.root_path = root_path
        # .bin
        self.ext = ext
        # glob模块的主要方法就是glob,该方法返回所有匹配的文件路径列表（list）；得到的bin文件列表是乱序的
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        # 重新排序
        data_file_list.sort()

        # 赋值给 sample_file_list
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        #
        if self.ext == '.bin':
            # numpy.fromfile 函数用于从文件中读取数据，并将其存储为 ndarray 对象。
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)

        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])

        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        # 对数据进行预处理
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    # 创建 logger 对象
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # 创建 demo_dataset 对象
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    #
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    #
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            ''' 这里是调用 open3d_vis_utils.py 中的 draw_scenes() 函数 '''
            V.draw_scenes(
                # (num_points, 4)
                points=data_dict['points'][:, 1:],
                # 预测框
                ref_boxes=pred_dicts[0]['pred_boxes'],
                # 预测分数
                ref_scores=pred_dicts[0]['pred_scores'],
                # 预测类别
                ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')

    # print(get_model_params(model))


# 计算参数量
def get_model_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn

    return pp



if __name__ == '__main__':
    main()

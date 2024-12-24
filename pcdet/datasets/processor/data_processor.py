from functools import partial

import numpy as np
from skimage import transform
from tools.visual_utils.open3d_vis_utils import draw_scenes

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, rot_num, num_point_features):
        self.rot_num = rot_num
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        for rot_num_id in range(self.rot_num):
            if rot_num_id == 0:
                rot_num_id_str = ''
            else:
                rot_num_id_str = str(rot_num_id)
            mask = common_utils.mask_points_by_range(data_dict['points'+rot_num_id_str], self.point_cloud_range)
            data_dict['points'+rot_num_id_str] = data_dict['points'+rot_num_id_str][mask]

            if 'mm' in data_dict:
                mask = common_utils.mask_points_by_range(data_dict['points_mm'+rot_num_id_str], self.point_cloud_range)
                data_dict['points_mm'+rot_num_id_str] = data_dict['points_mm'+rot_num_id_str][mask]

            if data_dict.get('gt_boxes'+rot_num_id_str, None) is not None and config.REMOVE_OUTSIDE_BOXES:
                mask = box_utils.mask_boxes_outside_range_numpy(
                    data_dict['gt_boxes'+rot_num_id_str], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
                )
                data_dict['gt_boxes'+rot_num_id_str] = data_dict['gt_boxes'+rot_num_id_str][mask]

                if rot_num_id==0 and 'gt_tracklets'+rot_num_id_str in data_dict:
                    data_dict['gt_tracklets'] = data_dict['gt_tracklets'][mask]
                    data_dict['num_bbs_in_tracklets'] = data_dict['num_bbs_in_tracklets'][mask]

        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:

            for rot_num_id in range(self.rot_num):
                if rot_num_id == 0:
                    rot_num_id_str = ''
                else:
                    rot_num_id_str = str(rot_num_id)
                points = data_dict['points'+rot_num_id_str]
                shuffle_idx = np.random.permutation(points.shape[0])
                points = points[shuffle_idx]
                data_dict['points'+rot_num_id_str] = points
                if 'mm' in data_dict:
                    points = data_dict['points_mm'+rot_num_id_str]
                    shuffle_idx = np.random.permutation(points.shape[0])
                    points = points[shuffle_idx]
                    data_dict['points_mm'+rot_num_id_str] = points

        return data_dict

    """
    transform_points_to_voxels():
        将点变换为体素
    """
    def transform_points_to_voxels(self, data_dict=None, config=None):

        #
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE

            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        #
        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        # 判断是否为 POINTS_MM_USE_ALONE,如果是的话，就需要将 真实点云虚拟点云分类，因为 虚拟点云不进行体素化
        # self.rot_num = 1;
        for rot_num_id in range(self.rot_num):
            if rot_num_id == 0:
                rot_num_id_str = ''
            else:
                rot_num_id_str = str(rot_num_id)

            # 激光雷达点云与虚拟点云混合在一起
            points = data_dict['points'+rot_num_id_str]

            # True; 是否 激光雷达点云优先
            if config.get('LIDAR_FIRST', False):
                # 根据特征最后一位，获取激光雷达点云
                points_l = points[points[:,-1]==2]
                # 根据特征最后一位，获取激虚拟点云
                points_mm = points[points[:,-1]==1]

                # 判断是否单独使用 虚拟点云
                if data_dict.get('points_mm_use_alone', False):
                    points = points_l
                    data_dict['points' + rot_num_id_str] = points
                    data_dict['points_mm' + rot_num_id_str] = points_mm
                else:
                    # 将 激光雷达点云与虚拟点云重新排序（激光雷达点云、虚拟点云）
                    points = np.concatenate([points_l, points_mm])

            # 将 points 生成 voxel
            voxel_output = self.voxel_generator.generate(points)

            # False
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']

            else:
                # voxels:(40000,5,8);8:(x,y,z,r,R,G,B,point_flag)
                # coordinates:(40000,3); 每个 voxel 的坐标
                # num_points:(40000,); 每个 voxel中的point的个数
                voxels, coordinates, num_points = voxel_output

            # False ; use_lead_xyz = True
            if not data_dict['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

            # voxels:(40000,5,8)
            data_dict['voxels'+rot_num_id_str] = voxels
            # voxel_coords:(40000,3)
            data_dict['voxel_coords'+rot_num_id_str] = coordinates
            # voxel_num_points:(40000,)
            data_dict['voxel_num_points'+rot_num_id_str] = num_points

            # False
            if 'mm' in data_dict:
                points = data_dict['points_mm'+rot_num_id_str]

                voxel_output = self.voxel_generator.generate(points)
                if isinstance(voxel_output, dict):
                    voxels, coordinates, num_points = \
                        voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
                else:
                    voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

                data_dict['voxels_mm'+rot_num_id_str] = voxels
                data_dict['voxel_coords_mm'+rot_num_id_str] = coordinates
                data_dict['voxel_num_points_mm'+rot_num_id_str] = num_points

        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]

            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
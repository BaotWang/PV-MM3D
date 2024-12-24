"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

import scipy
from scipy.spatial import Delaunay

'''
一行代表点的颜色: 白色
二行代表汽车的颜色
三四行代表行人
自行车的颜色
'''

# box 颜色
# box_colormap = [
#     [1, 1, 1],
#     [0, 1, 0],
#     [0, 1, 1],
#     [1, 1, 0],
# ]

box_colormap = [
    [1, 1, 1],
    [255, 0, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):

    # True
    if isinstance(points, torch.Tensor):

        points = points.cpu().numpy()

    # False
    if isinstance(gt_boxes, torch.Tensor):

        gt_boxes = gt_boxes.cpu().numpy()

    # True
    if isinstance(ref_boxes, torch.Tensor):

        ref_boxes = ref_boxes.cpu().numpy()

    # vis = Visualizer with name Open3D
    vis = open3d.visualization.Visualizer()

    # 创建一个 画板窗口
    vis.create_window()

    # 点的大小
    vis.get_render_option().point_size = 1.2

    # 渲染背景颜色：黑色  = [0. 0. 0.]；
    # vis.get_render_option().background_color = np.zeros(3)
    vis.get_render_option().background_color = np.array([217,217,217],dtype=float)

    ''' draw origin '''
    # True
    if draw_origin:
        # 创建坐标系
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # 添加几何图形
        vis.add_geometry(axis_pcd)

    # 创建 PointCloud 对象
    pts = open3d.geometry.PointCloud()

    # 给 PointCloud 的 points属性 赋值
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    #
    vis.add_geometry(pts)

    # True
    if point_colors is None:
        # 给每个点 初始化颜色
        # pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3))) 白色
        pts.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))  # 黑色

        # colors = [[0., 191., 255.] for _ in range(points.shape[0])]  # 深天蓝
        # colors = [[0., 255., 255] for _ in range(points.shape[0])]   # 青色
        # pts.colors = open3d.utility.Vector3dVector(np.asarray(colors))

    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    # False
    if gt_boxes is not None:
        #
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    # True
    if ref_boxes is not None:

        #
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

        ''' 将检测框中的点云染成红色 '''
        vis = draw_point_in_box3d(vis, points, ref_boxes)

    #
    vis.run()
    #
    vis.destroy_window()


"""
translate_boxes_to_open3d_instance():
    gt_boxes


"""


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """

    # 获取 boxes 的 中心点
    center = gt_boxes[0:3]

    # 获取 boxes 的 长宽高
    lwh = gt_boxes[3:6]

    # 获取 boxes 的 轴角度
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])

    # 从轴角度得到旋转矩阵；rot:(3,3) ---> 是一个 旋转矩阵
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)

    # 构建 原始的 box
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    # 根据上述的 box3d 画出边界线
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    ''' import ipdb; ipdb.set_trace(context=20) '''
    #
    lines = np.asarray(line_set.lines)

    #
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    #
    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


"""
draw_box():
    vis:open3D  实例对象


"""


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    # 逐 boxes 处理
    for i in range(gt_boxes.shape[0]):

        ''' translate_boxes_to_open3d_instance(); '''
        #
        #
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])

        # False
        if ref_labels is None:
            line_set.paint_uniform_color(color)

        else:
            # 画上均匀颜色 （可以根据不同的类别画上不同的颜色）
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        #
        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_point_in_box3d(vis, points, boxes3d):
    corner3ds = boxes_to_corners_3d(boxes3d)  # [N,8,3]
    pc_in_boxes_sum = np.zeros((1, 4))
    for i in range(corner3ds.shape[0]):
        flag = in_hull(points[:, 0:3], corner3ds[i])
        pc_in_boxes = points[flag]
        pc_in_boxes_sum = np.vstack((pc_in_boxes_sum, pc_in_boxes))

    points_in_boxes = pc_in_boxes_sum
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points_in_boxes[:, :3])
    vis.add_geometry(pts)

    colors = [[1., 0., 1.] for _ in range(points_in_boxes.shape[0])]
    pts.colors = open3d.utility.Vector3dVector(np.asarray(colors))
    return vis


def in_hull(p, hull):
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)
    return flag


# 可视化框boxes 转corner角（如下图  和pcdet/utils/box_utils.py中boxes_to_corners_3d函数一样
def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

import torch

"""
scatter_point_inds():
    函数的作用是在一个指定形状的张量中，根据给定的索引将特定的点标记为正数
    indices:稀疏 tensors 对应的索引 ;point_inds: 初始化 稀疏 tensors 对应的 indices
"""


def scatter_point_inds(indices, point_inds, shape):
    # ([1, 21, 400, 352]); 函数创建一个与指定形状相同大小的张量ret，并将其所有元素初始化为-1； 此步骤为后续标记操作提供了“画布”
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    # 4;
    ndim = indices.shape[-1]
    # 函数通过展平indices张量的形状，获取其每个维度的索引
    flattened_indices = indices.view(-1, ndim)
    # {list:4}; 按照维度遍历，将 flattened_indices 张量中的每个元素插入到对应的位置
    slices = [flattened_indices[:, i] for i in range(ndim)]
    # 按照索引插入到对应的位置
    ret[slices] = point_inds
    return ret


"""
generate_voxel2pinds():
    sparse_tensor: SparseConvTensor[shape=torch.Size([80787, 64])]

"""


def generate_voxel2pinds(sparse_tensor):
    # cuda
    device = sparse_tensor.indices.device
    # bs
    batch_size = sparse_tensor.batch_size
    # spatial_shape: [21, 400, 352]; 空间形状
    spatial_shape = sparse_tensor.spatial_shape
    # indices(80787, 4); 获取 稀疏tensor对应的索引值; 4:(bs_idx,x,y,z)
    indices = sparse_tensor.indices.long()
    # 初始化一个索引张量 point_indices:(80787,)
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    # output_shape:[bs,21,400,352] = [bs]+[21, 400, 352]
    output_shape = [batch_size] + list(spatial_shape)

    ''' scatter_point_inds(): 用于在指定形状的张量中，根据提供的多维索引将特定点的值设置为特定值（由 point_indices 提供）'''
    # (bs,21,400,352)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor

def generate_voxel2pinds2(batch_size,spatial_shape,indices):
    indices = indices.long()
    device = indices.device
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor

from typing import Set

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch.nn as nn


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

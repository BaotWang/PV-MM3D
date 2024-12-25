
# PV-MM3D: Point-Voxel Parallel Dual-Stream Framework with Dual-Attention Region Adaptive Fusion for Multimodal 3D Object Detection 
This is a official code release of PV-MM3D. 
This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), some codes are from [VirConv](https://github.com/hailanyi/VirConv),[TED](https://github.com/hailanyi/TED), 
[CasA](https://github.com/hailanyi/CasA), [PENet](https://github.com/JUGGHM/PENet_ICRA2021) and [SFD](https://github.com/LittlePey/SFD).

## Detection Framework

The detection frameworks are shown below.

![](/tools/image/framework.png)


## Getting Started
```
conda create -n pvmm3d python=3.9
conda activate pvmm3d
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython prefetch-generator
```
### Dependency

Our released implementation is tested on.
+ Ubuntu 20.04
+ Python 3.9.13 
+ PyTorch 1.8.1
+ Numba 0.53.1
+ Spconv 2.1.22 # pip install spconv-cu111
+ NVIDIA CUDA 11.1 
+ 3x NVIDIA A100 GPUs





### Setup

```
cd PV-MM3D
python setup.py develop
```


## License

This code is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
[VirConv](https://github.com/hailanyi/VirConv)

[TED](https://github.com/hailanyi/TED)

[CasA](https://github.com/hailanyi/CasA)

[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

[PENet](https://github.com/JUGGHM/PENet_ICRA2021)

[SFD](https://github.com/LittlePey/SFD)






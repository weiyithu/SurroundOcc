# SurroundOcc
### [Paper](https://arxiv.org/abs/2303.09551) | [Video](https://cloud.tsinghua.edu.cn/d/97b74c039b8d4fd48830/) | [Data](https://cloud.tsinghua.edu.cn/d/8dcb547238144d08a0bb/)
<br/>

> SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving  
> [Yi Wei*](https://weiyithu.github.io/), [Linqing Zhao*](https://github.com/lqzhao), [Wenzhao Zheng](https://scholar.google.com/citations?user=LdK9scgAAAAJ&hl=en), [Zheng Zhu](http://www.zhengzhu.net/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)  

## News
- [2022/3/17]: Initial code and paper release. 
- [2022/2/27]: Demo release.

## Demo  
Demos are a little bit large; please wait a moment to load them. If you cannot load them or feel them blurry, you can click the hyperlink of each demo for the full-resolution raw video. 

### [Occupancy prediction:](https://cloud.tsinghua.edu.cn/f/f7768f1f110c414297cc/)

<p align='center'>
<img src="./assets/demo1.gif" width="720px">
<img src="./assets/bar.jpg" width="720px">
</p>

### [Generated dense occupancy labels:](https://cloud.tsinghua.edu.cn/f/65d91a4c891f447da731/)
<p align='center'>
<img src="./assets/demo2.gif" width="720px">
</p>


## Introduction
Towards a more comprehensive and consistent scene reconstruction, in this paper, we propose a SurroundOcc method to predict the volumetric occupancy with multi-camera images. We first extract multi-scale features for each image and adopt spatial cross attention to lift them to the 3D volume space. Then we apply 3D convolutions to progressively upsample the volume features and impose supervision on multiple levels. To train the multi-camera 3D scene reconstruction model, we design a pipeline to generate dense occupancy ground truth with sparse LiDAR points. The generation pipeline only needs existed 3D detection and 3D semantic segmentation labels without extra human annotations. Specifically, we fuse multi-frame LiDAR points of dynamic objects and static scenes separately. Then we adopt Poisson Reconstruction to fill the holes and voxelize the mesh to get dense volumetric occupancy.

## Method 

Method Pipeline:

<p align='center'>
<img src="./assets/pipeline.jpg" width="720px">
</p>

Occupancy Ground Truth Generation Pipeline:

<p align='center'>
<img src="./assets/groundtruth_pipeline.jpg" width="800px">
</p>

## Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/data.md)
- [Train and Eval](docs/run.md)

You can download our pretrained model for [3D semantic occupancy prediction](https://cloud.tsinghua.edu.cn/f/7b2887a8fe3f472c8566/?dl=1) and [3D scene reconstruction tasks](https://cloud.tsinghua.edu.cn/f/ca595f31c8bd4ec49cf7/?dl=1). The difference is whether use semantic labels to train the model. The models are trained on 8 RTX 3090s with about 2.5 days.  

## Try your own data
Occupancy Prediction:
You can try our nuScenes [pretrained model](https://cloud.tsinghua.edu.cn/f/7b2887a8fe3f472c8566/?dl=1) on your own data!  Here we give a template [pickle file](https://cloud.tsinghua.edu.cn/f/5c710efd78854c529705/?dl=1). You should place it in ./data and change the corresponding infos. Specifically, you need to change the 'lidar2img', 'intrinsic' and 'data_path' as the extrinsic matrix, intrinsic matrix and path of your multi-camera images. Note that the order of frames should be same to their timestamps. 'occ_path' in this pickle file indicates the save path and you will get raw results (.npy) and point coulds (.ply) for further visualization:
```
./tools/dist_inference.sh ./projects/configs/surroundocc/surroundocc_inference.py ./path/to/ckpts.pth 8
```


## Acknowledgement
Many thanks to these excellent projects:
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [MonoScene](https://github.com/astra-vision/MonoScene)

Related Projects:
- [TPVFormer](https://github.com/wzzheng/TPVFormer)
- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)
- [VoxFormer](https://github.com/NVlabs/VoxFormer)

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{wei2023surroundocc, 
      title={SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving}, 
      author={Yi Wei and Linqing Zhao and Wenzhao Zheng and Zheng Zhu and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2303.09551},
      year={2023}
}
```


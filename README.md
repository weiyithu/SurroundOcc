# SurroundOcc
### [Paper](https://arxiv.org/abs/) | [Video](https://cloud.tsinghua.edu.cn/d/97b74c039b8d4fd48830/) | [Data](https://cloud.tsinghua.edu.cn/d/8dcb547238144d08a0bb/)
<br/>

> SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving  
> [Yi Wei*](https://weiyithu.github.io/), [Linqing Zhao*](https://github.com/lqzhao), [Wenzhao Zheng](https://scholar.google.com/citations?user=LdK9scgAAAAJ&hl=en), [Zheng Zhu](http://www.zhengzhu.net/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)  

## News
- [2022/3/17]: Initial code and paper release. 
- [2022/2/27]: Demo release.

## Introduction
Towards a more comprehensive and consistent scene reconstruction, in this paper, we propose a SurroundOcc method to predict the volumetric occupancy with multi-camera images. We first extract multi-scale features for each image and adopt spatial cross attention to lift them to the 3D volume space. Then we apply 3D convolutions to progressively upsample the volume features and impose supervision on multiple levels. To train the multi-camera 3D scene reconstruction model, we design a pipeline to generate dense occupancy ground truth with sparse LiDAR points. The generation pipeline only needs existed 3D detection and 3D semantic segmentation labels without extra human annotations. Specifically, we fuse multi-frame LiDAR points of dynamic objects and static scenes separately. Then we adopt Poisson Reconstruction to fill the holes and voxelize the mesh to get dense volumetric occupancy.

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

### [Comparison with TPVFormer:](https://cloud.tsinghua.edu.cn/f/01031a906f3249c4aab4/)
<p align='center'>
<img src="./assets/demo3.gif" width="720px">
</p>

### [In the wild demo](https://cloud.tsinghua.edu.cn/f/81f9980ed2854b74b15e/) (trained on nuScenes, tested on Beijing street):
<p align='center'>
<img src="./assets/demo4.gif" width="720px">
</p>


## Method 

Method Pipeline:

<p align='center'>
<img src="./assets/pipeline.jpg" width="720px">
</p>

Occupancy Ground Truth Generation Pipeline:

<p align='center'>
<img src="./assets/groundtruth_pipeline.jpg" width="800px">
</p>

# Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/data.md)
- [Run and Eval](docs/run.md)

You can download our pretrained model for [3D semantic occupancy prediction]() and 3D scene reconstruction tasks. The difference is whether use semantic labels to train the model. The models are trained on 8 RTX 3090s with about 2.5 days.  

## Related Work
Welcome to see another our lab's work [TPVFormer](https://github.com/wzzheng/TPVFormer).

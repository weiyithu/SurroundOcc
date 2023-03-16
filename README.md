# SurroundOcc
### [Paper](https://arxiv.org/abs/) | [Video](https://cloud.tsinghua.edu.cn/d/97b74c039b8d4fd48830/) | [Data](https://cloud.tsinghua.edu.cn/d/8dcb547238144d08a0bb/)
<br/>

> SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving  
> [Yi Wei*](https://weiyithu.github.io/), [Linqing Zhao*](https://github.com/lqzhao), [Wenzhao Zheng](https://scholar.google.com/citations?user=LdK9scgAAAAJ&hl=en), [Zheng Zhu](http://www.zhengzhu.net/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)  

<p align='center'>
<img src="imgs/demo.gif" width='80%'/>
</p>

## Demo  
Demos are a little bit large; please wait a moment to load them. If you cannot load them or feel them blurry, you can click the hyperlink of each demo for the full-resolution raw video. 

### [Occupancy prediction:](https://cloud.tsinghua.edu.cn/f/5dd268c93e36474d8f64/)

<p align='center'>
<img src="./assets/demo1.gif" width="720px">
<img src="./assets/bar.jpg" width="720px">
</p>

### [Generated dense occupancy labels:](https://cloud.tsinghua.edu.cn/f/55d79fa1b5fc4fcd9308/)
<img src="./assets/demo2.gif" width="800px">

### [Comparison with TPVFormer:](https://cloud.tsinghua.edu.cn/f/1a0be6a660224dbfa28a/)
<img src="./assets/demo3.gif" width="800px">

### [In the wild demo](https://cloud.tsinghua.edu.cn/f/021dd2105a3c4251add8/) (trained on nuScenes, tested on Beijing street):
<img src="./assets/demo4.gif" width="800px">

## Introduction
Towards a more comprehensive and consistent scene reconstruction, in this paper, we propose a SurroundOcc method to predict the volumetric occupancy with multi-camera images. We first extract multi-scale features for each image and adopt spatial cross attention to lift them to the 3D volume space. Then we apply 3D convolutions to progressively upsample the volume features and impose supervision on multiple levels. To train the multi-camera 3D scene reconstruction model, we design a pipeline to generate dense occupancy ground truth with sparse LiDAR points. The generation pipeline only needs existed 3D detection and 3D semantic segmentation labels without extra human annotations. Specifically, we fuse multi-frame LiDAR points of dynamic objects and static scenes separately. Then we adopt Poisson Reconstruction to fill the holes and voxelize the mesh to get dense volumetric occupancy.

## Method 

Method Pipeline:

<img src="./assets/pipeline.jpg" width="800px">

Occupancy Ground Truth Generation Pipeline:

<img src="./assets/groundtruth_pipeline.jpg" width="800px">

## Todo
Paper, code and dataset will be released in March.

## Related Work
Welcome to see another our lab's work [TPVFormer](https://github.com/wzzheng/TPVFormer).

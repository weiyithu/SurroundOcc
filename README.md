# SurroundOcc
Multi-camera 3D Occupancy Prediction for Autonomous Driving

- We generate dense occupancy ground truth with multi-frame sparse LiDAR points for training and evaluation.
- We lift image features to the voxel space and progressively upsample them to increase the resolution.


## Demo  
Demos are a little bit large; please wait a moment to load them. If you cannot load them, you can click the hyperlink of each demo for the raw video. 

### [Occupancy prediction:](https://cloud.tsinghua.edu.cn/f/5dd268c93e36474d8f64/)

<img src="./assets/demo1.gif" width="800px">
<img src="./assets/bar.jpg" width="800px">

### [Generated dense occupancy labels:](https://cloud.tsinghua.edu.cn/f/55d79fa1b5fc4fcd9308/)
<img src="./assets/demo2.gif" width="800px">

### [Comparison with other methods:](https://cloud.tsinghua.edu.cn/f/1a0be6a660224dbfa28a/)
<img src="./assets/demo3.gif" width="800px">

### [In the wild demo](https://cloud.tsinghua.edu.cn/f/021dd2105a3c4251add8/) (trained on nuScenes, tested on Beijing street):
<img src="./assets/demo4.gif" width="800px">

## Introduction
Towards a more comprehensive and consistent scene reconstruction, in this paper, we propose a SurroundOcc method to predict the volumetric occupancy with multi-camera images. We first extract multi-scale features for each image and adopt spatial cross attention to lift them to the 3D volume space. Then we apply 3D convolutions to progressively upsample the volume features and impose supervision on multiple levels. To train the multi-camera 3D scene reconstruction model, we design a pipeline to generate dense occupancy ground truth with sparse LiDAR points. Specifically, we fuse multi-frame LiDAR points of dynamic objects and static scenes separately. Then we adopt Poisson Reconstruction to fill the holes and voxelize the mesh to get dense volumetric occupancy.

## Method 
<img src="./assets/pipeline.jpg" width="800px">

## Related Work
Welcome to see another our lab's work [TPVFormer](https://github.com/wzzheng/TPVFormer).

# SurroundOcc
Multi-camera 3D Occupancy Prediction for Autonomous Driving


## Demo
### Occupancy prediction on nuScenes validation set:

<img src="./assets/demo1.gif" width="800px">
<img src="./assets/bar.jpg" width="800px">

### Generated dense occupancy labels:
<img src="./assets/demo2.gif" width="800px">

### Comparison with other method:
<img src="./assets/demo3.gif" width="800px">

### In the wild demo (trained on nuScenes, tested on Beijing street):
<img src="./assets/demo4.gif" width="800px">

## Introduction
Towards a more comprehensive and consistent scene reconstruction, in this paper, we propose a Surround Occupancy method to predict the volumetric occupancy with multi-camera images. We first extract multi-scale features for each image and adopt spatial cross attention to lift them to the 3D volume space. Then we apply 3D convolutions to progressively upsample the volume features and impose supervision on multiple levels. To train the multi-camera 3D scene reconstruction model, we design a pipeline to generate dense occupancy ground truth with sparse LiDAR points. Specifically, we fuse multi-frame LiDAR points of dynamic objects and static scenes separately. Then we adopt Poisson Reconstruction to densify the points and voxelize the mesh to get dense volumetric occupancy.

## Method 
<img src="./assets/pipeline.jpg" width="800px">

## Related Work
Welcome to see our lab's another work [TPVFormer](https://github.com/wzzheng/TPVFormer)

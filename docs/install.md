# Step-by-step installation instructions
Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

**1. Create a conda virtual environment and activate it.**
```shell
conda create -n surroundocc python=3.7 -y
conda activate surroundocc
```

**2. Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.1).**
```shell
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
```

**3. Install gcc>=5 in conda env.**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**4. Install MMCV following the [official instructions](https://github.com/open-mmlab/mmcv).**
```shell
pip install mmcv-full==1.4.0
```

**5. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**6. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**7. Install other dependencies.**
```shell
pip install timm
pip install open3d-python
```

**8. Install Chamfer Distance.**
```shell
cd SurroundOcc/extensions/chamfer_dist
python setup.py install --user
```

**9. Prepare pretrained models.**
```shell
cd SurroundOcc 
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

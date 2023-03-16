**1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Folder structure:**
```
SurroundOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
```


**2. Download the generated [train](https://cloud.tsinghua.edu.cn/f/ebbed36c37b248149192/?dl=1)/[val](https://cloud.tsinghua.edu.cn/f/b3f169f4db034764bb87/?dl=1) pickle files and put them in data.**

**3. Download our generated dense occupancy labels (resolution 200x200x16 with voxel size 0.5m) and put and unzip it in data**
| Subset | Tsinghua Cloud| Size |
| :---: | :---: | :---: |
| train | [link](https://cloud.tsinghua.edu.cn/f/f021006560b54bc78349/?dl=1) | 4.3G |
| val | [link](https://cloud.tsinghua.edu.cn/f/290276f4a4024896b733/?dl=1) | 627M |

**Folder structure:**
```
SurroundOcc
├── data/
│   ├── nuscenes/
│   ├── nuscenes_occ/
│   ├── nuscenes_infos_train.pkl
│   ├── nuscenes_infos_val.pkl

```

***4. (Optional) We also provide the code to generate occupancy on nuScenes, which needs LiDAR point semantic labels [HERE](https://www.nuscenes.org/download). Folder structure:**
```
SurroundOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
|   |   ├── lidarseg
|   |   |   ├── v1.0-test
|   |   |   ├── v1.0-trainval
|   |   |   ├── v1.0-mini
```

You can generate train/val split of nuScenes from 850 sequences. 

```
cd $Home/tools/generate_occupancy_nuscenes
python generate_occupancy_nuscenes.py --config_path ./config.yaml --label_mapping ./nuscenes.yaml --split [train/val] --save_path [your/save/path] 
```

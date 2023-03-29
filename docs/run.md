# Train and Test

Train SurroundOcc with 8 RTX3090 GPUs 
```
./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc.py 8  ./work_dirs/surroundocc
```

Eval SurroundOcc with 8 RTX3090 GPUs
```
./tools/dist_test.sh ./projects/configs/surroundocc/surroundocc.py ./path/to/ckpts.pth 8
```
You can substitute surroundocc.py with surroundocc_nosemantic.py for 3D scene reconstruction task.

Visualize occupancy predictions:

First, you need to generate prediction results. Here we use whole validation set as an example.
```
cp ./data/nuscenes_infos_val.pkl ./data/infos_inference.pkl
./tools/dist_inference.sh ./projects/configs/surroundocc/surroundocc_inference.py ./path/to/ckpts.pth 8
```
You will get prediction results in './visual_dir'. You can directly use meshlab to visualize .ply files or run visual.py to visualize raw .npy files with mayavi:
```
cd ./tools
python visual.py $npy_path$
```

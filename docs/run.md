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
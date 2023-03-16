#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
SAVE_PATH=$3
PORT=${PORT:-28108}
NCCL_DEBUG=INFO

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir ${SAVE_PATH}  --launcher pytorch ${@:4} --deterministic

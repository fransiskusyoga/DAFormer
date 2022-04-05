#!/usr/bin/env bash
set -e
set -x

CONFIG=$1
GPUS=$2
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/pan_train.py $CONFIG --launcher pytorch   ${@:3}   --deterministic

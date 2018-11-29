#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
ITERS=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB="icdar2015ch4"

LOG="experiments/logs/q_rfcn_tloc_with_ref_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# icdar2015ch4  replace \n with \\\n or replace \\\n with \n
time ./tools/train_net.py   \
  --gpu ${GPU_ID} \
  --solver models/q_rfcn_tloc_with_ref/solver_q_rfcn_tloc_with_ref.prototxt \
  --weights data/imagenet_models/ResNet-50-model.caffemodel \
  --iters ${ITERS} \
  --imdb ${TRAIN_IMDB} \
  --cfg experiments/cfgs/q_rfcn_tloc_with_ref.yml \
  --train_label_list data/icdar2015ch4/ch4_training_localization_transcription_gt.txt \
  --train_image_list data/icdar2015ch4/ch4_training_images.txt \
  ${EXTRA_ARGS}

 

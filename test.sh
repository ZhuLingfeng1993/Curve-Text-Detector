#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

it=$1
#NET="output/ctd_tloc.caffemodel"
NET="output/rfcn_ctd/ctw1500/ctd_tloc_iter_26000.caffemodel"

LOG="experiments/logs/test_rfcn_ctd_ResNet50.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

./tools/test_net.py --gpu 3 \
  --def models/ctd/test_ctd_tloc.prototxt \
  --net ${NET} \
  --imdb ctw1500_test \
  --cfg experiments/cfgs/rfcn_ctd.yml \
  --test_label data/ctw1500/test/test_label_curve.txt \
  --test_image data/ctw1500/test/test.txt \
  # --vis 

#!/usr/bin/env bash
set -x
set -e

NET="ResNet-50"
LOG="experiments/logs/test_q_rfcn_tloc_with_ref_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

TEST_IMDB="icdar2015ch4_test"

# icdar2015ch4
./tools/test_net.py --gpu 3 \
  --def models/q_rfcn_tloc_with_ref/test_q_rfcn_tloc_with_ref.prototxt \
  --net output/cpu_fast_version_q_rfcn_tloc_with_ref/icdar2015ch4/q_rfcn_tloc_with_ref_iter_3.caffemodel \
  --imdb icdar2015ch4_test \
  --cfg experiments/cfgs/cpu_fast_version_q_rfcn_tloc_with_ref.yml \
  --test_label data/icdar2015ch4/Challenge4_Test_Task1_GT.txt \
  --test_image data/icdar2015ch4/ch4_test_images.txt \
  --vis

#./tools/test_net.py --gpu 3
#  --def models/q_rfcn_tloc_with_ref/test_q_rfcn_tloc_with_ref.prototxt
#  --net output/cpu_fast_version_q_rfcn_tloc_with_ref/icdar2015ch4/q_rfcn_tloc_with_ref_iter_3.caffemodel
#  --imdb icdar2015ch4_test
#  --cfg experiments/cfgs/cpu_fast_version_q_rfcn_tloc_with_ref.yml
#  --test_label data/icdar2015ch4/Challenge4_Test_Task1_GT.txt
#  --test_image data/icdar2015ch4/ch4_test_images.txt
#  --vis
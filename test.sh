#!/usr/bin/env bash
set -x
set -e

it=$1

TEST_IMDB="icdar2015ch4_test"

# icdar2015ch4
./tools/test_net.py --gpu 3 \
  --def models/q_rfcn_tloc/test_q_rfcn_tloc.prototxt \
  --net output/q_rfcn_tloc/icdar2015ch4/q_rfcn_tloc_iter_3.caffemodel \
  --imdb icdar2015ch4_test \
  --cfg experiments/cfgs/q_rfcn_tloc.yml \
  --test_label data/icdar2015ch4/Challenge4_Test_Task1_GT.txt \
  --test_image data/icdar2015ch4/ch4_test_images.txt \
  #--vis


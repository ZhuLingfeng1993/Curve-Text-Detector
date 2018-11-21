#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
# from fast_rcnn.test import test_net
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=400, type=int)
    parser.add_argument('--rpn_file', dest='rpn_file',
                        default=None, type=str)

    parser.add_argument('--test_label_list', dest='test_label_list',
                        help='place to save', default=None, type=str)
    parser.add_argument('--test_image_list', dest='test_image_list',
                        help='place to save', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # ### run parameters

    # ctd
    # --gpu 0 --def models/ctd/test_ctd_tloc.prototxt --net output/ctd_tloc.caffemodel --imdb ctw1500_test --cfg experiments/cfgs/rfcn_ctd.yml --test_label_list data/ctw1500/test/test_label_curve.txt --test_image_list data/ctw1500/test/test.txt
    #  --vis

    # icdar2015ch4
    '''
    --gpu
    0
    --def
    models/ctd/test_ctd_tloc.prototxt
    --net
    output/rfcn_ctd/icdar2015ch4/ctd_tloc_iter_3.caffemodel
    --imdb
    icdar2015ch4_test
    --cfg
    experiments/cfgs/rfcn_ctd.yml
    --test_label
    data/icdar2015ch4/Challenge4_Test_Task1_GT.txt
    --test_image
    data/icdar2015ch4/ch4_test_images.txt
'''

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    dataset = {}
    dataset['name'] = args.imdb_name
    if args.test_label_list is None:
        assert(0), 'Can not find test label list.'
    if args.test_image_list is None:
        assert(0), 'Can not find test image list.'
    dataset['label_list_file'] = args.test_label_list
    dataset['image_list_file'] = args.test_image_list

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    # set up caffe
    if cfg.USE_GPU_IN_CAFFE == True:        
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    # imdb = get_imdb(args.imdb_name)
    imdb = get_imdb(dataset)
    imdb.competition_mode(args.comp_mode)

    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
        if cfg.TEST.PROPOSAL_METHOD == 'rpn':
            imdb.config['rpn_file'] = args.rpn_file

    test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis)

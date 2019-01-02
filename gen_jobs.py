"""
Setting config and generating files for training and
testing network in one python script.

files to be generated:
    solver.prototxt
    cfg.yml
    train.sh
    test.sh
"""
from __future__ import print_function
import os
import sys
import shutil
import stat
import subprocess
from caffe.proto import caffe_pb2
from easydict import EasyDict

HOMEDIR = os.path.expanduser("~")

# ####### Switch definition #########
switches = EasyDict()
# override existing directory


switches.override = False
# check net
switches.check_net = True

switches.use_gpu = True

switches.use_pretrain_model = True

switches.run_soon = True
# On my own computer, only use cpu to check net
if HOMEDIR == "/home/zhulingfeng":
    switches.check_net = True
    switches.use_gpu = False
# Not caring of override in check net
if switches.check_net:
    switches.override = True

# ##### extra param of solving #####
params = EasyDict()
params.gpu_id = 0
params.max_iter = 70000
params.snapshot = 10000
if switches.check_net:
    params.max_iter = 1

# ############ Name definition ############
names = EasyDict()
# job name
#
# to avoid duplicated model definition
names.job_w_o_flag = "q_rfcn"
# actual job name
names.job = names.job_w_o_flag
if switches.check_net:
    names.job = "check_net_" + names.job

names.basenet = "ResNet50"  # "VGGNet"#"MobileNetMove4"#
# dataset name
names.dataset = "quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset"  # "icdar2015ch4"   #
# train dataset name
names.train_dataset = names.dataset + "_train"  # "VOC2007"#
#  test dataset name
names.test_dataset = names.dataset + "_test"  # "VOC2007"#
# The name of the model. Modify it if you want.
names.model = "{}_{}_{}".format(names.basenet, names.job, names.dataset)

# ############ directories ############
dirs = EasyDict()
# Directory which stores the model .prototxt file.
dirs.model_def = "models/{}/{}/{}".format(names.dataset, names.basenet, names.job_w_o_flag)
#  Directory which stores the cfg and script file.
dirs.job = "jobs/{}/{}/{}".format(names.dataset, names.basenet, names.job)

dirs.log = "{}/logs".format(dirs.job)

# refer to fast_rcnn.config.get_output_dir
dirs.snapshot = os.path.join('output', names.job, names.train_dataset)

# ############ files: model, solver, config ############
files = EasyDict()
# model definition files.
files.train_net = "{}/train_{}.prototxt".format(dirs.model_def, names.job)
files.test_net = "{}/test_{}.prototxt".format(dirs.model_def, names.job_w_o_flag)
# solver file
files.solver = "{}/solver_{}.prototxt".format(dirs.model_def, names.job)
# config file
files.cfg = "{}/{}.yml".format(dirs.job, names.job)
# train scirpt
files.train_script = "{}/train_{}.sh".format(dirs.job, names.job)
# test scirpt
files.test_script = "{}/test_{}.sh".format(dirs.job, names.job)

# log file
files.train_log = "{}/train_{}.log".format(dirs.log, names.model)
files.test_log = "{}/test_{}.log".format(dirs.log, names.model)

# snapshot prefix.
# snapshot_prefix = "{}/{}".format(dirs.snapshot, names.model)
snapshot_prefix = names.model

# The pretrained model.
pretrain_model = None
if switches.use_pretrain_model:
    if names.basenet == "MobileNet":
        pretrain_model = "models/MobileNet/mobilenet_iter_73000.caffemodel"
        # pretrain_model = "models/MobileNet/coco/SSD_lr0_0005_batch80_300x300/snapshot/MobileNet_SSD_lr0_0005_batch80_300x300_coco_iter_260000.caffemodel"
    if names.basenet == "VGGNet":
        # The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
        pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
    if names.basenet == "ResNet50":
        pretrain_model = "data/imagenet_models/ResNet-50-model.caffemodel"


# ############# Check directory and file ###############


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_isfile(file_path):
    if not os.path.isfile(file_path):
        print("File not exits: {}".format(file_path))
        exit(1)


def check_and_choose(path):
    if os.path.exists(path) and (not switches.override):
        choice = raw_input("Directory \"{}\" already exist\n, "
                           "files under it maybe overrided, do you"
                           " want to continue?[y/n]".format(path))
        if choice != 'y':
            print("File generation stopped!")
            sys.exit(0)
    elif not os.path.exists(path):
        os.makedirs(path)


check_and_choose(dirs.model_def)
check_and_choose(dirs.job)
check_and_choose(dirs.log)

check_isfile(files.train_net)
check_isfile(files.test_net)
if pretrain_model is not None:
    check_isfile(pretrain_model)


# ############# File Generation ###############

def train_script_gen():
    print("Generating file: {} ...".format(files.train_script))
    with open(files.train_script, 'w') as f:
        # train part
        f.write("#!/bin/bash\n"
                "set -x\n"
                "set -e\n"
                "\n"
                "export PYTHONUNBUFFERED=\"True\"\n"
                "\n")
        f.write("LOG=\"{}.`date +'%Y-%m-%d_%H-%M-%S'`\"\n".format(files.train_log))
        f.write("exec &> >(tee -a \"$LOG\")\n"
                "echo Logging output to \"$LOG\"\n"
                "\n")
        f.write("time ./tools/train_net.py \\\n")
        f.write("  --gpu {} \\\n".format(params.gpu_id))
        f.write("  --solver {} \\\n".format(files.solver))
        f.write("  --weights {} \\\n".format(pretrain_model))
        f.write("  --iters {} \\\n".format(params.max_iter))
        f.write("  --imdb {} \\\n".format(names.train_dataset))
        f.write("  --cfg {} \\\n".format(files.cfg))
        if names.dataset == 'icdar2015ch4':
            f.write("  --train_label_list data/icdar2015ch4/ch4_training_localization_transcription_gt.txt \\\n"
                "  --train_image_list data/icdar2015ch4/ch4_training_images.txt \n\n")
        elif names.dataset == 'quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset':
            f.write("  --train_label_list data/quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset/VOC2007/ImageSets/Main/train.txt \\\n"
                "  --train_image_list data/quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset/VOC2007/ImageSets/Main/train.txt \n\n")
        f.write("\n")
        # test part
        f.write("set +x\n"
                "NET_FINAL=`grep -B 1 \"done solving\" ${LOG} | "
                "grep \"Wrote snapshot\" | awk '{print $4}'`\n"
                "set -x\n"
                "\n")
        f.write("time ./tools/test_net.py \\\n")
        f.write("  --net ${NET_FINAL} \\\n")
        f.write("  --gpu {} \\\n".format(params.gpu_id))
        f.write("  --imdb {} \\\n".format(names.test_dataset))
        f.write("  --def {} \\\n".format(files.test_net))
        f.write("  --cfg {} \\\n".format(files.cfg))

        if names.dataset == 'icdar2015ch4':
            f.write("  --test_label data/icdar2015ch4/Challenge4_Test_Task1_GT.txt \\\n"
                    "  --test_image data/icdar2015ch4/ch4_test_images.txt \\\n")
        elif names.dataset == 'quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset':
            f.write("  --test_label data/quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset/VOC2007/ImageSets/Main/test.txt \\\n"
                "  --test_image data/quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset/VOC2007/ImageSets/Main/test.txt \n\n")


def test_script_gen():
    print("Generating file: {} ...".format(files.test_script))
    with open(files.test_script, 'w') as f:
        f.write("#!/usr/bin/env bash\n"
                "set -x\n"
                "set -e\n"
                "\n")
        f.write("LOG=\"{}.`date +'%Y-%m-%d_%H-%M-%S'`\"\n".format(files.test_log))
        f.write("exec &> >(tee -a \"$LOG\")\n"
                "echo Logging output to \"$LOG\"\n"
                "\n")
        f.write("time ./tools/test_net.py \\\n")
        f.write("  --net {}/{}_iter_{}.caffemodel \\\n".format(dirs.snapshot, snapshot_prefix, params.max_iter))
        f.write("  --gpu {} \\\n".format(params.gpu_id))
        f.write("  --imdb {} \\\n".format(names.test_dataset))
        f.write("  --def {} \\\n".format(files.test_net))
        f.write("  --cfg {} \\\n".format(files.cfg))
        if names.dataset == 'icdar2015ch4':
            f.write("  --test_label data/icdar2015ch4/Challenge4_Test_Task1_GT.txt \\\n"
                    "  --test_image data/icdar2015ch4/ch4_test_images.txt \\\n")
        elif names.dataset == 'quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset':
            f.write("  --test_label data/quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset/VOC2007/ImageSets/Main/test.txt \\\n"
                "  --test_image data/quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset/VOC2007/ImageSets/Main/test.txt \\\n")
        f.write("  #--vis")


def solver_param_def():
    display = 1000 if not switches.check_net else 1
    solver_param = {
        'train_net': "{}".format(files.train_net),
        'base_lr': 0.001,
        'lr_policy': "step",
        'gamma': 0.1,
        'stepsize': 40000,
        'display': display,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        # We disable standard caffe solver snapshotting and implement our own snapshot
        # function
        'snapshot': 0,
        # We still use the snapshot prefix, though
        'snapshot_prefix': snapshot_prefix,
        # 'clip_gradients': 20,
        'iter_size': 1,
        # 'debug_info': true,
    }
    return solver_param


def solver_gen():
    print("Generating file: {} ...".format(files.solver))
    solver_param = solver_param_def()
    solver = caffe_pb2.SolverParameter(**solver_param)
    with open(files.solver, 'w') as f:
        # need lib: from __future__ import print_function
        print(solver, file=f)
    shutil.copy(files.solver, dirs.job)


def cfg_gen():
    print("Generating file: {} ...".format(files.cfg))
    # # use yaml module to generate yaml file,
    # # especially when with no existing template file or with many variables
    # import yaml
    # configs = EasyDict()
    # configs.EXP_DIR = names.job
    # configs.NUM_IMAGES = -1 if not switches.check_net else 2
    # with open(files.cfg, 'w') as f:
    #     yaml.dump(configs, f)

    with open(files.cfg, 'w') as f:
        f.write("EXP_DIR: {}\n".format(names.job))
        f.write("NUM_IMAGES: {}\n".format(
            -1 if not switches.check_net else 2))
        f.write("USE_GPU_NMS: {}\n".format(switches.use_gpu))
        f.write("USE_GPU_IN_CAFFE: {}\n".format(switches.use_gpu))
        f.write("VIS_DATASET: False\n")
        f.write("TRAIN:\n"
                "  HAS_RPN: True\n"
                "  IMS_PER_BATCH: 1\n"
                "  BBOX_NORMALIZE_TARGETS_PRECOMPUTED: True\n"
                "  RPN_POSITIVE_OVERLAP: 0.7\n"
                "  RPN_BATCHSIZE: 256 #\n"
                "  PROPOSAL_METHOD: gt\n"
                "  BG_THRESH_LO: 0.0\n"
                "  BATCH_SIZE: -1\n"
                "  AGNOSTIC: True\n")
        f.write("  SNAPSHOT_ITERS: {} \n".format(params.snapshot))
        f.write("  RPN_PRE_NMS_TOP_N: 6000 \n"
                "  RPN_POST_NMS_TOP_N: 300 \n"
                "  USE_FLIPPED: False\n"
                "TEST:\n"
                "  HAS_RPN: True\n"
                "  AGNOSTIC: True\n"
                "  NMS: 0.3\n"
                "  PNMS: 0.1\n"
                "  USE_PNMS: False\n"
                "  #RPN_PRE_NMS_TOP_N: 120\n"
                "  #RPN_POST_NMS_TOP_N: 30")


solver_gen()
cfg_gen()
train_script_gen()
test_script_gen()

# Copy the python script to dirs.job.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, dirs.job)

# Run the train job.
os.chmod(files.train_script, stat.S_IRWXU)
os.chmod(files.test_script, stat.S_IRWXU)
if switches.run_soon:
    print('Run the train job...\n')
    subprocess.call(files.train_script, shell=True)

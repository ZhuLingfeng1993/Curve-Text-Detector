# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.ctw1500 import ctw1500
from datasets.icdar2015ch4 import icdar2015ch4
from datasets.labelme_qua_data import labelme_qua_data

# # Set up ctw1500_<split>
# for split in ['train', 'test']:
#     name = 'ctw1500_{}'.format(split)
#     __sets[name] = (lambda split=split: ctw1500(split))
#
# # Set up icdar2015ch4_<split>
# for split in ['train', 'test']:
#     name = 'icdar2015ch4_{}'.format(split)
#     __sets[name] = (lambda split=split: icdar2015ch4(split))
#
#
# def get_imdb(name):
#     """Get an imdb (image database) by name."""
#     if not __sets.has_key(name):
#         raise KeyError('Unknown dataset: {}'.format(name))
#     return __sets[name]()


def get_imdb(dataset):
    # imdb = icdar2015ch4(dataset)
    imdb = labelme_qua_data(dataset)
    return imdb

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

# coding=utf-8
import os

import cv2
from datasets.imdb import imdb_text
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import re
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval, voc_eval_polygon
from fast_rcnn.config import cfg
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry import Point
import codecs
import yaml


def vis_data(boxes, polygons, image_path, rel_polygon=True, show_order=False):
    """
    Visualize data by showing one image and label.
    :param show_order:
    :param boxes:
    :param polygons:
    :param image_path:
    :param rel_polygon: if the coordinates of polygon is relative to
                        left top point of bounding box or not
    :return:
    """
    assert (boxes.shape[0] == polygons.shape[0]), \
        'boxes and polygons should have equal row number.'
    import cv2
    # Read image
    img = cv2.imread(image_path)
    #
    if rel_polygon:
        # change polygons from relative coordinates into absolute
        # ones to left top of boxes
        lt_point_tile = np.tile(boxes[:, 0:2], (1, 4))
        # (N,8) = (N,8) - (N,2)
        polygons = polygons + lt_point_tile
    # plot polygons
    n_coors = polygons.shape[1]
    assert (n_coors % 2 == 0), 'n_coors % 2 should be 0 '
    for polygon in polygons:
        for j in xrange(0, n_coors, 2):
            p1 = (int(polygon[j % n_coors]), int(polygon[(j + 1) % n_coors]))
            p2 = (int(polygon[(j + 2) % n_coors]), int(polygon[(j + 3) % n_coors]))
            cv2.line(img, p1, p2, (0, 0, 255), 2)
            if show_order:
                cv2.putText(img, str(j / 2 + 1), p1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=2)
    imk = cv2.resize(img, (1280, 720))  # visualization
    cv2.imshow('data visulization.', imk)
    cv2.waitKey()


class labelme_qua_data(imdb_text):
    def __init__(self, dataset):
        imdb_text.__init__(self, dataset['name'])
        self._label_list_file = dataset['label_list_file']
        self._image_list_file = dataset['image_list_file']
        self._classes = ('__background__', 'car')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._salt = str(uuid.uuid4())  # 通用唯一识别码（Universally Unique Identifier）
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._label_list_file), \
            '_label_list_file path does not exist: {}'.format(self._label_list_file)
        assert os.path.exists(self._image_list_file), \
            '_image_list_file does not exist: {}'.format(self._image_list_file)

        self._labels = self._load_label_image()
        self._image_index = [x for x in range(len(self._labels))]

        self._roidb = self._load_roidb(self._labels)

    def check_polygon_order(self, points):
        points = np.array(points)
        coors_x = points[:, 0]
        coors_y = points[:, 1]
        coors_x_sorted_ind = np.argsort(coors_x)
        coors_y_sorted_ind = np.argsort(coors_y)
        left_set = set(coors_x_sorted_ind[:2])
        right_set = set(coors_x_sorted_ind[2:])
        top_set = set(coors_y_sorted_ind[:2])
        bottom_set = set(coors_y_sorted_ind[2:])
        print(coors_x)
        if {0, 3} == left_set and {1, 2} == right_set and \
                {2, 3} == top_set and {0, 1} == bottom_set:
            return True
        else:
            return False

    def _load_label_image(self):
        """
        Load label and image info into labels from label files and image path files
        labels:list of size N as the number of images
        each element of labels is a dictionary with keys:
            'name'
            'imagePath'
            'gt_boxes': circumscribed rectangle of polygon box ,size = (M,4), M is number of boxes
            'gt_info': 4 points xy coordinates of each quadrilateral box, size = (M,8)
        """
        labels = []
        with open(self._label_list_file, 'r') as fs:
            file_list = fs.readlines()
            assert (cfg.NUM_IMAGES <= len(file_list)), \
                'NUM_IMAGES should be no larger than image lists number'
            # Number of images to use
            if cfg.NUM_IMAGES != -1:
                file_list = file_list[:cfg.NUM_IMAGES]
            for ix, file_index in enumerate(file_list):
                label = {}
                # each line in self._label_list_file is a relative path of data set name,
                # so join the DATA_DIR, so does the image path
                label_file = os.path.join(cfg.DATA_DIR, "quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset",
                                          'VOC2007', 'Annotations', file_index.strip() + '.json')
                label['name'] = label_file
                label['imagePath'] = os.path.join(cfg.DATA_DIR, "quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-car-dataset", 'VOC2007',
                                                  'JPEGImages', file_index.strip() + '.jpg')
                img = Image.open(label['imagePath'])

                with open(label_file, 'r') as f:
                    json_data = yaml.load(f)
                    shapes = json_data['shapes']
                    num_shapes = len(shapes)
                    # quadrilateral points: (x1, y1, x2, y2, x3, y3, x4, y4)
                    gt_info = np.zeros((num_shapes, 8), np.float32)  # syn
                    # circumscribed rectangle box: (xmin, ymin, xmax, ymax)
                    boxes = np.zeros((num_shapes, 4), np.float32)
                    for shape_idx, shape in enumerate(shapes):
                        assert shape['label'] == 'car', 'shape label should be car.'
                        points = shape['points']
                        assert len(points) == 4, 'len(points) = {}, should be 4.'.format(len(points))
                        coordinates = []
                        for point in points:
                            for coor in point:
                                coordinates.append(coor)
                        # if not self.check_polygon_order(points):
                        #     print "check_polygon_order failed!"
                        #     print 'In annotation file: {} , label: {},  points: {}'. \
                        #         format(os.path.splitext(label_file)[0], shape['label'], points)
                        # assert self.check_polygon_order(points), \
                        #     'check_polygon_order failed: {}'.format(points)
                        gt_info[shape_idx, :] = [float(coordinates[i]) for i in range(0, 8)]

                        tuple_points = [(gt_info[shape_idx, ind_p:ind_p + 2])
                                        for ind_p in [0, 2, 4, 6]]
                        quadrilateral = Polygon(tuple_points)
                        assert quadrilateral.is_valid, \
                            ('Not a valid quadrilateral: {}'.format(points))
                        # get polygon bounding box
                        box = quadrilateral.bounds
                        self.check_box(box, img.size)
                        boxes[shape_idx, :] = box
                # change gt_info from absolute coordinates into relative
                # ones to left top of boxes
                lt_point_tile = np.tile(boxes[:, 0:2], (1, 4))
                # (N,8) = (N,8) - (N,2)
                gt_info_rel = gt_info - lt_point_tile

                label['gt_boxes'] = boxes
                label['gt_info'] = gt_info_rel  # syn

                # vis image and label
                if cfg.VIS_DATASET:
                    vis_data(boxes, gt_info_rel, label['imagePath'], rel_polygon=True, show_order=True)

                labels.append(label)
        print "\nload images number = {}\n".format(len(labels))
        return labels

    ################## the images' path and names
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # image_path = os.path.join(self._data_path,'JPEGImages' , 'img_'+index + self._image_ext)
        image_path = os.path.join(self._labels[index]['imagePath'])
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    ################# the annotation's path and name
    def _load_icdar2015ch4_annotation(self, labels, index):
        """
        Load image and bounding boxes info a dictionary from item of index of labels i.
        This dictionary will be an element of gt_roidb
        """
        num_objs = len(labels[index]['gt_boxes'])

        gt_boxes = np.zeros((num_objs, 4), dtype=np.uint8)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        gt_info = np.zeros((num_objs, 8), dtype=np.int32)

        gt_boxes = labels[index]['gt_boxes']
        gt_info = labels[index]['gt_info']  # syn
        # 
        gt_name = labels[index]['imagePath']
        cls = self._class_to_ind['car']

        gt_classes[:] = cls

        return {'boxes': gt_boxes,  # circumscribed rectangle of polygon boxes ,
                # size = (M,4), M is number of boxes
                'gt_classes': gt_classes,  #
                'gt_info': gt_info,  # 4 points xy coordinates of
                # each polygon boxes, size = (M,8)
                'flipped': False,
                'imagePath': gt_name}

    def _load_roidb(self, labels):
        print 'Loading {} gt roidb...'.format(self.name)
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            use_update = 'u'
            # if not force to update cache file, use interactive mode to deal with it
            if not cfg.FORCE_UPDATE_CACHE:
                print 'Cache file already exists: {}'.format(cache_file)
                use_update = raw_input('If you have not modify anything about train data set'
                                       ' or gt roidb, input \'u\' to update it or '
                                       'press any other key to use cache.\n'
                                       'Your input is: ')
            # interactively chosen not update, then use c
            # interactively not update, use cache
            if use_update != 'u':
                with open(cache_file, 'rb') as fid:
                    roidb = cPickle.load(fid)
                print '{} gt roidb loaded from {}'.format(self.name, cache_file)
                return roidb
        # load new roidb or update roidb
        num = len(labels)
        print 'load sample number = ', num
        gt_roidb = [self._load_icdar2015ch4_annotation(labels, i) for i in range(num)]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    ############################# the path of results
    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + 'text' + '_{:s}.txt'
        path = os.path.join(
            self.output_dir,
            filename)
        return path

    #############################the detection result of test will be writen in results folder in txt
    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            print(filename)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        # a=input('check here')
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _qua_write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            print(filename)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        # (ind  score  x1  y1  x2  y2  x3  y3  x4  y4)
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, 4],
                                       dets[k, 0] + 1 + dets[k, 5], dets[k, 1] + 1 + dets[k, 6],
                                       dets[k, 0] + 1 + dets[k, 7], dets[k, 1] + 1 + dets[k, 8],
                                       dets[k, 0] + 1 + dets[k, 9], dets[k, 1] + 1 + dets[k, 10],
                                       dets[k, 0] + 1 + dets[k, 11], dets[k, 1] + 1 + dets[k, 12]
                                       ))

    def _curve_write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            print(filename)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        info_bbox = dets[k, 5:33]  # indexing
                        pts = [info_bbox[i] for i in xrange(28)]
                        f.write(
                            '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, 4],
                                       dets[k, 0] + pts[0], dets[k, 1] + pts[1],
                                       dets[k, 0] + pts[2], dets[k, 1] + pts[3],
                                       dets[k, 0] + pts[4], dets[k, 1] + pts[5],
                                       dets[k, 0] + pts[6], dets[k, 1] + pts[7],
                                       dets[k, 0] + pts[8], dets[k, 1] + pts[9],
                                       dets[k, 0] + pts[10], dets[k, 1] + pts[11],
                                       dets[k, 0] + pts[12], dets[k, 1] + pts[13],
                                       dets[k, 0] + pts[14], dets[k, 1] + pts[15],
                                       dets[k, 0] + pts[16], dets[k, 1] + pts[17],
                                       dets[k, 0] + pts[18], dets[k, 1] + pts[19],
                                       dets[k, 0] + pts[20], dets[k, 1] + pts[21],
                                       dets[k, 0] + pts[22], dets[k, 1] + pts[23],
                                       dets[k, 0] + pts[24], dets[k, 1] + pts[25],
                                       dets[k, 0] + pts[26], dets[k, 1] + pts[27]
                                       ))

                    #########################call voc_eval to evaluate the rec and prec ,mAP

    def _do_python_polygon_eval(self, output_dir='output'):
        cachedir = os.path.join(self.output_dir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            # rec, prec, ap = voc_eval(
            # filename, self._label_list_file, self._image_list_file, cls, cachedir, ovthresh=0.5,
            # use_07_metric=use_07_metric)
            rec, prec, ap = voc_eval_polygon(
                filename, self._label_list_file, self._image_list_file,
                cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            F = 2.0 / (1 / rec[-1] + 1 / prec[-1])
            # print F
            if not os.path.isdir('results'):
                os.mkdir('results')
            f = open('results/test_result.txt', 'a')
            f.writelines('rec:%.3f prec:%.3f F-measure:%.3f \n\n' % (rec[-1], prec[-1], F))
            f.close()
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        # for ap in aps:
        # print('{:.3f}'.format(ap))
        print 'rec:%%%%%%%%%%%%'
        print rec[-1]
        print 'prec:###########'
        print prec[-1]
        print 'F-measure'
        print F
        # print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')

    def evaluate_polygon_detections(self, all_boxes, output_dir):
        self.output_dir = output_dir
        # self._curve_write_voc_results_file(all_boxes)
        self._qua_write_voc_results_file(all_boxes)
        self._do_python_polygon_eval(output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
        else:
            self.config['use_salt'] = True

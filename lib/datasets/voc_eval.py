# coding=utf-8
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

# import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
from fast_rcnn.config import cfg

from shapely.geometry import *

import  yaml

DEBUG = True

def parse_rec_txt(filename):
    with open(filename.strip(), 'r') as f:
        gts = f.readlines()
        objects = []
        for obj in gts:
            cors = obj.strip().split(',')
            # obj_struct['difficult'] =
            obj_struct = {}
            # class name
            obj_struct['name'] = 'text'
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(cors[0]),
                                  int(cors[1]),
                                  int(cors[2]),
                                  int(cors[3])]
            objects.append(obj_struct)
            # assert(0), obj_struct['bbox']
    return objects


def curve_parse_rec_txt(filename):
    """ Parse a ctw1500 txt file """
    with open(filename.strip(), 'r') as f:
        gts = f.readlines()
        objects = []
        for obj in gts:
            cors = obj.strip().split(',')
            # obj_struct['difficult'] =
            obj_struct = {}
            # class name
            obj_struct['name'] = 'text'
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(cors[i]) for i in xrange(32)]
            objects.append(obj_struct)
            # assert(0), obj_struct['bbox']
    return objects

def qua_parse_rec_txt(filename):
    """ Parse a quadrilateral json file """
    with open(filename.strip(), 'r') as f:
        json_data = yaml.load(f)
        shapes = json_data['shapes']
        num_shapes = len(shapes)
        objects = []
        for obj in shapes:
            points = obj['points']
            assert len(points) == 4, 'len(points) = {}, should be 4.'.format(len(points))
            coordinates = []
            for point in points:
                for coor in point:
                    coordinates.append(coor)
            gt_info = [float(coordinates[i]) for i in range(0, 8)]
            tuple_points = [(gt_info[ind_p:ind_p + 2]) for ind_p in [0, 2, 4, 6]]
            quadrilateral = Polygon(tuple_points)
            assert quadrilateral.is_valid, ('Not a valid quadrilateral: {}'.format(obj))
            bbox = quadrilateral.bounds
            # change gt_info from absolute coordinates into relative ones to left top of box
            lt_point_tile = np.tile(bbox[0:2], (1, 4))
            # (N,8) = (N,8) - (N,2)
            gt_info_rel = gt_info - lt_point_tile
            cors = np.zeros(12)
            cors[:4] = bbox
            cors[4:] = gt_info_rel
            # obj_struct['difficult'] =
            obj_struct = {}
            # class name
            obj_struct['name'] = 'car'
            obj_struct['difficult'] = 0
            # (x_min y_min x_man ymax  x1  y1  x2  y2  x3  y3  x4  y4)
            obj_struct['bbox'] = [int(cors[i]) for i in xrange(12)]
            objects.append(obj_struct)
            # assert(0), obj_struct['bbox']
    return objects

# def qua_parse_rec_txt(filename):
#     """ Parse a icdar2015ch4 txt file """
#     with open(filename.strip(), 'r') as f:
#         # rm utf8 info
#         lightSen = []
#         for line in f.readlines():
#             if '\xef\xbb\xbf' in line:
#                 str1 = line.replace('\xef\xbb\xbf', '')  # 用replace替换掉'\xef\xbb\xbf'
#                 lightSen.append(str1.strip())  # strip()去掉\n
#             else:
#                 lightSen.append(line.strip())
#
#         gts = lightSen
#         objects = []
#         for obj in gts:
#             cors = obj.strip().split(',')
#             gt_info = [float(cors[i]) for i in range(0, 8)]
#             tuple_points = [(gt_info[ind_p:ind_p + 2]) for ind_p in [0, 2, 4, 6]]
#             quadrilateral = Polygon(tuple_points)
#             assert quadrilateral.is_valid, ('Not a valid quadrilateral: {}'.format(obj))
#             bbox = quadrilateral.bounds
#             # change gt_info from absolute coordinates into relative ones to left top of box
#             lt_point_tile = np.tile(bbox[0:2], (1, 4))
#             # (N,8) = (N,8) - (N,2)
#             gt_info_rel = gt_info - lt_point_tile
#             cors = np.zeros(12)
#             cors[:4] = bbox
#             cors[4:] = gt_info_rel
#             # obj_struct['difficult'] =
#             obj_struct = {}
#             # class name
#             obj_struct['name'] = 'text'
#             obj_struct['difficult'] = 0
#             # (x_min y_min x_man ymax  x1  y1  x2  y2  x3  y3  x4  y4)
#             obj_struct['bbox'] = [int(cors[i]) for i in xrange(12)]
#             objects.append(obj_struct)
#             # assert(0), obj_struct['bbox']
#     return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)

    Steps:
        load gt
        extract gt objects for this class
        read dets
        sort detections by confidence
        go down dets and mark TPs and FPs
        compute precision and recall
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # ########### first load gt ###########
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f, open(annopath, 'r') as fa:
        lines = f.readlines()
        anno_lines = fa.readlines()
    imagenames = [x.strip() for x in lines]
    anno_names = [y.strip() for y in anno_lines]
    assert (len(imagenames) == len(anno_names)), 'each image should correspond to one label file'

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            print(anno_names[i].strip())
            recs[imagename] = parse_rec_txt(anno_names[i])
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # assert(0), recs     
    # ########### extract gt objects for this class ###########
    class_recs = {}
    npos = 0
    for ix, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]  # text
        assert (R), 'Can not find any object in ' + classname + ' class.'
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        # npos = npos
        # class_recs[imagename] = {'bbox': bbox,
        #                          'det': det}
        # index class
        class_recs[str(ix)] = {'bbox': bbox,
                               'det': det}
    # assert(0), class_recs 
    # ########### read dets ###########
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # ########### sort detections by confidence ###########
    # with '-', sort from large to small
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # ########### go down dets and mark TPs and FPs ###########
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # in each image
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
            # IoU
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            # if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # ########### compute precision and recall ###########
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    # print(rec, prec, ap)
    # yldebug = input('yldebug')
    return rec, prec, ap


# def voc_eval_polygon(detpath,
#                      annopath,
#                      imagesetfile,
#                      classname,
#                      cachedir,
#                      ovthresh=0.5,
#                      use_07_metric=False):
#     """rec, prec, ap = voc_eval(detpath,
#                                 annopath,
#                                 imagesetfile,
#                                 classname,
#                                 [ovthresh],
#                                 [use_07_metric])
#
#     Top level function that does the PASCAL VOC evaluation.
#
#     detpath: Path to detections
#         detpath.format(classname) should produce the detection results file.
#     annopath: Path to annotations
#         annopath.format(imagename) should be the xml annotations file.
#     imagesetfile: Text file containing the list of images, one image per line.
#     classname: Category name (duh)
#     cachedir: Directory for caching the annotations
#     [ovthresh]: Overlap threshold (default = 0.5)
#     [use_07_metric]: Whether to use VOC07's 11 point AP computation
#         (default False)
#
#     Steps:
#         load gt
#         extract gt objects for this class
#         read dets
#         sort detections by confidence
#         # go down dets and mark TPs and FPs
#         for each det in sorted order:
#             calculate overlaps with all gt in corresponding image
#             calculate max overlap over all gts in corresponding image
#             if max overlap > threshold:
#                 mark this det as TP
#             else:
#                 mark this det as FP
#         compute precision and recall(vectors by cumsum)
#         compute AP
#     """
#     # assumes detections are in detpath.format(classname)
#     # assumes annotations are in annopath.format(imagename)
#     # assumes imagesetfile is a text file with each line an image name
#     # cachedir caches the annotations in a pickle file
#
#     # ########### first load gt ###########
#     if not os.path.isdir(cachedir):
#         os.mkdir(cachedir)
#     cachefile = os.path.join(cachedir, 'annots.pkl')
#     print '\nLoading cachefile: {} ...'.format(cachefile)
#     # read list of images
#     with open(imagesetfile, 'r') as f, open(annopath, 'r') as fa:
#         lines = f.readlines()
#         anno_lines = fa.readlines()
#     imagenames = [x.strip() for x in lines]
#     anno_names = [y.strip() for y in anno_lines]
#     assert (len(imagenames) == len(anno_names)), 'each image should correspond to one label file'
#
#     use_update = 'u'
#     if os.path.isfile(cachefile):
#         # if not force to update cache file, use interactive mode to deal with it
#         if not cfg.FORCE_UPDATE_CACHE:
#             print 'Cache file already exists: {}'.format(cachefile)
#             use_update = raw_input('If you have not modify anything about test data set, '
#                                    'input \'u\' to update it or press any other key to use '
#                                    'cache.\n'
#                                    'Your input is: ')
#             # interactively chosen not update, then use cache
#             if use_update != 'u':
#                 # load
#                 with open(cachefile, 'r') as f:
#                     gt_objs_all = cPickle.load(f)
#                 print 'test data set annotations loaded from {}'.format(cachefile)
#     # load new annotations or update annotations
#     if use_update == 'u':
#         # load annots
#         gt_objs_all = {}
#         for i, imagename in enumerate(imagenames):
#             # gt_objs_all[imagename] = curve_parse_rec_txt(anno_names[i])
#             # each line in self._label_list_file is a relative path of data set name, so join the DATA_DIR
#             anno_names[i] = os.path.join(cfg.DATA_DIR, anno_names[i])
#             print(anno_names[i].strip())
#             gt_objs_all[imagename] = qua_parse_rec_txt(anno_names[i])
#
#             if i % 100 == 0:
#                 print 'Reading annotation for {:d}/{:d}'.format(
#                     i + 1, len(imagenames))
#         # save
#         print 'Saving cached annotations to {:s}'.format(cachefile)
#         with open(cachefile, 'w') as f:
#             cPickle.dump(gt_objs_all, f)
#
#     # ########### extract gt objects for this class ###########
#     class_gt_objs = {}
#     # number of not difficlut objects
#     num_obj_not_diff = 0
#     for ix, imagename in enumerate(imagenames):
#         cls_gt_objs_one_img = [obj for obj in gt_objs_all[imagename] if obj['name'] == classname]  # text
#         # assert(cls_gt_objs_one_img), 'Can not find any object in '+ classname+' class.'
#         if not cls_gt_objs_one_img: continue
#         bbox = np.array([x['bbox'] for x in cls_gt_objs_one_img])
#         difficult = np.array([x['difficult'] for x in cls_gt_objs_one_img]).astype(np.bool)
#         det = [False] * len(cls_gt_objs_one_img)
#         num_obj_not_diff = num_obj_not_diff + sum(~difficult)
#         # num_obj_not_diff = num_obj_not_diff
#         # class_gt_objs[imagename] = {'bbox': bbox,
#         #                          'det': det}
#         # index class
#         class_gt_objs[str(ix)] = {'bbox': bbox,
#                                   'det': det}
#     # ########### read dets(in absolute coordinates) ###########
#     detfile = detpath.format(classname)
#     with open(detfile, 'r') as f:
#         # (ind  score  x1  y1  x2  y2  x3  y3  x4  y4)
#         det_lines = f.readlines()
#
#     det_splitlines = [x.strip().split(' ') for x in det_lines]
#     det_lines_img_id = [x[0] for x in det_splitlines]
#     det_lines_conf = np.array([float(x[1]) for x in det_splitlines])
#     # (x1  y1  x2  y2  x3  y3  x4  y4)
#     det_lines_polygon = np.array([[float(z) for z in x[2:]] for x in det_splitlines])
#
#     # ########### sort all dets by det_lines_conf ###########
#     # with '-', sort from large to small
#     sorted_ind = np.argsort(-det_lines_conf)
#     sorted_scores = np.sort(-det_lines_conf)
#     det_lines_polygon_sorted = det_lines_polygon[sorted_ind, :]
#     det_lines_img_id_sorted = [det_lines_img_id[x] for x in sorted_ind]
#
#     # ########### go down dets and mark TPs and FPs in sorted order ###########
#     num_dets = len(det_lines_img_id_sorted)
#     tp = np.zeros(num_dets)
#     fp = np.zeros(num_dets)
#     # for each det in sorted order
#     for det_ind in range(num_dets):
#         class_gt_objs_one_image = class_gt_objs[det_lines_img_id_sorted[det_ind]]
#         # (x1  y1  x2  y2  x3  y3  x4  y4)
#         det_polygon_one_img = det_lines_polygon_sorted[det_ind, :].astype(float)
#         # (x1  y1  x2  y2  x3  y3  x4  y4)
#         det_polygon_pts = det_polygon_one_img[:2 * cfg.NUM_QUA_POINTS]
#         pts = tuple([(det_polygon_pts[i], det_polygon_pts[i + 1]) for i in range(0, 2 * cfg.NUM_QUA_POINTS, 2)])
#         # pts = ((det_polygon_pts[0], det_polygon_pts[1]), (det_polygon_pts[2], det_polygon_pts[3]), (det_polygon_pts[4], det_polygon_pts[5]), (det_polygon_pts[6], det_polygon_pts[7]), (det_polygon_pts[8], det_polygon_pts[9]), (det_polygon_pts[10], det_polygon_pts[11]), (det_polygon_pts[12], det_polygon_pts[13]),
#         #        (det_polygon_pts[14], det_polygon_pts[15]), (det_polygon_pts[16], det_polygon_pts[17]), (det_polygon_pts[18], det_polygon_pts[19]), (det_polygon_pts[20], det_polygon_pts[21]), (det_polygon_pts[22], det_polygon_pts[23]), (det_polygon_pts[24], det_polygon_pts[25]), (det_polygon_pts[26], det_polygon_pts[27]))
#         det_polygon = Polygon(pts)
#         # assert(det_polygon.is_valid), 'polygon has intersection sides.'
#         # if not det_polygon.is_valid:
#         # print('det_polygon')
#         # continue
#
#         ovmax = -np.inf
#         # (x_min y_min x_man ymax  x1  y1  x2  y2  x3  y3  x4  y4)
#         gt_bbox_polygon_one_img = class_gt_objs_one_image['bbox'].astype(float)
#         # print(gt_bbox_polygon_one_img)
#         #  (x_min y_min x_man ymax)
#         gt_bbox = gt_bbox_polygon_one_img[:, :4]
#         # (x1  y1  x2  y2  x3  y3  x4  y4))
#         gt_polygon_pts = gt_bbox_polygon_one_img[:, 4: 4 + 2 * cfg.NUM_QUA_POINTS]
#         ls_pgt = []
#         # calculate overlaps with all gt in corresponding image
#         overlaps = np.zeros(gt_bbox_polygon_one_img.shape[0])
#         # for each gt polygon in one image
#         for polygon_ind in xrange(gt_bbox_polygon_one_img.shape[0]):
#             # transform polygon points to absolute coordinates
#             pts = [(int(gt_bbox[polygon_ind, 0]) + gt_polygon_pts[polygon_ind, j],
#                     int(gt_bbox[polygon_ind, 1]) + gt_polygon_pts[polygon_ind, j + 1])
#                    for j in xrange(0, 2 * cfg.NUM_QUA_POINTS, 2)]
#             gt_polygon = Polygon(pts)
#             if not gt_polygon.is_valid:
#                 print('GT polygon has intersecting sides.')
#                 continue
#             try:
#                 sec = det_polygon.intersection(gt_polygon)
#             except Exception as e:
#                 print(e)
#                 continue
#             assert (sec.is_valid), 'polygon has intersection sides.'
#             inters = sec.area
#             uni = gt_polygon.area + det_polygon.area - inters
#             overlaps[polygon_ind] = inters * 1.0 / uni
#             # ls_overlaps.append(inters*1.0 / uni)
#         # calculate max overlap over all gts in corresponding image
#         ovmax = np.max(overlaps)
#         jmax = np.argmax(overlaps)
#         if ovmax > ovthresh:
#             # if not class_gt_objs_one_image['difficult'][jmax]:
#             if not class_gt_objs_one_image['det'][jmax]:
#                 tp[det_ind] = 1.
#                 class_gt_objs_one_image['det'][jmax] = 1
#             else:
#                 fp[det_ind] = 1.
#         else:
#             fp[det_ind] = 1.
#
#     # ########### compute precision and recall ###########
#     fp = np.cumsum(fp)
#     tp = np.cumsum(tp)
#     rec = tp / float(num_obj_not_diff)
#     # avoid divide by zero in case the first detection matches a difficult
#     # ground truth
#     prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#     ap = voc_ap(rec, prec, use_07_metric)
#     if DEBUG:
#         print('fp={}'.format(fp))
#         print('tp={}'.format(tp))
#         print 'num_obj_not_diff={}'.format(num_obj_not_diff)
#         print(rec, prec, ap)
#         # yldebug = input('yldebug')
#     return rec, prec, ap

# json
def voc_eval_polygon(detpath,
                     annopath,
                     imagesetfile,
                     classname,
                     cachedir,
                     ovthresh=0.5,
                     use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)

    Steps:
        load gt
        extract gt objects for this class
        read dets
        sort detections by confidence
        # go down dets and mark TPs and FPs
        for each det in sorted order:
            calculate overlaps with all gt in corresponding image
            calculate max overlap over all gts in corresponding image
            if max overlap > threshold:
                mark this det as TP
            else:
                mark this det as FP
        compute precision and recall(vectors by cumsum)
        compute AP
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # ########### first load gt ###########
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    print '\nLoading cachefile: {} ...'.format(cachefile)
    # read list of images
    with open(imagesetfile, 'r') as f, open(annopath, 'r') as fa:
        lines = f.readlines()
        anno_lines = fa.readlines()
    anno_names = [ os.path.join(cfg.DATA_DIR, "quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-dataset",
                                          'VOC2007', 'Annotations', x.strip() + '.json') for x in anno_lines]
    imagenames = [os.path.join(cfg.DATA_DIR, "quadrilateral-fisheye-ep21h-1-sekonix-2018-12-06-dataset", 'VOC2007', 'JPEGImages', y.strip() + '.jpg') for y in lines]
    assert (len(imagenames) == len(anno_names)), 'each image should correspond to one label file'

    use_update = 'u'
    if os.path.isfile(cachefile):
        # if not force to update cache file, use interactive mode to deal with it
        if not cfg.FORCE_UPDATE_CACHE:
            print 'Cache file already exists: {}'.format(cachefile)
            use_update = raw_input('If you have not modify anything about test data set, '
                                   'input \'u\' to update it or press any other key to use '
                                   'cache.\n'
                                   'Your input is: ')
            # interactively chosen not update, then use cache
            if use_update != 'u':
                # load
                with open(cachefile, 'r') as f:
                    gt_objs_all = cPickle.load(f)
                print 'test data set annotations loaded from {}'.format(cachefile)
    # load new annotations or update annotations
    if use_update == 'u':
        # load annots
        gt_objs_all = {}
        for i, imagename in enumerate(imagenames):
            # gt_objs_all[imagename] = curve_parse_rec_txt(anno_names[i])
            # each line in self._label_list_file is a relative path of data set name, so join the DATA_DIR
            anno_names[i] = anno_names[i]
            print(anno_names[i].strip())
            gt_objs_all[imagename] = qua_parse_rec_txt(anno_names[i])

            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(gt_objs_all, f)

    # ########### extract gt objects for this class ###########
    class_gt_objs = {}
    # number of not difficlut objects
    num_obj_not_diff = 0
    ind_class_gt_objs = 0
    for ix, imagename in enumerate(imagenames):
        gt_objs = gt_objs_all[imagename]
        cls_gt_objs_one_img = [obj for obj in gt_objs if obj['name'] == classname]  # text
        # assert(cls_gt_objs_one_img), 'Can not find any object in '+ classname+' class.'
        if not cls_gt_objs_one_img: continue
        bbox = np.array([x['bbox'] for x in cls_gt_objs_one_img])
        difficult = np.array([x['difficult'] for x in cls_gt_objs_one_img]).astype(np.bool)
        det = [False] * len(cls_gt_objs_one_img)
        num_obj_not_diff = num_obj_not_diff + sum(~difficult)
        # num_obj_not_diff = num_obj_not_diff
        # class_gt_objs[imagename] = {'bbox': bbox,
        #                          'det': det}
        # index class
        # class_gt_objs[str(ix)] = {'bbox': bbox,
        # to keep consistent with test roidb data
        class_gt_objs[str(ind_class_gt_objs)] = {'bbox': bbox,
                                  'det': det}
        #                                          'det': det}
        ind_class_gt_objs += 1
    # ########### read dets(in absolute coordinates) ###########
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        # (ind  score  x1  y1  x2  y2  x3  y3  x4  y4)
        det_lines = f.readlines()

    det_splitlines = [x.strip().split(' ') for x in det_lines]
    det_lines_img_id = [x[0] for x in det_splitlines]
    det_lines_conf = np.array([float(x[1]) for x in det_splitlines])
    # (x1  y1  x2  y2  x3  y3  x4  y4)
    det_lines_polygon = np.array([[float(z) for z in x[2:]] for x in det_splitlines])

    # ########### sort all dets by det_lines_conf ###########
    # with '-', sort from large to small
    sorted_ind = np.argsort(-det_lines_conf)
    sorted_scores = np.sort(-det_lines_conf)
    det_lines_polygon_sorted = det_lines_polygon[sorted_ind, :]
    det_lines_img_id_sorted = [det_lines_img_id[x] for x in sorted_ind]

    # ########### go down dets and mark TPs and FPs in sorted order ###########
    num_dets = len(det_lines_img_id_sorted)
    tp = np.zeros(num_dets)
    fp = np.zeros(num_dets)
    # for each det in sorted order
    for det_ind in range(num_dets):
        class_gt_objs_one_image = class_gt_objs[det_lines_img_id_sorted[det_ind]]
        # (x1  y1  x2  y2  x3  y3  x4  y4)
        det_polygon_one_img = det_lines_polygon_sorted[det_ind, :].astype(float)
        # (x1  y1  x2  y2  x3  y3  x4  y4)
        det_polygon_pts = det_polygon_one_img[:2 * cfg.NUM_QUA_POINTS]
        pts = tuple([(det_polygon_pts[i], det_polygon_pts[i + 1]) for i in range(0, 2 * cfg.NUM_QUA_POINTS, 2)])
        # pts = ((det_polygon_pts[0], det_polygon_pts[1]), (det_polygon_pts[2], det_polygon_pts[3]), (det_polygon_pts[4], det_polygon_pts[5]), (det_polygon_pts[6], det_polygon_pts[7]), (det_polygon_pts[8], det_polygon_pts[9]), (det_polygon_pts[10], det_polygon_pts[11]), (det_polygon_pts[12], det_polygon_pts[13]),
        #        (det_polygon_pts[14], det_polygon_pts[15]), (det_polygon_pts[16], det_polygon_pts[17]), (det_polygon_pts[18], det_polygon_pts[19]), (det_polygon_pts[20], det_polygon_pts[21]), (det_polygon_pts[22], det_polygon_pts[23]), (det_polygon_pts[24], det_polygon_pts[25]), (det_polygon_pts[26], det_polygon_pts[27]))
        det_polygon = Polygon(pts)
        # assert(det_polygon.is_valid), 'polygon has intersection sides.'
        # if not det_polygon.is_valid:
        # print('det_polygon')
        # continue

        ovmax = -np.inf
        # (x_min y_min x_man ymax  x1  y1  x2  y2  x3  y3  x4  y4)
        gt_bbox_polygon_one_img = class_gt_objs_one_image['bbox'].astype(float)
        # print(gt_bbox_polygon_one_img)
        #  (x_min y_min x_man ymax)
        gt_bbox = gt_bbox_polygon_one_img[:, :4]
        # (x1  y1  x2  y2  x3  y3  x4  y4))
        gt_polygon_pts = gt_bbox_polygon_one_img[:, 4: 4 + 2 * cfg.NUM_QUA_POINTS]
        ls_pgt = []
        # calculate overlaps with all gt in corresponding image
        overlaps = np.zeros(gt_bbox_polygon_one_img.shape[0])
        # for each gt polygon in one image
        for polygon_ind in xrange(gt_bbox_polygon_one_img.shape[0]):
            # transform polygon points to absolute coordinates
            pts = [(int(gt_bbox[polygon_ind, 0]) + gt_polygon_pts[polygon_ind, j],
                    int(gt_bbox[polygon_ind, 1]) + gt_polygon_pts[polygon_ind, j + 1])
                   for j in xrange(0, 2 * cfg.NUM_QUA_POINTS, 2)]
            gt_polygon = Polygon(pts)
            if not gt_polygon.is_valid:
                print('GT polygon has intersecting sides.')
                continue
            try:
                sec = det_polygon.intersection(gt_polygon)
            except Exception as e:
                print(e)
                continue
            assert (sec.is_valid), 'polygon has intersection sides.'
            inters = sec.area
            uni = gt_polygon.area + det_polygon.area - inters
            overlaps[polygon_ind] = inters * 1.0 / uni
            # ls_overlaps.append(inters*1.0 / uni)
        # calculate max overlap over all gts in corresponding image
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            # if not class_gt_objs_one_image['difficult'][jmax]:
            if not class_gt_objs_one_image['det'][jmax]:
                tp[det_ind] = 1.
                class_gt_objs_one_image['det'][jmax] = 1
            else:
                fp[det_ind] = 1.
        else:
            fp[det_ind] = 1.

    # ########### compute precision and recall ###########
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_obj_not_diff)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    if DEBUG:
        print('fp={}'.format(fp))
        print('tp={}'.format(tp))
        print 'num_obj_not_diff={}'.format(num_obj_not_diff)
        print(rec, prec, ap)
        # yldebug = input('yldebug')
    return rec, prec, ap


if __name__ == '__main__':
    detpath = 'temp_dir/dets.txt'
    def qua_write_voc_results_file():
        print 'Writing {} VOC results file'.format('text')

        index = 0
        dets = np.array([
           922, 255, 954, 255, 956, 277, 936, 277,
           172, 323, 195, 324, 195, 339, 177, 339,
           83, 270, 118, 271, 115, 294, 88, 291,
           940, 310, 962, 310, 962, 320, 940, 320,
           946, 356, 976, 351, 978, 368, 950, 374,
           940, 322, 962, 322, 964, 333, 943, 334,
           128, 344, 210, 342, 206, 361, 128, 362,
           312, 303, 360, 303, 360, 312, 312, 312,
        ])
        dets = dets.reshape(-1, 8)
        score = 0.6
        with open(detpath, 'wt') as f:
            for k in xrange(dets.shape[0]):
                # (ind  score  x1  y1  x2  y2  x3  y3  x4  y4)
                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        format(str(index), score,
                               dets[k, 0],
                               dets[k, 1], dets[k, 2],
                               dets[k, 3], dets[k, 4],
                               dets[k, 5], dets[k, 6],
                               dets[k, 7]))
    qua_write_voc_results_file()
    label_list_file = '../../data/icdar2015ch4/Challenge4_Test_Task1_GT.txt'
    image_list_file = '../../data/icdar2015ch4/ch4_test_images.txt'
    cls = 'text'
    cachedir = 'temp_dir'
    # if not os.path.exists(cachedir):
    #     os.mkdir(cachedir)
    use_07_metric = True
    rec, prec, ap = voc_eval_polygon(
        detpath, label_list_file, image_list_file,
        cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
    print(rec, prec, ap)

# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

# import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np

from shapely.geometry import *
from fast_rcnn.config import cfg
def parse_rec_txt(filename):
    with open(filename.strip(),'r') as f:
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
    with open(filename.strip(),'r') as f:
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
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f, open(annopath, 'r') as fa:
        lines = f.readlines()
        anno_lines = fa.readlines()
    imagenames = [x.strip() for x in lines]
    anno_names = [y.strip() for y in anno_lines]
    assert(len(imagenames) == len(anno_names)), 'each image should correspond to one label file'

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
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for ix, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # text
        assert(R), 'Can not find any object in '+ classname+' class.'
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
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
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

    # compute precision recall
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
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f, open(annopath, 'r') as fa:
        lines = f.readlines()
        anno_lines = fa.readlines()
    imagenames = [x.strip() for x in lines]
    anno_names = [y.strip() for y in anno_lines]
    assert(len(imagenames) == len(anno_names)), 'each image should correspond to one label file'

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
        recs = {}
        for i, imagename in enumerate(imagenames):
            print(anno_names[i].strip())
            recs[imagename] = curve_parse_rec_txt(anno_names[i])
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

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for ix, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # text
        # assert(R), 'Can not find any object in '+ classname+' class.'
        if not R: continue
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
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        det_bbox = bb[:28]
        pts = tuple([(det_bbox[i], det_bbox[i+1]) for i in range(0,28,2)])
        # pts = ((det_bbox[0], det_bbox[1]), (det_bbox[2], det_bbox[3]), (det_bbox[4], det_bbox[5]), (det_bbox[6], det_bbox[7]), (det_bbox[8], det_bbox[9]), (det_bbox[10], det_bbox[11]), (det_bbox[12], det_bbox[13]),
        #        (det_bbox[14], det_bbox[15]), (det_bbox[16], det_bbox[17]), (det_bbox[18], det_bbox[19]), (det_bbox[20], det_bbox[21]), (det_bbox[22], det_bbox[23]), (det_bbox[24], det_bbox[25]), (det_bbox[26], det_bbox[27]))
        pdet = Polygon(pts)
        # assert(pdet.is_valid), 'polygon has intersection sides.'
        # if not pdet.is_valid:
            # print('pdet')
            # continue

        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        # print(BBGT)
        gt_bbox = BBGT[:, :4]
        info_bbox_gt = BBGT[:, 4:32]
        ls_pgt = []
        overlaps = np.zeros(BBGT.shape[0])
        for iix in xrange(BBGT.shape[0]):
            pts = [(int(gt_bbox[iix, 0]) + info_bbox_gt[iix, j], int(gt_bbox[iix, 1]) + info_bbox_gt[iix, j+1]) for j in xrange(0,28,2)]
            pgt = Polygon(pts)
            if not pgt.is_valid:
                print('GT polygon has intersecting sides.')
                continue
            try:
                sec = pdet.intersection(pgt)
            except Exception as e:
                print(e)
                continue
            assert(sec.is_valid), 'polygon has intersection sides.'
            inters = sec.area
            uni = pgt.area + pdet.area - inters
            overlaps[iix] = inters*1.0 / uni
            # ls_overlaps.append(inters*1.0 / uni)

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

    # compute precision recall
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
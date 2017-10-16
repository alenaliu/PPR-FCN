import _init_paths
from vrd.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import h5py
import cv2

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2

import caffe
import h5py
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
import random
import utils.zl_utils as zl
def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors


def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    # if cfg.TEST.BBOX_REG:
    if True:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
    fc7 = net.blobs['fc7'].data
    return net.blobs['cls_score'].data[:, :], scores, fc7, pred_boxes

def detect(im,net,thresh,imid,m):
    rets = []
    box_proposals = None
    score_raw, scores, fc7, boxes = im_detect(net, im, box_proposals)
    boxes_tosort = []
    for j in xrange(1, 201):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(cls_dets, .5, force_cpu=True)  # nms threshold
        cls_dets = cls_dets[keep, :]
        boxes_tosort.append(cls_dets)

    for j in xrange(len(boxes_tosort)):
        cls_dets = boxes_tosort[j]
        for di in xrange(cls_dets.shape[0]):
            di = cls_dets[di]
            score = di[-1]
            cls_idx = j + 1
            #cls_name = str(m['meta/cls/idx2name/' + str(cls_idx)][...])
            cls_name = zl.idx2name_cls(m,cls_idx)
            if score > 1:
                score = 1
            ret = {}
            ret['cls_name'] = cls_name
            ret['x1'] = di[0]
            ret['y1'] = di[1]
            ret['x2'] = di[2]
            ret['y2'] = di[3]
            ret['score'] = score
            rets.append(ret)
            # cv2.putText(im, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            # cv2.rectangle(im, (di[0], di[1]), (di[2], di[3]), (255, 0, 0), 2)
    return rets

def run_test_visualize():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5', 'r', 'core')
    net = caffe.Net('models/vg1_2/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/models/vg1_2_vgg16_faster_rcnn_finetune_iter_120000.caffemodel',
                    caffe.TEST)

    net2 = caffe.Net('models/vg1_2/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/models/vg1_2_vgg16_faster_rcnn_finetune_iter_55000.caffemodel',
                    caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    # if not cfg.TEST.HAS_RPN:
        # imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    """Test a Fast R-CNN network on an image database."""
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    root = 'data/vg1_2_2016/Data/test/'
    cnt = 0
    cv2.namedWindow('im',0)
    cv2.namedWindow('im_single',0)
    images = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt += 1
            imid= name.split('.')[0]
            fpath = os.path.join(path, name)
            images.append(fpath)
    random.shuffle(images)
    for fpath in images:

        imid = fpath.split('/')[-1].replace('.jpg','')
        print imid
        imorig = cv2.imread(fpath)
        dets1 = detect(imorig,net,0.01,imid,m)
        dets2 = detect(imorig,net2,0.01,imid,m)
        zl.sort_dict_by_key_val(dets1,'score')
        zl.sort_dict_by_key_val(dets2,'score')
        dets1=dets1[::-1]
        dets2=dets2[::-1]
        # print dets1[:4]
        # exit(0)
        cv2.namedWindow('ctrl')
        cv2.destroyWindow('ctrl')
        cv2.namedWindow('ctrl')
        cv2.createTrackbar('t1','ctrl',200,1000,nothing)
        cv2.createTrackbar('t2','ctrl',200,1000,nothing)
        cv2.createTrackbar('t1idx','ctrl',0,len(dets1)-1,nothing)
        cv2.createTrackbar('t2idx','ctrl',0,len(dets2)-1,nothing)
        while True:
            saved_idx = 0
            im = imorig.copy()
            im_single = imorig.copy()
            t1 = cv2.getTrackbarPos('t1','ctrl')/1000.
            t2 = cv2.getTrackbarPos('t2','ctrl')/1000.

            t1idx = cv2.getTrackbarPos('t1idx','ctrl')
            t2idx = cv2.getTrackbarPos('t2idx','ctrl')
            d1 = dets1[t1idx]
            cls_name = d1['cls_name']+'.'+str(t1idx)
            di = [d1['x1'],d1['y1'],d1['x2'],d1['y2']]
            x, y = int(di[0]), int(di[1])
            if x < 10:
                x = 15
            if y < 10:
                y = 15
            cv2.putText(im_single, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.rectangle(im_single, (di[0], di[1]), (di[2], di[3]), (255, 0, 0), 2)

            d2 = dets2[t2idx]
            cls_name = d2['cls_name']+'.'+str(t2idx)
            di = [d2['x1'],d2['y1'],d2['x2'],d2['y2']]
            x, y = int(di[0]), int(di[1])
            if x < 10:
                x = 15
            if y < 10:
                y = 15
            cv2.putText(im_single, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.rectangle(im_single, (di[0], di[1]), (di[2], di[3]), (0, 255, 0), 2)

            for i in xrange(len(dets1)):
                d = dets1[i]
                score = d['score']
                if score <t1:continue
                cls_name = d['cls_name']+'.'+str(i)
                di = [d['x1'],d['y1'],d['x2'],d['y2']]

                x, y = int(di[0]), int(di[1])
                if x < 10:
                    x = 15
                if y < 10:
                    y = 15
                cv2.putText(im, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.rectangle(im, (di[0], di[1]), (di[2], di[3]), (255, 0, 0), 2)
                #print '%s %f %d %d %d %f\n' % (imid,  score, di[0], di[1], di[2], di[3])
            for i in xrange(len(dets2)):
                d = dets2[i]
                score = d['score']

                if score <t2:continue
                cls_name = d['cls_name']+'.'+str(i)
                di = [d['x1'],d['y1'],d['x2'],d['y2']]

                x, y = int(di[0]), int(di[1])
                if x < 10:
                    x = 15
                if y < 10:
                    y = 15
                cv2.putText(im, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.rectangle(im, (di[0], di[1]), (di[2], di[3]), (0, 255, 0), 2)
                #print '%s %f %d %d %d %f\n' % (imid,  score, di[0], di[1], di[2], di[3])
            cv2.imshow('im', im)
            cv2.imshow('im_single', im_single)
            c = cv2.waitKey(100)&0xFF
            if c == ord('s'):
                im_folder = 'output/results/examples_obj/'+imid
                zl.make_dirs(im_folder)
                if not os.path.exists('output/results/examples_obj/'+imid+'/orig_'+imid+'.jpg'):
                    cv2.imwrite('output/results/examples_obj/'+imid+'/orig_'+imid+'.jpg',imorig)
                cv2.imwrite('output/results/examples_obj/'+imid+'/single_'+imid+'.%d.jpg'%saved_idx,im_single)
                saved_idx+=1
            elif c== ord(' '):
                break
            elif c == 27:
                exit(0)
def nothing(x):
    pass

def run_single(imid):
    caffe.set_mode_gpu()
    caffe.set_device(0)
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    net = caffe.Net('models/sg_vrd/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/faster_rcnn_end2end/sg_vrd_2016_train/vgg16_faster_rcnn_iter_80000.caffemodel',
                    caffe.TEST)

    net2 = caffe.Net('models/sg_vrd/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/faster_rcnn_end2end/sg_vrd_2016_train/vgg16_faster_rcnn_finetune_iter_95000.caffemodel',
                    caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    imdb = get_imdb('sg_vrd_2016_test')
    imdb.competition_mode(0)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    cnt = 0
    fpath = root+imid+'.jpg'
    im_orig = cv2.imread(fpath)
    if im_orig == None:
        print fpath

    cv2.namedWindow('ctrl')
    cv2.createTrackbar('t1','ctrl',200,1000,nothing)
    cv2.createTrackbar('t2','ctrl',200,1000,nothing)

    dets1 = detect(im_orig,net,0.1,imid,m)
    dets2 = detect(im_orig,net2,0.1,imid,m)
    print_text = True
    while True:
        im = im_orig.copy()
        t1 = cv2.getTrackbarPos('t1','ctrl')/1000.
        t2 = cv2.getTrackbarPos('t2','ctrl')/1000.
        # t1idx = []
        # t2idx = []
        t1idx = [2,10]
        t2idx = [1,9]
        for i in xrange(len(dets1)):
            #if t1idx != -1 and t1idx!=i: continue
            if len(t1idx)>0 and i not in t1idx:continue
            d = dets1[i]
            score = d['score']
            if score <t1:continue
            cls_name = d['cls_name']+'.'+str(i)
            di = [d['x1'],d['y1'],d['x2'],d['y2']]

            x, y = int(di[0]), int(di[1])
            if x < 10:
                x = 15
            if y < 10:
                y = 15
            if print_text:
                cv2.putText(im, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.rectangle(im, (di[0], di[1]), (di[2], di[3]), (0, 0, 255), 1)
            #print '%s %f %d %d %d %f\n' % (imid,  score, di[0], di[1], di[2], di[3])
        for i in xrange(len(dets2)):
            if len(t2idx)>0 and i not in t2idx:continue
            d = dets2[i]
            score = d['score']
            if score <t2:continue
            cls_name = d['cls_name']+'.'+str(i)
            di = [d['x1'],d['y1'],d['x2'],d['y2']]

            x, y = int(di[0]), int(di[1])
            if x < 10:
                x = 15
            if y < 10:
                y = 15
            if print_text:
                cv2.putText(im, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.rectangle(im, (di[0], di[1]), (di[2], di[3]), (0, 255, 0), 1)
            #print '%s %f %d %d %d %f\n' % (imid,  score, di[0], di[1], di[2], di[3])
        cv2.imshow('im',im)
        c = cv2.waitKey(100)&0xFF
        if c == ord('s'):
            cv2.imwrite('output/results/'+imid+'.jpg',im)
            print 'written'
        elif c == ord('b'):
            print_text=not print_text
        elif c== ord(' '):
            break
        elif c == 27:
            exit(0)
run_test_visualize()
#run_single('5615329604_4e715d25db_b')

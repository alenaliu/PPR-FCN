#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo_orig(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    im_detect(net, im)
    im_info = net.blobs['im_info'].data
    rois = net.blobs['rois'].data
    rois = rois/net.blobs['im_info'].data[0,2]
    roi_scores = net.blobs['rois_score'].data
    attention = net.blobs['attention'].data.squeeze()
    ind = np.argsort(attention)[::-1]
    attention = attention[ind]
    rois_all = np.hstack((rois[:,1:],roi_scores))
    rois_all =  rois_all[ind]
    for i in xrange(5):
        ascore = attention[i]
        roi = rois_all[i]
        cv2.rectangle(im,(roi[0],roi[1]),(roi[2],roi[3]),(255,0,0),1)

    cv2.imshow('im',im)
    cv2.waitKey(0)
    timer.toc()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    im_detect(net, im)
    cls_score = net.blobs['cls_score'].data.copy()
    cls_score_reindexed_caffe= net.blobs['cls_score_reindexed'].data.copy()
    vatt_caffe = net.blobs['vatt'].data.copy()
    cls_score_tiled_caffe= net.blobs['cls_score_tiled'].data.copy()
    cls_score_tiled_transposed_caffe = net.blobs['cls_score_tiled_transposed'].data.copy()
    vatt_raw_caffe = net.blobs['vatt_raw'].data.copy()
    attention_caffe = net.blobs['attention'].data.copy()
    attention_tiled_caffe = net.blobs['attention_tiled'].data.copy()
    cls_score_tiled_caffe = net.blobs['cls_score_tiled'].data.copy()

    cls_score_transposed = cls_score.transpose((1,0,2,3))
    cls_score_reindexed = cls_score_transposed[15,...]

    attention = softmax(cls_score_reindexed.squeeze())
    rois = net.blobs['rois'].data
    rois = rois/net.blobs['im_info'].data[0,2]
    roi_scores = net.blobs['rois_score'].data

    vatt = np.zeros((rois.shape[0],21,1,1),np.float32)
    for i in xrange(vatt.shape[0]):
        vatt[i] += attention[i] * cls_score[i]
    #vatt = vatt.sum(axis=0)
    vatt_summed= vatt.sum(axis=0)
    attention = net.blobs['attention'].data[:,0].squeeze()
    ind = np.argsort(attention)[::-1]
    attention = attention[ind]
    rois_all = np.hstack((rois[:,1:],roi_scores))
    rois_all =  rois_all[ind]
    for i in xrange(5):
        ascore = attention[i]
        roi = rois_all[i]
        cv2.rectangle(im,(roi[0],roi[1]),(roi[2],roi[3]),(255,0,0),1)

    cv2.imshow('im',im)
    cv2.waitKey(0)
    timer.toc()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-50')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_iccv_rpn.prototxt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models_iccv/resnet50_rfcn_iter_10000.caffemodel')

    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models/resnet50_rfcn_final.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        im_detect(net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                #'001763.jpg', '004545.jpg']
    im_names = ['004545.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
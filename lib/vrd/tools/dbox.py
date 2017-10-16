from fast_dbox_config import cfg, get_output_path
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
import cPickle
import heapq
import utils.blob
import os
import pdb
import scipy.io as sio

def _get_image_blob(im):
    im_pyra = []
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
    blob = utils.blob.im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    rois, levels = _scale_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))[:, :, np.newaxis, np.newaxis]
    return rois_blob.astype(np.float32, copy=False)

def _scale_im_rois(im_rois, scales):
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
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def im_obj_detect(net, im, boxes):
    blobs, im_scale_factors = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'][:, :, 0, 0] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :, :, :]
        boxes = boxes[index, :]

    # reshape network inputs
    base_shape = blobs['data'].shape
    num_rois = blobs['rois'].shape[0]
    net.blobs['data'].reshape(base_shape[0], base_shape[1],
                              base_shape[2], base_shape[3])
    net.blobs['rois'].reshape(num_rois, 5, 1, 1)

    rois=blobs['rois'].astype(np.float32, copy=False)
    data=blobs['data'].astype(np.float32, copy=False)

    blobs_out = net.forward(data=data,rois=rois)
    # use softmax estimated probabilities
    scores = blobs_out['cls_prob']
    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]

    return scores

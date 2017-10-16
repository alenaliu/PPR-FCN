import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import h5py
import cv2

import scipy.io as sio

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
    # if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        # hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        # _, index, inv_index = np.unique(hashes, return_index=True,
                                        # return_inverse=True)
        # blobs['rois'] = blobs['rois'][index, :]
        # boxes = boxes[index, :]

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
    if False:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    # if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # # Map scores and predictions back to the original set of boxes
        # scores = scores[inv_index, :]
        # pred_boxes = pred_boxes[inv_index, :]
    fc7 = net.blobs['fc7'].data
    return net.blobs['cls_score'].data[:, :], scores, fc7, pred_boxes

def prep_jointbox_train():
    caffe.set_mode_gpu()
    caffe.set_device(0)

    rdata = sio.loadmat('data/meta/vrd/annotation_train.mat', struct_as_record=False,squeeze_me=True)
    # map im_id to annotation
    r_annos = {}
    for i in xrange(len(rdata['annotation_train'])):
        anno = rdata['annotation_train'][i]
        im_id = anno.filename.split('.')[0]
        r_annos[im_id] = anno

    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')

    net = caffe.Net('models/sg_vrd/vgg16/faster_rcnn_end2end/test_jointbox.prototxt',
                    'output/faster_rcnn_end2end/sg_vrd_2016_train/vgg16_faster_rcnn_finetune_iter_40000.caffemodel',caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    imdb = get_imdb('sg_vrd_2016_train')
    imdb.competition_mode(0)
    cfg.TEST.HAS_RPN=False
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    h5path = 'output/' + imdb.name + '_predicate_exp_train.hdf5'

    h5f = h5py.File(h5path)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    root = 'data/sg_vrd_2016/Data/sg_train_images/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = .01
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt += 1
            if cnt %100==0:
                print cnt
            im_id = name.split('.')[0]

            fpath = os.path.join(path, name)
            im = cv2.imread(fpath)
            if im == None:
                print fpath
            r_anno = r_annos[im_id]
            sub_boxes = []
            obj_boxes = []
            joint_boxes = []

            boxes_batch = []
            b_type = {}
            b_idx = 0
            sub_visual = []
            obj_visual = []
            joint_visual = []
            pre_label = []
            if hasattr(r_anno, 'relationship'):
                if not isinstance(r_anno.relationship, np.ndarray):
                    r_anno.relationship = [r_anno.relationship]
                for r in xrange(len(r_anno.relationship)):
                    if not hasattr(r_anno.relationship[r], 'phrase'):
                        continue
                    predicate = r_anno.relationship[r].phrase[1]
                    pre_idx = int(str(m['meta/pre/name2idx/' + predicate][...]))
                    pre_label.append(pre_idx)
                    sub_lbl = r_anno.relationship[r].phrase[0]
                    obj_lbl = r_anno.relationship[r].phrase[2]
                    #print sub_lbl,predicate,obj_lbl
                    ymin, ymax, xmin, xmax = r_anno.relationship[r].subBox
                    sub_bbox = [xmin, ymin, xmax, ymax]
                    ymin, ymax, xmin, xmax = r_anno.relationship[r].objBox
                    obj_bbox= [xmin, ymin, xmax, ymax]
                    joint_bbox = [min(sub_bbox[0],obj_bbox[0]), min(sub_bbox[1],obj_bbox[1]),max(sub_bbox[2],obj_bbox[2]),max(sub_bbox[3],obj_bbox[3])]

                    joint_boxes.append(joint_bbox)
                    sub_boxes.append(sub_bbox)
                    obj_boxes.append(obj_bbox)
                    # cv2.rectangle(im,(joint_bbox[0],joint_bbox[1]),(joint_bbox[2],joint_bbox[3]),(255,255,255),4)
                    # cv2.rectangle(im,(sub_bbox[0],sub_bbox[1]),(sub_bbox[2],sub_bbox[3]),(0,0,255),2)
                    # cv2.rectangle(im,(obj_bbox[0],obj_bbox[1]),(obj_bbox[2],obj_bbox[3]),(255,0,0),2)

                for i in xrange(len(sub_boxes)):
                    boxes_batch.append(sub_boxes[i])
                    b_type[b_idx]='s'
                    b_idx += 1
                for i in xrange(len(obj_boxes)):
                    boxes_batch.append(obj_boxes[i])
                    b_type[b_idx]='o'
                    b_idx += 1
                for i in xrange(len(joint_boxes)):
                    boxes_batch.append(joint_boxes[i])
                    b_type[b_idx]='j'
                    b_idx += 1
                box_proposals = None
                _t['im_detect'].tic()
                score_raw, scores, fc7, boxes = im_detect(net, im, np.array(boxes_batch))

                for i in xrange(scores.shape[0]):
                    s_idx = np.argmax(scores[i,1:])+1
                    cls_box=None
                    cls_box = boxes[i, s_idx * 4:(s_idx + 1) * 4]
                    if b_type[i] == 's':
                        sub_visual.append(fc7[i])
                    if b_type[i] == 'o':
                        obj_visual.append(fc7[i])
                    if b_type[i] == 'j':
                        joint_visual.append(fc7[i])
                    # cls_name = str(m['meta/cls/idx2name/' + str(s_idx)][...])
                    # if b_type[i] == 's':
                        # print str(m['meta/pre/idx2name/'+str(pre_label[i])][...])
                    # cv2.rectangle(im,(cls_box[0],cls_box[1]),(cls_box[2],cls_box[3]),(255,0,0),2)
                # cv2.imshow('im',im)
                _t['im_detect'].toc()

                _t['misc'].tic()
                sub_visual= np.array(sub_visual).astype(np.float16)
                obj_visual= np.array(obj_visual).astype(np.float16)
                joint_visual= np.array(joint_visual).astype(np.float16)
                pre_label = np.array(pre_label).astype(np.int32)
                h5f.create_dataset(im_id + '/sub_visual', dtype='float16', data=sub_visual)
                h5f.create_dataset(im_id + '/obj_visual', dtype='float16', data=obj_visual)
                h5f.create_dataset(im_id + '/joint_visual', dtype='float16', data=joint_visual)
                h5f.create_dataset(im_id + '/pre_label', dtype='float16', data=pre_label)
                _t['misc'].toc()
                # rpn_rois = net.blobs['rois'].data
                # pool5 = net.blobs['pool5'].data
                # _t['misc'].tic()
                # cnt += 1
                print 'im_detect: {:d} {:.3f}s {:.3f}s' \
                    .format(cnt, _t['im_detect'].average_time,
                            _t['misc'].average_time)

def prep_jointbox(db_type):
    caffe.set_mode_gpu()
    caffe.set_device(0)

    rdata = sio.loadmat('data/meta/vrd/annotation_%s.mat'%db_type, struct_as_record=False,squeeze_me=True)
    # map im_id to annotation
    r_annos = {}
    for i in xrange(len(rdata['annotation_%s'%db_type])):
        anno = rdata['annotation_%s'%db_type][i]
        im_id = anno.filename.split('.')[0]
        r_annos[im_id] = anno

    m = h5py.File('data/sg_vrd_meta.h5', 'r', 'core')

    net = caffe.Net('models/sg_vrd/vgg16/faster_rcnn_end2end/test_jointbox.prototxt',
                    'output/faster_rcnn_end2end/sg_vrd_2016_train/vgg16_faster_rcnn_finetune_iter_40000.caffemodel',caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    imdb = get_imdb('sg_vrd_2016_test')
    imdb.competition_mode(0)
    cfg.TEST.HAS_RPN=False
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    h5path = 'output/' + imdb.name + '_predicate_exp_%s.hdf5'%db_type

    h5f = h5py.File(h5path)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    root = 'data/sg_vrd_2016/Data/sg_%s_images/'%db_type
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = .01
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt += 1
            if cnt %100==0:
                print cnt
            im_id = name.split('.')[0]

            fpath = os.path.join(path, name)
            im = cv2.imread(fpath)
            if im == None:
                print fpath
            r_anno = r_annos[im_id]
            sub_boxes = []
            obj_boxes = []
            joint_boxes = []

            boxes_batch = []
            b_type = {}
            b_idx = 0
            sub_visual = []
            obj_visual = []
            joint_visual = []
            sub_classeme = []
            obj_classeme = []
            joint_classeme = []
            pre_label = []
            sub_cls = []
            obj_cls = []
            if hasattr(r_anno, 'relationship'):
                if not isinstance(r_anno.relationship, np.ndarray):
                    r_anno.relationship = [r_anno.relationship]
                for r in xrange(len(r_anno.relationship)):
                    if not hasattr(r_anno.relationship[r], 'phrase'):
                        continue
                    predicate = r_anno.relationship[r].phrase[1]
                    sub = r_anno.relationship[r].phrase[0]
                    obj = r_anno.relationship[r].phrase[2]
                    pre_idx = int(str(m['meta/pre/name2idx/' + predicate][...]))
                    sub_cls_idx = int(str(m['meta/cls/name2idx/' + sub][...]))
                    obj_cls_idx = int(str(m['meta/cls/name2idx/' + obj][...]))

                    sub_cls.append(sub_cls_idx)
                    obj_cls.append(obj_cls_idx)
                    pre_label.append(pre_idx)

                    sub_lbl = r_anno.relationship[r].phrase[0]
                    obj_lbl = r_anno.relationship[r].phrase[2]
                    #print sub_lbl,predicate,obj_lbl
                    ymin, ymax, xmin, xmax = r_anno.relationship[r].subBox
                    sub_bbox = [xmin, ymin, xmax, ymax]
                    ymin, ymax, xmin, xmax = r_anno.relationship[r].objBox
                    obj_bbox= [xmin, ymin, xmax, ymax]
                    joint_bbox = [min(sub_bbox[0],obj_bbox[0]), min(sub_bbox[1],obj_bbox[1]),max(sub_bbox[2],obj_bbox[2]),max(sub_bbox[3],obj_bbox[3])]

                    joint_boxes.append(joint_bbox)
                    sub_boxes.append(sub_bbox)
                    obj_boxes.append(obj_bbox)
                    # cv2.rectangle(im,(joint_bbox[0],joint_bbox[1]),(joint_bbox[2],joint_bbox[3]),(255,255,255),4)
                    # cv2.rectangle(im,(sub_bbox[0],sub_bbox[1]),(sub_bbox[2],sub_bbox[3]),(0,0,255),2)
                    # cv2.rectangle(im,(obj_bbox[0],obj_bbox[1]),(obj_bbox[2],obj_bbox[3]),(255,0,0),2)

                for i in xrange(len(sub_boxes)):
                    boxes_batch.append(sub_boxes[i])
                    b_type[b_idx]='s'
                    b_idx += 1
                for i in xrange(len(obj_boxes)):
                    boxes_batch.append(obj_boxes[i])
                    b_type[b_idx]='o'
                    b_idx += 1
                for i in xrange(len(joint_boxes)):
                    boxes_batch.append(joint_boxes[i])
                    b_type[b_idx]='j'
                    b_idx += 1
                box_proposals = None
                _t['im_detect'].tic()
                score_raw, scores, fc7, boxes = im_detect(net, im, np.array(boxes_batch))

                for i in xrange(scores.shape[0]):
                    s_idx = np.argmax(scores[i,1:])+1
                    cls_box=None
                    cls_box = boxes[i, s_idx * 4:(s_idx + 1) * 4]
                    if b_type[i] == 's':
                        sub_visual.append(fc7[i])
                        sub_classeme.append(scores[i])
                    if b_type[i] == 'o':
                        obj_visual.append(fc7[i])
                        obj_classeme.append(scores[i])
                    if b_type[i] == 'j':
                        joint_visual.append(fc7[i])
                        joint_classeme.append(scores[i])
                    # cls_name = str(m['meta/cls/idx2name/' + str(s_idx)][...])
                    # if b_type[i] == 's':
                        # print str(m['meta/pre/idx2name/'+str(pre_label[i])][...])
                    # cv2.rectangle(im,(cls_box[0],cls_box[1]),(cls_box[2],cls_box[3]),(255,0,0),2)
                # cv2.imshow('im',im)
                _t['im_detect'].toc()

                _t['misc'].tic()
                sub_visual= np.array(sub_visual).astype(np.float16)
                obj_visual= np.array(obj_visual).astype(np.float16)
                joint_visual= np.array(joint_visual).astype(np.float16)
                sub_classeme= np.array(sub_classeme).astype(np.float16)
                obj_classeme= np.array(obj_classeme).astype(np.float16)
                joint_classeme= np.array(joint_classeme).astype(np.float16)
                pre_label = np.array(pre_label).astype(np.int32)
                sub_boxes = np.array(sub_boxes).astype(np.int32)
                obj_boxes = np.array(obj_boxes).astype(np.int32)
                sub_cls = np.array(sub_cls).astype(np.int32)
                obj_cls= np.array(obj_cls).astype(np.int32)

                h5f.create_dataset(im_id + '/sub_visual', dtype='float16', data=sub_visual)
                h5f.create_dataset(im_id + '/obj_visual', dtype='float16', data=obj_visual)
                h5f.create_dataset(im_id + '/joint_visual', dtype='float16', data=joint_visual)

                h5f.create_dataset(im_id + '/sub_classeme', dtype='float16', data=sub_classeme)
                h5f.create_dataset(im_id + '/obj_classeme', dtype='float16', data=obj_classeme)
                h5f.create_dataset(im_id + '/joint_classeme', dtype='float16', data=joint_classeme)

                h5f.create_dataset(im_id + '/pre_label', dtype='float16', data=pre_label)
                h5f.create_dataset(im_id + '/sub_boxes', dtype='float16', data=sub_boxes)
                h5f.create_dataset(im_id + '/obj_boxes', dtype='float16', data=obj_boxes)
                h5f.create_dataset(im_id + '/sub_cls', dtype='float16', data=sub_cls)
                h5f.create_dataset(im_id + '/obj_cls', dtype='float16', data=obj_cls)
                _t['misc'].toc()
                # rpn_rois = net.blobs['rois'].data
                # pool5 = net.blobs['pool5'].data
                # _t['misc'].tic()
                # cnt += 1
                print 'im_detect: {:d} {:.3f}s {:.3f}s' \
                    .format(cnt, _t['im_detect'].average_time,
                            _t['misc'].average_time)

prep_jointbox('test')

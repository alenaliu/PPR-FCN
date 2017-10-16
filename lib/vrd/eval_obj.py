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
import zl_config as C
import utils.zl_utils as zl

import scipy.io as sio
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

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        pass
        # Simply repeat the boxes, once for each class
    pred_boxes_rpn = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes,pred_boxes_rpn


def run_test_object_detection_eval():
    cfg_from_file('experiments/cfgs/rfcn_end2end.yml')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    h5f = h5py.File('output/vr_object_detections_rfcn.hdf5')
    m = h5py.File('data/sg_vrd_meta.h5', 'r', 'core')

    net = caffe.Net( 'models/sg_vrd/resnet50/test.prototxt',
                    'data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_64000.caffemodel',
                    caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    imdb = get_imdb('sg_vrd_2016_test')
    imdb.competition_mode(0)

    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = .1
    img_set_file = 'data/sg_vrd_2016/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}
    output = open('output/vr_object_detections.txt','w')
    for imid in imlist.keys():
        im_path = root  + imlist[imid] + '.jpg'
        cnt += 1
        im = cv2.imread(im_path)
        cv2.imshow('im',im)
        if im == None:
            print im_path
        _t['im_detect'].tic()
        scores,  boxes,boxes_rpn = im_detect(net, im, None)
        _t['im_detect'].toc()

        # scores = score_raw
        _t['misc'].tic()
        # h5f.create_dataset(imid + '/scores',dtype='float16', data=scores.astype(np.float16))
        # h5f.create_dataset(imid + '/boxes',dtype='short', data=boxes.astype(np.short))
        # scores=np.apply_along_axis(softmax,1,scores)
        # scores[:,16]+=icr
        boxes_tosort = []
        for j in xrange(1, 101):
            inds = np.where(scores[:, j] > 0.001)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, 4:8]
            #cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            # cls_boxes = boxes[inds]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .4, force_cpu=True)  # nms threshold
            # keep = nms_fast(cls_dets,.3)
            cls_dets = cls_dets[keep, :]
            boxes_tosort.append(cls_dets)
        for j in xrange(len(boxes_tosort)):
            cls_dets = boxes_tosort[j]
            for di in xrange(cls_dets.shape[0]):
                #    print 'here'
                di = cls_dets[di]
                score = di[-1]
                cls_idx = j + 1
                cls_name = zl.idx2name_cls(m,cls_idx)
                #cls_name = str(m['meta/cls/idx2name/' + str(cls_idx)][...])
                if score > 1:
                    score = 1
                if score < thresh:
                    continue
                cv2.rectangle(im,(di[0],di[1]),(di[2],di[3]),(255,0,0),2)
                x, y = int(di[0]), int(di[1])
                if x < 10:
                    x = 15
                if y < 10:
                    y = 15

                #cv2.putText(im,cls_name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                res_line = '%s %d %f %d %d %d %d'%(imid,cls_idx,score,di[0],di[1],di[2],di[3])
                output.write(res_line+'\n')
        #cv2.imshow('im',im)
        #if cv2.waitKey(0) == 27:
        #    exit(0)
        _t['misc'].toc()

        print 'im_detect: {:d} {:.3f}s {:.3f}s' \
            .format(cnt, _t['im_detect'].average_time,
                    _t['misc'].average_time)

def run_gen_recall():
    cfg_from_file('experiments/cfgs/rfcn_end2end.yml')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    h5f = h5py.File('output/vr_object_detections_rfcn.hdf5')
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')

    net = caffe.Net( 'models/sg_vrd/resnet50/test.prototxt',
                    #'output/rfcn_end2end/sg_vrd_2016_train/vrd_resnet50_rfcn_iter_500.caffemodel',
#'output/rfcn_end2end/sg_vrd_2016_train/vrd_resnet50_rfcn_iter_80000.caffemodel',
                    'data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_70000.caffemodel',
                    caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    imdb = get_imdb('sg_vrd_2016_test')
    imdb.competition_mode(0)

    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = 0.001
    img_set_file = 'data/sg_vrd_2016/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}


    mat_pred_label = []
    mat_pred_conf  = []
    mat_pred_bb = []
    mat_gt_label = []
    mat_gt_bb = []
    for i in xrange(1000):
        imid = str(m['db/testidx/'+str(i)][...])
        im_path = root  + imid + '.jpg'
        cnt += 1
        im = cv2.imread(im_path)
        cv2.imshow('im',im)
        if im == None:
            print im_path
        _t['im_detect'].tic()
        scores,  boxes,boxes_rpn = im_detect(net, im, None)
        _t['im_detect'].toc()

        # scores = score_raw
        _t['misc'].tic()
        # h5f.create_dataset(imid + '/scores',dtype='float16', data=scores.astype(np.float16))
        # h5f.create_dataset(imid + '/boxes',dtype='short', data=boxes.astype(np.short))
        # scores=np.apply_along_axis(softmax,1,scores)
        # scores[:,16]+=icr
        boxes_tosort = []
        for j in xrange(1, 101):
            inds = np.where(scores[:, j] > 0.00001)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, 4:8]
            #cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            # cls_boxes = boxes[inds]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold
            # keep = nms_fast(cls_dets,.3)
            cls_dets = cls_dets[keep, :]
            boxes_tosort.append(cls_dets)
        mat_pred_label_i = []
        mat_pred_conf_i = []
        mat_pred_bb_i = []
        for j in xrange(len(boxes_tosort)):
            cls_dets = boxes_tosort[j]
            for di in xrange(cls_dets.shape[0]):
                #    print 'here'
                di = cls_dets[di]
                score = di[-1]
                cls_idx = j + 1
                cls_name = zl.idx2name_cls(m,cls_idx)
                #cls_name = str(m['meta/cls/idx2name/' + str(cls_idx)][...])
                if score > 1:
                    score = 1
                if score < thresh:
                    continue
                cv2.rectangle(im,(di[0],di[1]),(di[2],di[3]),(255,0,0),2)
                x, y = int(di[0]), int(di[1])
                if x < 10:
                    x = 15
                if y < 10:
                    y = 15
                mat_pred_label_i.append(cls_idx)
                mat_pred_conf_i.append(score)
                mat_pred_bb_i.append([di[0],di[1],di[2],di[3]])
                cv2.putText(im,cls_name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                res_line = '%s %d %f %d %d %d %d'%(imid,cls_idx,score,di[0],di[1],di[2],di[3])
        mat_pred_label.append(mat_pred_label_i)
        mat_pred_conf.append(mat_pred_conf_i)
        mat_pred_bb.append(mat_pred_bb_i)
        obj_boxes = m['gt/test/%s/obj_boxes'%imid][...]
        sub_boxes = m['gt/test/%s/sub_boxes'%imid][...]
        rlp_labels = m['gt/test/%s/rlp_labels'%imid][...]
        mat_gt_label_i = []
        mat_gt_bb_i = []
        mat_gt_i = []
        for gti in xrange(obj_boxes.shape[0]):
            mat_gt_i.append([rlp_labels[gti,0],sub_boxes[gti,0],sub_boxes[gti,1],sub_boxes[gti,2],sub_boxes[gti,3]])
            mat_gt_i.append([rlp_labels[gti,2],obj_boxes[gti,0],obj_boxes[gti,1],obj_boxes[gti,2],obj_boxes[gti,3]])
        if len(mat_gt_i)>0:
            mat_gt_i = np.array(mat_gt_i)
            mat_gt_i=zl.unique_arr(mat_gt_i)
            for gti in xrange(mat_gt_i.shape[0]):
                mat_gt_bb_i.append(mat_gt_i[gti,1:])
                mat_gt_label_i.append(mat_gt_i[gti,0])
        mat_gt_label.append(mat_gt_label_i)
        mat_gt_bb.append(mat_gt_bb_i)
        #matlab_gt.append(matlab_gt_i)
        #now get gt

        # cv2.imshow('im',im)
        # if cv2.waitKey(0) == 27:
           # exit(0)
        _t['misc'].toc()

        print 'im_detect: {:d} {:.3f}s {:.3f}s' \
            .format(cnt, _t['im_detect'].average_time,
                    _t['misc'].average_time)

    sio.savemat('output/sg_vrd_objs.mat', {'pred_bb': mat_pred_bb,
                                           'pred_conf':mat_pred_conf,
                                           'pred_label':mat_pred_label,
                                           'gt_bb':mat_gt_bb,
                                           'gt_label':mat_gt_label
                                           })

def run_gen_recall_vg():
    def run_gen_recall():
        cfg_from_file('experiments/cfgs/rfcn_end2end.yml')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    h5f = h5py.File('output/vr_object_detections_rfcn.hdf5')
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')

    net = caffe.Net( 'models/sg_vrd/resnet50/test.prototxt',
                     #'output/rfcn_end2end/sg_vrd_2016_train/vrd_resnet50_rfcn_iter_500.caffemodel',
                     'output/rfcn_end2end/sg_vrd_2016_train/vrd_resnet50_rfcn_iter_80000.caffemodel',
                     #'data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_70000.caffemodel',
                     caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    imdb = get_imdb('sg_vrd_2016_test')
    imdb.competition_mode(0)

    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = 0.001
    img_set_file = 'data/sg_vrd_2016/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}


    mat_pred_label = []
    mat_pred_conf  = []
    mat_pred_bb = []
    mat_gt_label = []
    mat_gt_bb = []
    for i in xrange(1000):
        imid = str(m['db/testidx/'+str(i)][...])
        im_path = root  + imid + '.jpg'
        cnt += 1
        im = cv2.imread(im_path)
        cv2.imshow('im',im)
        if im == None:
            print im_path
        _t['im_detect'].tic()
        scores,  boxes,boxes_rpn = im_detect(net, im, None)
        _t['im_detect'].toc()

        # scores = score_raw
        _t['misc'].tic()
        # h5f.create_dataset(imid + '/scores',dtype='float16', data=scores.astype(np.float16))
        # h5f.create_dataset(imid + '/boxes',dtype='short', data=boxes.astype(np.short))
        # scores=np.apply_along_axis(softmax,1,scores)
        # scores[:,16]+=icr
        boxes_tosort = []
        for j in xrange(1, 101):
            inds = np.where(scores[:, j] > 0.00001)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, 4:8]
            #cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            # cls_boxes = boxes[inds]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold
            # keep = nms_fast(cls_dets,.3)
            cls_dets = cls_dets[keep, :]
            boxes_tosort.append(cls_dets)
        mat_pred_label_i = []
        mat_pred_conf_i = []
        mat_pred_bb_i = []
        for j in xrange(len(boxes_tosort)):
            cls_dets = boxes_tosort[j]
            for di in xrange(cls_dets.shape[0]):
                #    print 'here'
                di = cls_dets[di]
                score = di[-1]
                cls_idx = j + 1
                cls_name = zl.idx2name_cls(m,cls_idx)
                #cls_name = str(m['meta/cls/idx2name/' + str(cls_idx)][...])
                if score > 1:
                    score = 1
                if score < thresh:
                    continue
                cv2.rectangle(im,(di[0],di[1]),(di[2],di[3]),(255,0,0),2)
                x, y = int(di[0]), int(di[1])
                if x < 10:
                    x = 15
                if y < 10:
                    y = 15
                mat_pred_label_i.append(cls_idx)
                mat_pred_conf_i.append(score)
                mat_pred_bb_i.append([di[0],di[1],di[2],di[3]])
                cv2.putText(im,cls_name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                res_line = '%s %d %f %d %d %d %d'%(imid,cls_idx,score,di[0],di[1],di[2],di[3])
                #output.write(res_line+'\n')
        mat_pred_label.append(mat_pred_label_i)
        mat_pred_conf.append(mat_pred_conf_i)
        mat_pred_bb.append(mat_pred_bb_i)
        obj_boxes = m['gt/test/%s/obj_boxes'%imid][...]
        sub_boxes = m['gt/test/%s/sub_boxes'%imid][...]
        rlp_labels = m['gt/test/%s/rlp_labels'%imid][...]
        mat_gt_label_i = []
        mat_gt_bb_i = []
        mat_gt_i = []
        for gti in xrange(obj_boxes.shape[0]):
            mat_gt_i.append([rlp_labels[gti,0],sub_boxes[gti,0],sub_boxes[gti,1],sub_boxes[gti,2],sub_boxes[gti,3]])
            mat_gt_i.append([rlp_labels[gti,2],obj_boxes[gti,0],obj_boxes[gti,1],obj_boxes[gti,2],obj_boxes[gti,3]])
        if len(mat_gt_i)>0:
            mat_gt_i = np.array(mat_gt_i)
            mat_gt_i=zl.unique_arr(mat_gt_i)
            for gti in xrange(mat_gt_i.shape[0]):
                mat_gt_bb_i.append(mat_gt_i[gti,1:])
                mat_gt_label_i.append(mat_gt_i[gti,0])
        mat_gt_label.append(mat_gt_label_i)
        mat_gt_bb.append(mat_gt_bb_i)
        #matlab_gt.append(matlab_gt_i)
        #now get gt

        # cv2.imshow('im',im)
        # if cv2.waitKey(0) == 27:
        # exit(0)
        _t['misc'].toc()

        print 'im_detect: {:d} {:.3f}s {:.3f}s' \
            .format(cnt, _t['im_detect'].average_time,
                    _t['misc'].average_time)

    sio.savemat('output/sg_vrd_objs.mat', {'pred_bb': mat_pred_bb,
                                           'pred_conf':mat_pred_conf,
                                           'pred_label':mat_pred_label,
                                           'gt_bb':mat_gt_bb,
                                           'gt_label':mat_gt_label
                                           })

def run_gen_result():
    cfg_from_file('experiments/cfgs/rfcn_end2end.yml')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    h5f = h5py.File('output/precalc/sg_vrd_objs.hdf5')
    m = h5py.File('data/sg_vrd_meta.h5', 'r', 'core')
    net = caffe.Net( 'models/sg_vrd/resnet50/test.prototxt',
                     'data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_64000.caffemodel',
                     caffe.TEST)
    net.name = 'sgvrd'
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = 0.01
    for imid in m['gt/test/'].keys():
        im_path = root  + imid + '.jpg'
        cnt += 1
        im = cv2.imread(im_path)
        cv2.imshow('im',im)
        if im == None:
            print im_path
        _t['im_detect'].tic()
        scores,  boxes,boxes_rpn = im_detect(net, im, None)
        _t['im_detect'].toc()

        # scores = score_raw
        _t['misc'].tic()
        # h5f.create_dataset(imid + '/scores',dtype='float16', data=scores.astype(np.float16))
        # h5f.create_dataset(imid + '/boxes',dtype='short', data=boxes.astype(np.short))
        # scores=np.apply_along_axis(softmax,1,scores)
        # scores[:,16]+=icr
        boxes_tosort = []
        for j in xrange(1, 101):
            inds = np.where(scores[:, j] > 0.00001)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, 4:8]
            #cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            # cls_boxes = boxes[inds]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold
            # keep = nms_fast(cls_dets,.3)
            cls_dets = cls_dets[keep, :]
            boxes_tosort.append(cls_dets)
        h5_tosave = []
        for j in xrange(len(boxes_tosort)):
            cls_dets = boxes_tosort[j]
            for di in xrange(cls_dets.shape[0]):
                #    print 'here'
                di = cls_dets[di]
                score = di[-1]
                cls_idx = j + 1
                cls_name = zl.idx2name_cls(m,cls_idx)

                #cls_name = str(m['meta/cls/idx2name/' + str(cls_idx)][...])
                if score > 1:
                    score = 1
                if score < thresh:
                    continue

                dilist = di.tolist()
                dilist.append(cls_idx)
                h5_tosave.append(dilist)
                cv2.rectangle(im,(di[0],di[1]),(di[2],di[3]),(255,0,0),2)
                x, y = int(di[0]), int(di[1])
                if x < 10:
                    x = 15
                if y < 10:
                    y = 15
                cv2.putText(im,cls_name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                res_line = '%s %d %f %d %d %d %d'%(imid,cls_idx,score,di[0],di[1],di[2],di[3])
                #output.write(res_line+'\n')
        h5f['test/%s/boxes'%imid]=np.array(h5_tosave).astype(np.float16)
        #cv2.imshow('im',im)
        #if cv2.waitKey(0) == 27:
        #    exit(0)
        _t['misc'].toc()

        print 'im_detect: {:d} {:.3f}s {:.3f}s' \
            .format(cnt, _t['im_detect'].average_time,
                    _t['misc'].average_time)

def gen_obj_detection_results_from_hdf5_vr(h5_path):
    m = h5py.File('data/sg_vrd_meta.h5', 'r', 'core')
    h5f = h5py.File(h5_path,'r')
    thresh = 0.001
    cnt = 0
    zl.tic()

    mat_pred_label = []
    mat_pred_conf  = []
    mat_pred_bb = []
    mat_gt_label = []
    mat_gt_bb = []

    imid2path = {}
    impath2id = {}
    imidx2id = {}
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    img_set_file = 'data/sg_vrd_2016/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}
    #for k in m['gt/test/'].keys():
    for k in imlist.keys():
        imid  = str(m['db/testidx/%s'%k][...])
        #imid = imlist[k]
        if cnt %1000==0:
            print cnt,zl.toc()
            zl.tic()
        cnt += 1
        scores = h5f['%s/scores'%k][...]
        boxes = h5f['%s/boxes'%k][...]
        im = cv2.imread(root+imid+'.jpg')
        boxes_tosort = []
        mat_pred_label_i = []
        mat_pred_conf_i = []
        mat_pred_bb_i = []
        for j in xrange(1, 101):
            inds = np.where(scores[:, j] > 0.001)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold
            cls_dets = cls_dets[keep, :]
            boxes_tosort.append(cls_dets)
        for j in xrange(len(boxes_tosort)):
            cls_dets = boxes_tosort[j]
            for di in xrange(cls_dets.shape[0]):
                #    print 'here'
                di = cls_dets[di]
                score = di[-1]
                cls_idx = j + 1
                if score > 1:
                    score = 1
                if score < thresh:
                    continue

                mat_pred_label_i.append(cls_idx)
                mat_pred_conf_i.append(score)
                mat_pred_bb_i.append([di[0],di[1],di[2],di[3]])
        #for i in xrange(5):
        #    bb = np.array(mat_pred_bb_i[i]).astype(np.int)
        #    cv2.rectangle(im,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2)
        #cv2.imshow('im',im)
        #if cv2.waitKey(0)==27:
        #    exit(0)
        mat_pred_label.append(mat_pred_label_i)
        mat_pred_conf.append(mat_pred_conf_i)
        mat_pred_bb.append(mat_pred_bb_i)
        obj_boxes = m['gt/test/%s/obj_boxes'%imid][...]
        sub_boxes = m['gt/test/%s/sub_boxes'%imid][...]
        rlp_labels = m['gt/test/%s/rlp_labels'%imid][...]
        mat_gt_label_i = []
        mat_gt_bb_i = []
        mat_gt_i = []
        for gti in xrange(obj_boxes.shape[0]):
            mat_gt_i.append([rlp_labels[gti,0],sub_boxes[gti,0],sub_boxes[gti,1],sub_boxes[gti,2],sub_boxes[gti,3]])
            mat_gt_i.append([rlp_labels[gti,2],obj_boxes[gti,0],obj_boxes[gti,1],obj_boxes[gti,2],obj_boxes[gti,3]])
        if len(mat_gt_i)>0:
            mat_gt_i = np.array(mat_gt_i)
            mat_gt_i=zl.unique_arr(mat_gt_i)
            for gti in xrange(mat_gt_i.shape[0]):
                mat_gt_bb_i.append(mat_gt_i[gti,1:])
                mat_gt_label_i.append(mat_gt_i[gti,0])
        mat_gt_label.append(mat_gt_label_i)
        mat_gt_bb.append(mat_gt_bb_i)
                #outfile.write(res_line+'\n')

    sio.savemat('output/vg_vrd_objs.mat', {'pred_bb': mat_pred_bb,
                                           'pred_conf':mat_pred_conf,
                                           'pred_label':mat_pred_label,
                                           'gt_bb':mat_gt_bb,
                                           'gt_label':mat_gt_label
                                           })

def gen_obj_detection_results_from_hdf5_vgg(h5_path):
    m = h5py.File('data/vg1_2_meta.h5', 'r', 'core')
    h5f = h5py.File(h5_path,'r')
    thresh = 0.01
    cnt = 0
    zl.tic()

    mat_pred_label = []
    mat_pred_conf  = []
    mat_pred_bb = []
    mat_gt_label = []
    mat_gt_bb = []

    imid2path = {}
    impath2id = {}
    imidx2id = {}
    for i in m['gt/test/'].keys():
        imid2path[str(i)] = str(m['meta/imid2path/%s'%i][...])[:-4]

    for i in imid2path.keys():
        path = imid2path[i]
        impath2id[str(path)]=i
    img_set_file = 'data/vg1_2_2016/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}
    for imid in imlist.keys():
        path = imlist[imid]
        if path in impath2id:
            imidx2id[imid] = impath2id[path]
    #for k in m['gt/test/'].keys():
    for k in imlist.keys():
        if k not in imidx2id:
            continue
        imid  = imidx2id[k]
        if cnt %1000==0:
            print cnt,zl.toc()
            zl.tic()
        cnt += 1
        scores = h5f['%s/scores'%k][...]
        boxes = h5f['%s/boxes'%k][...]
        boxes_tosort = []
        mat_pred_label_i = []
        mat_pred_conf_i = []
        mat_pred_bb_i = []
        for j in xrange(1, 101):
            inds = np.where(scores[:, j] > 0.001)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold
            cls_dets = cls_dets[keep, :]
            boxes_tosort.append(cls_dets)
        for j in xrange(len(boxes_tosort)):
            cls_dets = boxes_tosort[j]
            for di in xrange(cls_dets.shape[0]):
                #    print 'here'
                di = cls_dets[di]
                score = di[-1]
                cls_idx = j + 1
                if score > 1:
                    score = 1
                if score < thresh:
                    continue
                res_line = '%s %d %f %d %d %d %d'%(k,cls_idx,score,di[0],di[1],di[2],di[3])

                mat_pred_label_i.append(cls_idx)
                mat_pred_conf_i.append(score)
                mat_pred_bb_i.append([di[0],di[1],di[2],di[3]])

        mat_pred_label.append(mat_pred_label_i)
        mat_pred_conf.append(mat_pred_conf_i)
        mat_pred_bb.append(mat_pred_bb_i)
        obj_boxes = m['gt/test/%s/obj_boxes'%imid][...]
        sub_boxes = m['gt/test/%s/sub_boxes'%imid][...]
        rlp_labels = m['gt/test/%s/rlp_labels'%imid][...]
        mat_gt_label_i = []
        mat_gt_bb_i = []
        mat_gt_i = []
        for gti in xrange(obj_boxes.shape[0]):
            mat_gt_i.append([rlp_labels[gti,0],sub_boxes[gti,0],sub_boxes[gti,1],sub_boxes[gti,2],sub_boxes[gti,3]])
            mat_gt_i.append([rlp_labels[gti,2],obj_boxes[gti,0],obj_boxes[gti,1],obj_boxes[gti,2],obj_boxes[gti,3]])
        if len(mat_gt_i)>0:
            mat_gt_i = np.array(mat_gt_i)
            mat_gt_i=zl.unique_arr(mat_gt_i)
            for gti in xrange(mat_gt_i.shape[0]):
                mat_gt_bb_i.append(mat_gt_i[gti,1:])
                mat_gt_label_i.append(mat_gt_i[gti,0])
        mat_gt_label.append(mat_gt_label_i)
        mat_gt_bb.append(mat_gt_bb_i)
                #outfile.write(res_line+'\n')

    sio.savemat('output/vg_vrd_objs.mat', {'pred_bb': mat_pred_bb,
                                           'pred_conf':mat_pred_conf,
                                           'pred_label':mat_pred_label,
                                           'gt_bb':mat_gt_bb,
                                           'gt_label':mat_gt_label
                                           })
    #outfile.close()
#run_test_remove_invalid_samples()
#run_test_object_detection_eval()
#gen_obj_detection_results_from_hdf5('output/vg_object_detections_100k.hdf5')
#gen_obj_detection_results_from_hdf5_vr('output/vr_object_detections.hdf5')
#run_test_object_detection_eval()
#run_gen_result()
#run_test_object_detection_eval()
#run_gen_recall()
#run_gen_result()


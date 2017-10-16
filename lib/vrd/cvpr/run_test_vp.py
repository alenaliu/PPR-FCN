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
        #im = cv2.resize(im_orig,(320,320))
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
    if False:
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
    fc7 = None#net.blobs['fc7'].data
    return None, scores, None, pred_boxes

def run_test_save_pool5():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')

    net = caffe.Net('models/sg_vrd/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/faster_rcnn_end2end/sg_vrd_2016_train/vgg16_faster_rcnn_finetune_iter_40000.caffemodel',
                    caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    imdb = get_imdb('sg_vrd_2016_test')
    imdb.competition_mode(0)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    h5path = 'output/' + imdb.name + '_pool5.hdf5'

    # if os.path.exists(h5path):
    # os.remove(h5path)
    h5f = h5py.File(h5path)
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
    thresh = .01
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt += 1
            if cnt %100==0:
                print cnt
            im_idx = name.split('.')[0]
            fpath = os.path.join(path, name)
            im = cv2.imread(fpath)
            if im == None:
                print fpath
            if im_idx + '/classemes' in h5f:
                continueG
            box_proposals = None
            _t['im_detect'].tic()
            score_raw, scores, fc7, boxes = im_detect(net, im, box_proposals)
            _t['im_detect'].toc()
            rpn_rois = net.blobs['rois'].data
            pool5 = net.blobs['pool5'].data
            # scores = score_raw
            res_rpn_rois = []
            res_pool5s = []
            res_locations = []
            res_visuals = []
            res_classemes = []
            res_cls_confs = []
            boxes_tosort = []
            _t['misc'].tic()
            for j in xrange(1, 101):
                inds = np.where(scores[:, j] > thresh)[0]
                cls_scores =scores[inds, j]
                cls_boxes = boxes[inds, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

                if len(cls_scores) <= 0:
                    boxes_tosort.append(cls_dets)
                    continue

                res_loc = cls_boxes
                res_vis = fc7[inds]
                res_classeme = scores[inds]
                res_cls_conf = np.column_stack((np.zeros(cls_scores.shape[0]) + j, cls_scores))
                res_pool5 = pool5[inds]
                res_rpn_roi = rpn_rois[inds]

                keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold
                cls_dets = cls_dets[keep, :]

                res_loc = res_loc[keep]
                res_vis = res_vis[keep]
                res_classeme = res_classeme[keep]
                res_cls_conf = res_cls_conf[keep]
                res_pool5 = res_pool5[keep]
                res_rpn_roi = res_rpn_roi[keep]

                res_classemes.extend(res_classeme)
                res_visuals.extend(res_vis)
                res_locations.extend(res_loc)
                res_cls_confs.extend(res_cls_conf)
                res_pool5s.extend(res_pool5)
                res_rpn_rois.extend(res_rpn_roi)

                boxes_tosort.append(cls_dets)
            # filter based on confidence
            inds = np.where(np.array(res_cls_confs)[:, 1] > 0.2)[0]

            res_classemes = np.array(res_classemes)[inds]
            res_visuals = np.array(res_visuals)[inds]
            res_locations = np.array(res_locations)[inds]
            res_cls_confs = np.array(res_cls_confs)[inds]

            res_pool5s = np.array(res_pool5s)[inds]
            res_rpn_rois = np.array(res_rpn_rois)[inds]

            h5f.create_dataset(im_idx + '/classemes', dtype='float16', data=res_classemes.astype(np.float16))
            h5f.create_dataset(im_idx + '/visuals', dtype='float16', data=res_visuals.astype(np.float16))
            h5f.create_dataset(im_idx + '/locations', dtype='short', data=res_locations.astype(np.short))
            h5f.create_dataset(im_idx + '/cls_confs', dtype='float16', data=res_cls_confs.astype(np.float16))
            h5f.create_dataset(im_idx + '/rpn_rois', dtype='float16', data=res_rpn_rois.astype(np.float16))
            h5f.create_dataset(im_idx + '/pool5s', dtype='float16', data=res_pool5s.astype(np.float16))
            _t['misc'].toc()
            cnt += 1
            print 'im_detect: {:d} {:.3f}s {:.3f}s' \
                .format(cnt, _t['im_detect'].average_time,
                        _t['misc'].average_time)

def run_test_save_raw():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net('models/sg_vrd/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/faster_rcnn_end2end/sg_vrd_2016_train/vgg16_faster_rcnn_finetune_iter_80000.caffemodel', caffe.TEST)
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
    h5path = 'output/' + imdb.name + '_conv.hdf5'

    # if os.path.exists(h5path):
    # os.remove(h5path)
    h5f = h5py.File(h5path)
    root = 'data/sg_vrd_2016/Data/sg_train_images/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    for path, subdirs, files in os.walk(root):
        for name in files:
            im_idx = name.split('.')[0]
            fpath = os.path.join(path, name)
            im = cv2.imread(fpath)
            if im == None:
                print fpath
            if im_idx in h5f:
                continue
            box_proposals = None
            _t['im_detect'].tic()
            score_raw, scores, fc7, boxes = im_detect(net, im, box_proposals)
            continue
            _t['im_detect'].toc()

            _t['misc'].tic()

            '''
            a = np.greater(score_raw,15)
            b = Gnp.any(a,axis=1)
            h5f.create_dataset(im_idx + '/scores',dtype='float16', data=score_raw[b,:].astype(np.float16))
            h5f.create_dataset(im_idx + '/boxes',dtype='short', data=boxes[b,:].astype(np.short))
            '''
            h5f.create_dataset(im_idx + '/scores_raw', dtype='float16', data=score_raw.astype(np.float16))
            h5f.create_dataset(im_idx + '/scores', dtype='float16', data=scores.astype(np.float16))
            h5f.create_dataset(im_idx + '/boxes', dtype='short', data=boxes.astype(np.short))
            h5f.create_dataset(im_idx + '/fc7', dtype='float16', data=fc7.astype(np.float16))

            _t['misc'].toc()
            cnt += 1
            print 'im_detect: {:d} {:.3f}s {:.3f}s' \
                .format(cnt, _t['im_detect'].average_time,
                        _t['misc'].average_time)

def nothing(x):
    pass

def convert_vp_result_for_matlab_eval2():
    _t = {'im_detect': Timer(), 'misc': Timer()}
    m_vp = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_vp_meta.h5', 'r', 'core')
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    h5path = 'output/precalc/sg_vrd_2016_test_new.hdf5'
    h5f = h5py.File(h5path,'r')
    h5path = 'output/precalc/sg_vrd_2016_test_nms.7.hdf5'
    h5f_nms = h5py.File(h5path)
    img_set_file = 'data/sg_vrd_2016/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}
    cnt = 1
    results = {}
    thresh = 0
    zl.tick()
    for imid in imlist.keys():
        cnt += 1
        if cnt %100==0:
            print cnt,zl.tock()
            zl.tick()
        imid_orig = imlist[imid]
        impath = imlist[imid] +'.jpg'
        impath = root+impath
        im = cv2.imread(impath)
        if im == None:
            print impath
        box_proposals = None

        _t['im_detect'].tic()
        scores,boxes = h5f[imid]['scores'][...],h5f[imid]['boxes'][...]
        _t['im_detect'].toc()
        _t['misc'].tic()
        boxes_tosort = []
        zl.tick()
        im_disp = im.copy()
        h5_boxes = []
        h5_labels = []
        h5_confs = []
        ind = np.argmax(scores[:,1:],axis=1)+1
        scores = scores[np.arange(scores.shape[0]),ind]
        dets  = np.hstack((boxes, scores[:,np.newaxis]))
        keep = nms(dets, .7,force_cpu=True)  # nms threshold
        dets = dets[keep,:]
        ind = ind[keep]
        h5f_nms.create_dataset(imid_orig+'/boxes',data = dets[:,:4])
        h5f_nms.create_dataset(imid_orig+'/confs',data = dets[:,4])
        h5f_nms.create_dataset(imid_orig+'/labels',data = ind)
        _t['misc'].toc()

def run_test_save_result():
    caffe.set_mode_gpu()
    caffe.set_device(0)

    m_vp = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_vp_meta.h5', 'r', 'core')
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    net = caffe.Net('models/sg_vrd_vp/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/faster_rcnn_end2end/sg_vrd_vp_2016_train/sg_vrd_vp_vgg16_faster_rcnn_finetune_iter_15000.caffemodel',caffe.TEST)
    h5path = 'output/precalc/sg_vrd_2016_test_new.hdf5'
    h5f = h5py.File(h5path)
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = .15
    img_set_file = 'data/sg_vrd_2016/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}

    # cv2.namedWindow('ctrl')
    # cv2.createTrackbar('thresh','ctrl',10,100,nothing)
    results = {}
    for imid in imlist.keys():
            cnt += 1
            if imid in h5f:continue
            impath = imlist[imid] +'.jpg'
            impath = root+impath
            im = cv2.imread(impath)
            if im == None:
                print impath
            box_proposals = None
            _t['im_detect'].tic()
            score_raw, scores, fc7, boxes = im_detect(net, im, box_proposals)
            _t['im_detect'].toc()
            boxes_tosort = []
            zl.tick()
            h5f.create_dataset(imid + '/scores', dtype='float16', data=scores.astype(np.float16))
            h5f.create_dataset(imid + '/boxes', dtype='short', data=boxes[:,:4].astype(np.short))
            t_misc = zl.tock()
            print 'im_detect: {:d} {:.3f}s {:.3f}s' \
                .format(cnt, _t['im_detect'].average_time,
                        t_misc)

def run_test_visualize():
    caffe.set_mode_gpu()
    caffe.set_device(0)

    m_vp = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_vp_meta.h5', 'r', 'core')
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5', 'r', 'core')
    net = caffe.Net('models/vg1_2_vp/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/faster_rcnn_end2end/vg1_2_vp2016_train/vg1_2_vp_vgg16_faster_rcnn_finetune_no_bbox_reg_iter_110000.caffemodel',caffe.TEST)
    root = 'data/vg1_2_2016/Data/test/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = .15
    img_set_file = 'data/vg1_2_2016/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}

    rel_types = {}
    rel_types['p']=[]
    rel_types['s']=[]
    rel_types['v']=[]
    rel_types['c']=[]

    for k in m_vp['meta/tri/name2idx'].keys():
        if k !='__background__':
            idx = int(str(m_vp['meta/tri/name2idx/'+k][...]))
            r_type = m_vp['meta/tri/name2idx/'+k].attrs['type']
            rel_types[r_type].append(idx)

    cv2.namedWindow('ctrl')
    cv2.createTrackbar('thresh','ctrl',10,100,nothing)
    results = {}
    for imid in imlist.keys():
            cnt += 1
            impath = imlist[imid] +'.jpg'
            if  '1059' not in impath and '107901' not in impath:continue
            impath = root+impath
            im = cv2.imread(impath)
            if im == None:
                print impath
            box_proposals = None
            _t['im_detect'].tic()
            score_raw, scores, fc7, raw_boxes = im_detect(net, im, box_proposals)
            _t['im_detect'].toc()
            boxes_tosort = []
            zl.tick()
            # boxes =np.array([])
            # labels =np.array([])
            boxes = None
            labels = None
            print 'generating boxes'
            for j in xrange(1, 19237):
                inds = np.where(scores[:, j] > 0.00)[0]
                cls_scores = scores[inds, j]
                cls_boxes = raw_boxes[inds, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                boxes_tosort.append(cls_dets)
                keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold

                cls_dets = cls_dets[keep, :]
                # sorted_ind = np.argsort(cls_dets[:,-1])[::-1]
                # cls_dets=cls_dets[sorted_ind]
                if cls_dets.shape[0]>0:
                    if boxes == None:
                        boxes = cls_dets
                    else:
                        boxes = np.vstack((boxes,cls_dets))
                    if labels == None:
                        labels = np.tile(j,cls_dets.shape[0])
                    else:
                        labels = np.hstack((labels,np.tile(j,cls_dets.shape[0])))
            # print boxes[:5]
            # print labels[:5]
            # exit(0)
            # sort the results
            print 'sorting'
            sorted_ind = np.argsort(boxes[:,-1])[::-1]
            boxes = boxes[sorted_ind]
            labels = labels[sorted_ind]

            ours_indices = {}
            ours_indices['p']=[]
            ours_indices['s']=[]
            ours_indices['v']=[]
            ours_indices['c']=[]
            indexor = np.arange(labels.shape[0])

            c_ind = np.in1d(labels,np.array(rel_types['c']))
            ours_indices['c'] = indexor[c_ind]
            p_ind = np.in1d(labels,np.array(rel_types['p']))
            ours_indices['p'] = indexor[p_ind]
            v_ind = np.in1d(labels,np.array(rel_types['v']))
            ours_indices['v'] = indexor[v_ind]
            s_ind = np.in1d(labels,np.array(rel_types['s']))
            ours_indices['s'] = indexor[s_ind]

            # exit(0)
            # for i in xrange(labels.shape[0]):

                # lbl =a labels[i]
                # if lbl in rel_types['p']: ours_indices['p'].append(i)
                # if lbl in rel_types['s']: ours_indices['s'].append(i)
                # if lbl in rel_types['v']: ours_indices['v'].append(i)
                # if lbl in rel_types['c']: ours_indices['c'].append(i)
            # print labels.shape[0]
            # print len(ours_indices['p'])
            # print len(ours_indices['s'])
            # print len(ours_indices['v'])
            # print len(ours_indices['c'])
            # print rel_types['c']
            # exit(0)
            _t['misc'].toc()
            t_misc = zl.tock()
            cv2.namedWindow('ctrl')
            cv2.destroyWindow('ctrl')
            cv2.namedWindow('ctrl')

            ours_p_len = ours_indices['p'].shape[0]-1
            ours_c_len = ours_indices['c'].shape[0]-1
            ours_v_len = ours_indices['v'].shape[0]-1
            ours_s_len = ours_indices['s'].shape[0]-1
            #ours_len = len(rlp_labels_ours)-1
            ours_len = labels.shape[0]-1

            if ours_len>0 :cv2.createTrackbar('idx_ours','ctrl',0,ours_len,nothing)
            if ours_p_len>0 :cv2.createTrackbar('idx_ours_p','ctrl',0,ours_p_len,nothing)
            if ours_c_len>0: cv2.createTrackbar('idx_ours_c','ctrl',0,ours_c_len,nothing)
            if ours_v_len>0:cv2.createTrackbar('idx_ours_v','ctrl',0, ours_v_len,nothing)
            if ours_s_len>0:cv2.createTrackbar('idx_ours_s','ctrl',0, ours_s_len,nothing)
            im_orig = im.copy()
            while True:

                if ours_len>=0:
                    idx_ours = cv2.getTrackbarPos('idx_ours','ctrl')
                    im_ours = im_orig.copy()
                    box = boxes[idx_ours]
                    lbl = zl.idx2name_tri(m_vp,labels[idx_ours])
                    cv2.putText(im_ours,lbl,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv2.rectangle(im_ours,(box[0],box[1]),(box[2],box[3]),(0,200,0),2)
                    cv2.imshow('im_ours',im_ours)

                if ours_c_len>=0:
                    idx_ours_c = cv2.getTrackbarPos('idx_ours_c','ctrl')
                    idx_ours = ours_indices['c'][idx_ours_c]
                    im_ours_c = im_orig.copy()
                    box = boxes[idx_ours]
                    lbl = zl.idx2name_tri(m_vp,labels[idx_ours])
                    cv2.putText(im_ours_c,lbl,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv2.rectangle(im_ours_c,(box[0],box[1]),(box[2],box[3]),(0,0,200),2)
                    cv2.imshow('im_ours_c',im_ours_c)
                if ours_s_len>=0:
                    idx_ours_s = cv2.getTrackbarPos('idx_ours_s','ctrl')
                    idx_ours = ours_indices['s'][idx_ours_s]
                    im_ours_s = im_orig.copy()
                    box = boxes[idx_ours]
                    lbl = zl.idx2name_tri(m_vp,labels[idx_ours])
                    cv2.putText(im_ours_s,lbl,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv2.rectangle(im_ours_s,(box[0],box[1]),(box[2],box[3]),(0,0,200),2)
                    cv2.imshow('im_ours_s',im_ours_s)
                if ours_v_len>=0:
                    idx_ours_v = cv2.getTrackbarPos('idx_ours_v','ctrl')
                    idx_ours = ours_indices['v'][idx_ours_v]
                    im_ours_v = im_orig.copy()
                    box = boxes[idx_ours]
                    lbl = zl.idx2name_tri(m_vp,labels[idx_ours])
                    cv2.putText(im_ours_v,lbl,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv2.rectangle(im_ours_v,(box[0],box[1]),(box[2],box[3]),(0,0,200),2)
                    cv2.imshow('im_ours_v',im_ours_v)
                if ours_p_len>=0:
                    idx_ours_p = cv2.getTrackbarPos('idx_ours_p','ctrl')
                    idx_ours = ours_indices['p'][idx_ours_p]
                    im_ours_p = im_orig.copy()
                    box = boxes[idx_ours]
                    lbl = zl.idx2name_tri(m_vp,labels[idx_ours])
                    cv2.putText(im_ours_p,lbl,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv2.rectangle(im_ours_p,(box[0],box[1]),(box[2],box[3]),(0,0,200),2)
                    cv2.imshow('im_ours_p',im_ours_p)

                c = cv2.waitKey(1)&0xFF
                if c == ord(' '):
                    break
                if c == ord('s'):
                    im_folder = 'output/results/examples/'+imid
                    if not os.path.exists('output/results/examples/'+imid):
                        os.makedirs('output/results/examples/'+imid)
                    if not os.path.exists('output/results/examples/'+imid+'/orig_'+imid+'.jpg'):
                        cv2.imwrite('output/results/examples/'+imid+'/orig_'+imid+'.jpg',im_orig)


                    if ours_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_'+imid+str(idx_ours)+'.jpg',im_ours)
                    if ours_v_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_v_'+imid+str(idx_ours)+'.jpg',im_ours_v)
                    if ours_p_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_p_'+imid+str(idx_ours)+'.jpg',im_ours_p)
                    if ours_c_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_c_'+imid+str(idx_ours)+'.jpg',im_ours_c)
                    if ours_s_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_s_'+imid+str(idx_ours)+'.jpg',im_ours_s)

            print 'im_detect: {:d} {:.3f}s {:.3f}s' \
                .format(cnt, _t['im_detect'].average_time,
                        t_misc)
def run_test_remove_invalid_samples():
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5','r','core')
    h5f = h5py.File('output/precalc/vg1_2_2016_test.hdf5')
    imids ={}
    for k in m['gt/test'].keys():
        imids[k]=0

    cnt = 0
    zl.tick()
    for k in h5f.keys():
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt+=1
        if k not in imids:
            del h5f[k]

def run_test_object_detection_eval():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    h5f = h5py.File('output/vg_object_detections_100k.hdf5')
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5', 'r', 'core')

    net = caffe.Net('models/vg1_2/vgg16/faster_rcnn_end2end/test.prototxt',
                    'output/models/vg1_2_vgg16_faster_rcnn_finetune_iter_120000.caffemodel',caffe.TEST)
    # net = caffe.Net('models/vg1_2/vgg16/faster_rcnn_end2end/test.prototxt',
                    # 'output/faster_rcnn_end2end/vg1_22016_train/vg1_2_vgg16_faster_rcnn_finetune_iter_50000.caffemodel',
                    # caffe.TEST)
    root = 'data/vg1_2_2016/Data/test/'
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = .05
    img_set_file = '/media/zawlin/ssd/data_vrd/vg_1.2/voc_format/ImageSets/test.txt'
    imlist = {line.strip().split(' ')[1]:line.strip().split(' ')[0] for line in open(img_set_file)}
    for imid in imlist.keys():
        im_path = root  + imlist[imid] + '.jpg'
        cnt += 1
        im = cv2.imread(im_path)
        if im == None:
            print im_path
        box_proposals = None
        _t['im_detect'].tic()
        score_raw, scores, fc7, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        # scores = score_raw
        _t['misc'].tic()
        h5f.create_dataset(imid + '/scores',dtype='float16', data=scores.astype(np.float16))
        h5f.create_dataset(imid + '/boxes',dtype='short', data=boxes.astype(np.short))

        _t['misc'].toc()

        print 'im_detect: {:d} {:.3f}s {:.3f}s' \
            .format(cnt, _t['im_detect'].average_time,
                    _t['misc'].average_time)

def gen_obj_detection_results_from_hdf5(h5_path,out_path):
    h5f = h5py.File(h5_path,'r')
    outfile = open(out_path,'w')
    thresh = 0.01
    cnt = 0
    zl.tick()
    for k in h5f.keys():
        if cnt %1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt += 1
        scores = h5f['%s/scores'%k][...]
        boxes = h5f['%s/boxes'%k][...]
        boxes_tosort = []
        for j in xrange(1, 201):
            inds = np.where(scores[:, j] > 0.001)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .2, force_cpu=False)  # nms threshold
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
                outfile.write(res_line+'\n')
    outfile.close()

def make_matlab_from_vp_nms():
    import scipy.io as sio
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    m_vp = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_vp_meta.h5')
    h5path = 'output/precalc/vg1_2_vp2016_test_nmsed.hdf5'
    h5f_nms = h5py.File(h5path)

    imids = sorted(m['gt/test'].keys())
    boxes_ours= np.empty((len(imids)),dtype=np.object)
    rlp_labels_ours = np.empty((len(imids)),dtype=np.object)
    rlp_confs_ours = np.empty((len(imids)),dtype=np.object)
    for i in xrange(boxes_ours.shape[0]):
        boxes_ours[i]=[]
        rlp_labels_ours[i]=[]
        rlp_confs_ours[i]=[]
    cnt = 1
    idx = 0
    for imid in imids:
        if cnt %100==0:
            print cnt
        cnt+=1
        if imid not in h5f_nms:
            rlp_labels_ours[idx]=[]
            rlp_confs_ours[idx]=[]
            boxes_ours[idx]=[]
            idx += 1
            continue
        boxes = h5f_nms[imid]['boxes'][...]
        confs = h5f_nms[imid]['confs'][...]
        labels = h5f_nms[imid]['labels'][...]

        ind = np.argsort(confs)[::-1]

        if ind.shape[0]>100:
            ind = ind[:100]

        boxes_raw = boxes[ind]
        confs = confs[ind]
        labels = labels[ind]
        boxes = []
        rlp_confs = []
        rlp_labels = []
        for i in xrange(confs.shape[0]):
            lbl = zl.idx2name_tri(m_vp,labels[i])
            sub_lbl = lbl.split('_')[0]
            pre_lbl = lbl.split('_')[1]
            obj_lbl = lbl.split('_')[2]
            rlp_label = [zl.name2idx_cls(m,sub_lbl),zl.name2idx_pre(m,pre_lbl),zl.name2idx_cls(m,obj_lbl)]
            rlp_labels.append(np.array(rlp_label).astype(np.float64))
            rlp_confs.append(np.array(confs[i]).astype(np.float64))
            boxes.append(np.array(boxes_raw[i]))
        # boxes = np.array(boxes).reshape((-1,4))
        # rlp_labels = np.array(rlp_labels).reshape((-1,3))
        # rlp_confs= np.array(rlp_confs)[:,np.newaxis]
        rlp_confs_ours[idx] = rlp_confs
        rlp_labels_ours[idx] = rlp_labels
        boxes_ours[idx] = boxes
        idx += 1
    # boxes_ours.append([])
    # boxes_ours=boxes_ours[:-1]
    sio.savemat('output/vg1_2_vp_results.mat', {'bboxes_ours': boxes_ours,
        'rlp_labels_ours':rlp_labels_ours,'rlp_confs_ours':rlp_confs_ours})#'relation_vectors':relation_vectors})


def make_matlab_from_vp_nms2():
    import scipy.io as sio
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5')
    m_vp = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_vp_meta.h5')
    h5path = 'output/precalc/sg_vrd_2016_test_nms.7.hdf5'
    h5f_nms = h5py.File(h5path)

    imids = sorted(m['gt/test'].keys())
    boxes_ours= np.empty((len(imids)),dtype=np.object)
    rlp_labels_ours = np.empty((len(imids)),dtype=np.object)
    rlp_confs_ours = np.empty((len(imids)),dtype=np.object)
    for i in xrange(boxes_ours.shape[0]):
        boxes_ours[i]=[]
        rlp_labels_ours[i]=[]
        rlp_confs_ours[i]=[]
    cnt = 1
    idx = 0
    for imid in imids:
        if cnt %100==0:
            print cnt
        cnt+=1
        if imid not in h5f_nms:
            rlp_labels_ours[idx]=[]
            rlp_confs_ours[idx]=[]
            boxes_ours[idx]=[]
            idx += 1
            continue
        boxes = h5f_nms[imid]['boxes'][...]
        confs = h5f_nms[imid]['confs'][...]
        labels = h5f_nms[imid]['labels'][...]
        ind = np.argsort(confs)[::-1]

        if ind.shape[0]>100:
            ind = ind[:100]

        boxes_raw = boxes[ind]
        confs = confs[ind]
        labels = labels[ind]
        boxes = []
        rlp_confs = []
        rlp_labels = []
        for i in xrange(confs.shape[0]):
            lbl = zl.idx2name_tri(m_vp,labels[i])
            sub_lbl = lbl.split('_')[0]
            pre_lbl = lbl.split('_')[1]
            obj_lbl = lbl.split('_')[2]
            rlp_label = [zl.name2idx_cls(m,sub_lbl),zl.name2idx_pre(m,pre_lbl),zl.name2idx_cls(m,obj_lbl)]
            rlp_labels.append(np.array(rlp_label).astype(np.float64))
            rlp_confs.append(np.array(confs[i]).astype(np.float64))
            boxes.append(np.array(boxes_raw[i]))
        # boxes = np.array(boxes).reshape((-1,4))
        # rlp_labels = np.array(rlp_labels).reshape((-1,3))
        # rlp_confs= np.array(rlp_confs)[:,np.newaxis]
        rlp_confs_ours[idx] = rlp_confs
        rlp_labels_ours[idx] = rlp_labels
        boxes_ours[idx] = boxes
        idx += 1
    # boxes_ours.append([])
    # boxes_ours=boxes_ours[:-1]
    sio.savemat('output/sg_vrd_vp_results_.7.mat', {'bboxes_ours': boxes_ours,
        'rlp_labels_ours':rlp_labels_ours,'rlp_confs_ours':rlp_confs_ours})#'relation_vectors':relation_vectors})

#run_test_remove_invalid_samples()
# run_test_object_detection_eval()
# gen_obj_detection_results_from_hdf5('output/vg_object_detections_100k.hdf5','output/vg_object_detections.txt')
# h5f_orig = h5py.File('output/precalc/vg1_2_2016_train_orig.hdf5')
# h5f = h5py.File('output/precalc/vg1_2_2016_train.hdf5')
# print len(h5f_orig.keys())
# print len(h5f.keys())
#run_test_save_result()
#run_test_visualize()
convert_vp_result_for_matlab_eval2()
make_matlab_from_vp_nms2()
#print np.hstack((np.tile(1,2),np.tile(3,2)))
#print np.vstack((np.array(),np.array([1,2,3,4]),np.array([4,5,6,7])))
# arr = np.array([1,2,3,4])
# c = np.array([2,3])
# indexor = np.arange(arr.shape[0])
# print np.in1d(arr,c)
# ind = indexor[np.in1d(arr,c)]
# print ind
# print np.any(arr == c)
# indices = numpy.arange(a.shape[0])[numpy.in1d(a, b)]
# np.in1d(np.array([1,2,3]), np.array([3,4]))

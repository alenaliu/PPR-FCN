import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import zl_config as C
from fast_rcnn.test import im_detect,im_detect_iccv
import matplotlib.pyplot as plt
from fast_rcnn.nms_wrapper import nms
import h5py
import cPickle
import os
from utils.timer import Timer
import utils.zl_utils as zl
import cv2
import dbox
import scipy.io as sio
import glog
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

db_net = caffe.Net('data/models/dbox/test.prototxt','data/models/dbox/fast-dbox-slidwindow-multiscale.caffemodel',caffe.TEST)
def combined_roidb(imdb_names):
    def get_roidb(imdb_name):

        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

def gen_recall():
    cfg_from_file('experiments/cfgs/rfcn_end2end_iccv_eb.yml')
    #cfg_from_file('experiments/cfgs/rfcn_end2end_iccv_eb.yml')

    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    h5objpath=  'output/precalc/sg_vrd_obj_rerank_wsd.h5'
    h5f_wsd = h5py.File(h5objpath,'a')
    h5path =  'data/sg_vrd_2016/EB/eb.h5'
    h5f = h5py.File(h5path,driver='core')
    h5_rois = {}

    for i in h5f['test/']:
        data=h5f['test/%s'%i][...].astype(np.float32)
        idx = np.argsort(data[:,-1],axis=0)
        data_sorted = data[idx][::-1]
        #data_sorted_idx = np.where((data_sorted[:,2]-data_sorted[:,0]>20) & (data_sorted[:,3]-data_sorted[:,1]>20))
        #data_sorted = data_sorted[data_sorted_idx]
        #print data_sorted
        h5_rois[i[:-4]] = data_sorted[:4000,:4]

    #cfg.TEST.HAS_RPN=False
    net =None
    #prototxt = 'models/pascal_voc/ResNet-50/rfcn_end2end/test_iccv_eb_sigmoid.prototxt'
    prototxt = 'models/sg_vrd/wsd/test_eb_wsddn_s.prototxt'
    #model = 'data/rfcn_models/resnet50_rfcn_iter_1200.caffemodel'
    root = 'data/sg_vrd_2016/Data/sg_test_images/'
    #model = 'output/rfcn_end2end/voc_0712_train/resnet50_rfcn_iter_16000.caffemodel'
    #model = 'output/rfcn_end2end/voc_0712_train/eb_wsddn_s_iter_5000.caffemodel'
    model = 'output/rfcn_end2end/sg_vrd_2016_train/eb_wsddn_s_iter_11000.caffemodel'

    #model = 'data/rfcn_models/resnet50_rfcn_final.caffemodel'
    #model = 'output/rfcn_end2end/voc_0712_train/resnet50_rfcn_eb_sigx_iter_100000.caffemodel'
    #model = 'data/rfcn_models_iccv/eb_resnet50_rfcn_iter_600.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, model, caffe.TEST)
    db_net = caffe.Net('data/models/dbox/test.prototxt','data/models/dbox/fast-dbox-slidwindow-multiscale.caffemodel',caffe.TEST)
    net.name = 'resnet50_rfcn_iter_1200'
    zl.tic()
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    max_per_image =5
    thresh = 0.00001
    cv2.namedWindow('im',0)
    cnt = 0
    for k in m['gt/test'].keys():
        im_path = root+k+'.jpg'
        im = cv2.imread(im_path)
        imid = k
        cnt+=1
        # filter out any ground truth boxes
        eb_roi = h5_rois[imid]
        db_scores = dbox.im_obj_detect(db_net,im,eb_roi)

        idx = np.argsort(db_scores[:,-1],axis=0)[::-1]
        eb_roi = eb_roi[idx[:100]]

        _t['im_detect'].tic()
        scores, boxes = im_detect_iccv(net, im, eb_roi)
        _t['im_detect'].toc()

        _t['misc'].tic()
        boxes_tosort = []
        h5_tosave = []
        for j in xrange(1, 101):
            inds = np.where(scores[:, j-1] > 0.00001)[0]
            cls_scores = scores[inds, j-1]
            cls_boxes = boxes[inds, 1:]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            cls_idx = np.zeros((cls_dets.shape[0],1))+j
            h5box = np.hstack((cls_boxes,cls_scores[:,np.newaxis],cls_idx))
            if h5box.shape[0]>0:
                h5_tosave.extend(h5box.tolist())
            keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold
            cls_dets = cls_dets[keep, :]
            boxes_tosort.append(cls_dets)
        h5_tosave = np.array(h5_tosave,dtype=np.float16)
        if h5_tosave.shape[0]>max_per_image:
            idx = np.argsort(h5_tosave[:,4],axis=0)[::-1]
            h5_tosave = h5_tosave[idx[:max_per_image]]
        for i in xrange(h5_tosave.shape[0]) :
            di=h5_tosave[i]
            x, y = int(di[0]), int(di[1])
            if x < 10:
                x = 15
            if y < 10:
                y = 15
            score = di[-2]
            if score > 1:
                score = 1
            if score < thresh:
                continue
            cls_idx = int(di[-1])
            cls_name = zl.idx2name_cls(m,cls_idx)
            cv2.rectangle(im,(di[0],di[1]),(di[2],di[3]),(255,0,0),2)
            cv2.putText(im,cls_name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)

        # 5 for vrd
        # 10 for vg

        #h5f_wsd.create_dataset('test/'+imid+'/boxes',dtype='float16', data=h5_tosave)
        cv2.imshow('im',im)
        if cv2.waitKey(0) == 27:
            exit(0)
        _t['misc'].toc()
        print 'im_detect: {:d} {:.3f}s {:.3f}s' \
            .format(cnt, _t['im_detect'].average_time,
                    _t['misc'].average_time)
#gen_recall()

def rerank_using_db():
    cfg_from_file('experiments/cfgs/rfcn_end2end_iccv_eb.yml')
    #cfg_from_file('experiments/cfgs/rfcn_end2end_iccv_eb.yml')

    caffe.set_mode_gpu()
    caffe.set_device(0)
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    h5objpath=  'output/precalc/sg_vrd_obj_rerank_wsd_db.h5'
    h5f_wsd = h5py.File(h5objpath,'a')
    h5path =  'data/sg_vrd_2016/EB/eb.h5'
    h5f = h5py.File(h5path,driver='core')
    h5_rois = {}

    for i in h5f['train/']:
        data=h5f['train/%s'%i][...].astype(np.float32)
        idx = np.argsort(data[:,-1],axis=0)
        data_sorted = data[idx][::-1]
        data_sorted_idx = np.where((data_sorted[:,2]-data_sorted[:,0]>20) & (data_sorted[:,3]-data_sorted[:,1]>20))
        data_sorted = data_sorted[data_sorted_idx]
        #print data_sorted
        h5_rois[i[:-4]] = data_sorted[:4000,:4]

    root = 'data/sg_vrd_2016/Data/sg_train_images/'
    db_net = caffe.Net('data/models/dbox/test.prototxt','data/models/dbox/fast-dbox-slidwindow-multiscale.caffemodel',caffe.TEST)
    cv2.namedWindow('im',0)
    cnt = 0
    for k in m['gt/train'].keys():
        im_path = root+k+'.jpg'
        im = cv2.imread(im_path)
        imid = k
        cnt+=1
        glog.info(cnt)
        # filter out any ground truth boxes
        eb_roi = h5_rois[imid]
        db_scores = dbox.im_obj_detect(db_net,im,eb_roi)

        idx = np.argsort(db_scores[:,-1],axis=0)[::-1]
        h5_tosave = np.concatenate((eb_roi[idx],db_scores[idx,1,np.newaxis]),axis=1)

        keep = nms(h5_tosave, .7, force_cpu=True)  # nms threshold
        h5_tosave=h5_tosave[keep]
        h5f_wsd.create_dataset('train/'+imid+'/boxes',dtype='float16', data=h5_tosave)
        #for i in xrange(100):
        #    di=h5_tosave[i]
        #    x, y = int(di[0]), int(di[1])
        #    if x < 10:
        #        x = 15
        #    if y < 10:
        #        y = 15
        #    cv2.rectangle(im,(di[0],di[1]),(di[2],di[3]),(255,0,0),2)
        #cv2.imshow('im',im)
        #if cv2.waitKey(0) == 27:
        #    exit(0)
gen_recall()
#rerank_using_db()

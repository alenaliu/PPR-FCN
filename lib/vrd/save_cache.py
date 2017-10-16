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
from fast_rcnn.test import im_detect
import matplotlib.pyplot as plt
from fast_rcnn.nms_wrapper import nms
import h5py
import cv2
from utils.blob import prep_im_for_blob,im_list_to_blob
import utils.zl_utils as zl
import os
import glog
from fast_rcnn.test import im_detect as im_detect_orig

def vis_square(data, index):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    #dbg
    # normalize data for display, from 1 ~ 0
    data = (data - data.min()) / (data.max() - data.min())
    copydata = data

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)

    # pad with ones (white)
    data = np.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data
    # plt.imshow(data)
    # plt.show()
    # plt.axis('off')
def _get_image_blob(im_path):
    im = cv2.imread(im_path)
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


def im_detect(net, im_path):
    blobs = {'data' : None}
    blobs['data'], im_scales = _get_image_blob(im_path)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
    blobs_out = net.forward(**blobs)

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  x2 = max(a[2],b[2])
  y2 = max(a[3],b[3])
  return (int(x), int(y), int(x2), int(y2))

def save_train():
    prototxt = 'models/sg_vrd/resnet50/test_cache_cls.prototxt'
    model = 'data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_70000.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, model, caffe.TEST)

    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r')
    cache_hdf5= h5py.File('output/precalc/sg_vrd_cache_vgg.h5', 'a')
    cnt = 0
    layer_name = 'rfcn_cls'
    for k in m['gt/train'].keys():
        cnt +=1
        glog.info(cnt)

        im_path = C.get_sg_vrd_path_train(k)
        im_detect(net,im_path)
        data = net.blobs[layer_name].data[...].astype(np.float16)

        rfcn_cls_reduced = np.zeros((data.shape[0],101,data.shape[2],data.shape[3]),np.float16)
        for i in xrange(0,101):
            rfcn_cls_reduced[:,i,:,:] = np.average(data[:,i*49:i*49+49,:,:],axis=1)
        cache_hdf5.create_dataset('train/%s'%k + '/rfcn_cls_reduced', dtype='float16', data=rfcn_cls_reduced)

# def save_test():
    # prototxt = 'models/sg_vrd/resnet50/test_cache.prototxt'
    # model = 'data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_64000.caffemodel'
    # caffe.set_mode_gpu()
    # caffe.set_device(0)
    # net = caffe.Net(prototxt, model, caffe.TEST)

    # m = h5py.File('data/sg_vrd_meta.h5', 'r')
    # cache_hdf5= h5py.File('output/sg_vrd_cache.h5', 'a')
    # cnt = 0
    # for k in m['gt/test'].keys():
        # cnt +=1
        # glog.info(cnt)

        # im_path = C.get_sg_vrd_path_test(k)
        # im_detect(net,im_path)
        # data = net.blobs['conv_new_1'].data[...].astype(np.float16)
        # cache_hdf5.create_dataset('test/%s'%k + '/conv_new_1', dtype='float16', data=data)

def save_train_rfcn_cls_reduced():
    m = h5py.File('data/sg_vrd_meta.h5', 'r')
    cache_hdf5= h5py.File('output/precalc/sg_vrd_cache_vgg.h5', 'a')
    cnt = 0
    layer_name = 'rfcn_cls_reduced1'
    for k in m['gt/train'].keys():
        cnt +=1
        glog.info(cnt)

        data = cache_hdf5['train/%s'%k+'/rfcn_cls'][...]

        rfcn_cls_reduced = np.zeros((data.shape[0],101,data.shape[2],data.shape[3]),np.float16)
        for i in xrange(0,101):
            rfcn_cls_reduced[:,i,:,:] = np.average(data[:,i*49:i*49+49,:,:],axis=1)
        cache_hdf5['train/%s'%k + '/'+layer_name]=rfcn_cls_reduced.astype(np.float16)

def save_train_vgg():
    prototxt = 'data/models/vgg/test.prototxt'
    model = 'data/models/vgg/vgg16_faster_rcnn_finetune_iter_40000.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, model, caffe.TEST)
    cfg.TEST.HAS_RPN=True
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r')
    cache_hdf5= h5py.File('output/precalc/sg_vrd_cache_vgg.h5', 'a')
    cnt = 0
    layer_name = 'conv5_3'
    thresh = 0.2
    for k in m['gt/test'].keys():
        cnt +=1
        glog.info(cnt)
        im_path = C.get_sg_vrd_path_test(k)
        im = cv2.imread(im_path)
        scores,boxes = im_detect_orig(net,im)
        #visualize(im,scores,boxes)
        data = net.blobs[layer_name].data[...].astype(np.float16)
        cache_hdf5.create_dataset('test/%s'%k + '/%s'%layer_name, dtype='float16', data=data)
def visualize(im,scores,boxes):
    thresh = 0.2
    boxes_tosort = []
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
            cv2.rectangle(im,(di[0],di[1]),(di[2],di[3]),(255,0,0),2)
    #for i in xrange(5):
    #    bb = np.array(mat_pred_bb_i[i]).astype(np.int)
    #    cv2.rectangle(im,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2)
    cv2.imshow('im',im)
    if cv2.waitKey(0)==27:
        exit(0)
def save_train_resnet():
    prototxt = 'models/sg_vrd/resnet50/test_cache.prototxt'
    model = 'data/imagenet/ResNet-50-model.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(prototxt, model, caffe.TEST)
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r')
    cache_hdf5= h5py.File('output/precalc/sg_vrd_cache_resnet.h5', 'a')
    cnt = 0
    layer_name = 'res5c'
    for k in m['gt/test'].keys():
        cnt +=1
        glog.info(cnt)
        im_path = C.get_sg_vrd_path_test(k)
        im_detect(net,im_path)
        data = net.blobs[layer_name].data[...].astype(np.float16)
        cache_hdf5.create_dataset('test/%s'%k + '/%s'%layer_name, dtype='float16', data=data)
def save_test():
    prototxt = 'models/sg_vrd/resnet50/test_cache.prototxt'
    model = 'data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_64000.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, model, caffe.TEST)

    m = h5py.File('data/sg_vrd_meta.h5', 'r')
    cache_hdf5= h5py.File('output/sg_vrd_cache.h5', 'a')
    cnt = 0
    for k in m['gt/train'].keys():
        cnt +=1
        glog.info(cnt)

        im_path = C.get_sg_vrd_path_train(k)
        im_detect(net,im_path)
        data = net.blobs['conv_new_1'].data[...].astype(np.float16)
        cache_hdf5.create_dataset('train/%s'%k + '/conv_new_1', dtype='float16', data=data)
def save_train_vg():
    prototxt = 'models/vg1_2/resnet50/test_cache.prototxt'
    model = 'data/models/vrd_rfcn/vg_resnet50_rfcn_iter_112000.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, model, caffe.TEST)

    m = h5py.File('data/vg1_2_meta.h5', 'r')
    cache_hdf5= h5py.File('output/vg_cache.h5', 'a')
    cnt = 0
    for k in m['gt/train'].keys():
        cnt +=1
        glog.info(cnt)

        path = zl.imid2path(m,k)
        im_path = C.get_vg_path_train(path)
        im_detect(net,im_path)
        data = net.blobs['conv_new_1'].data[...].astype(np.float16)
        cache_hdf5.create_dataset('train/%s'%k + '/conv_new_1', dtype='float16', data=data)
#save_train_rfcn_cls_reduced()
#save_train_vgg()
#save_train_resnet()
#save_test()
#save_test()
save_train_vg()

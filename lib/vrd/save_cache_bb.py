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


def im_detect(net, im_path, sub_boxes,obj_boxes):
    blobs = {'data' : None}
    data, im_scales = _get_image_blob(im_path)

    if sub_boxes.shape[0]>0:
        zeros = np.zeros((sub_boxes.shape[0],1), dtype=np.float)
        # first index is always zero since we do one image by one image
        sub_boxes = np.concatenate((zeros, sub_boxes),axis=1)
        obj_boxes = np.concatenate((zeros, obj_boxes),axis=1)
    else:
        glog.info('here')
        pass
    sub_boxes = sub_boxes * im_scales[0]
    obj_boxes = obj_boxes * im_scales[0]

    # reshape network inputs
    net.blobs['data'].reshape(*data.shape)
    net.blobs['sub_boxes'].reshape(sub_boxes.shape[0],5,1,1)
    net.blobs['obj_boxes'].reshape(obj_boxes.shape[0],5,1,1)

    forward_kwargs = {'data': data.astype(np.float32, copy=False),
                      'sub_boxes': sub_boxes.reshape((-1,5,1,1)),
                      'obj_boxes': obj_boxes.reshape((-1,5,1,1))}

    blobs_out = net.forward(**forward_kwargs)

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
    cache_hdf5= h5py.File('output/sg_vrd_cache.h5', 'a')
    cnt = 0
    layer_name = 'rfcn_cls'
    for k in m['gt/train'].keys():
        cnt +=1
        glog.info(cnt)

        s_classeme = []
        o_classeme = []
        pre_label = []
        im_path = C.get_sg_vrd_path_train(k)
        sub_boxes = m['gt/train/%s/sub_boxes'%k][...]
        obj_boxes = m['gt/train/%s/obj_boxes'%k][...]
        rlp_labels = m['gt/train/%s/rlp_labels'%k][...]
        if sub_boxes.shape[0]>0:
            #im_detect(net,im_path,sub_boxes,obj_boxes)
            #s_classeme = net.blobs['s_classeme'].data[...].astype(np.float16)
            #o_classeme = net.blobs['s_classeme'].data[...].astype(np.float16)
            pre_label = rlp_labels[:,1]
        #data = net.blobs[layer_name].data[...].astype(np.float16)

        #rfcn_cls_reduced = np.zeros((data.shape[0],101,data.shape[2],data.shape[3]),np.float16)
        #for i in xrange(0,101):
        #    rfcn_cls_reduced[:,i,:,:] = np.average(data[:,i*49:i*49+49,:,:],axis=1)
        #cache_hdf5.create_dataset('train/%s'%k + '/s_classeme', dtype='float16', data=s_classeme)
        #cache_hdf5.create_dataset('train/%s'%k + '/o_classeme', dtype='float16', data=o_classeme)
        cache_hdf5.create_dataset('train/%s'%k + '/pre_label', dtype='float16', data=pre_label)
#save_train_rfcn_cls_reduced()
save_train()
#save_test()
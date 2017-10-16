# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
import numpy as np
import yaml
#from multiprocessing import Process, Queue
from utils.blob import prep_im_for_blob,im_list_to_blob
import pdb
import zl_config as C
import glog
import h5py
import os
import utils.zl_utils as zl
import random
import cv2
def _get_image_blob(im_path, scale_inds):
    processed_ims = []
    im_scales = []
    im = cv2.imread(im_path)
    target_size = cfg.TRAIN.SCALES[scale_inds[0]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, im_scales

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    x2 = max(a[2],b[2])
    y2 = max(a[3],b[3])
    return (int(x), int(y), int(x2), int(y2))

def union_np(a,b):
    x = np.minimum(a[:,1], b[:,1])
    y = np.minimum(a[:,2], b[:,2])
    x2 = np.maximum(a[:,3],b[:,3])
    y2 = np.maximum(a[:,4],b[:,4])
    return np.concatenate((a[:,0,np.newaxis],x[:,np.newaxis],y[:,np.newaxis],x2[:,np.newaxis],y2[:,np.newaxis]),axis=1)

class ICCVDataLayerJointbox(caffe.Layer):
    """Fast R-CNN data layer used for training."""
    def get_minibatch(self):
        blobs = {}
        if self.imidx >=len(self.imids):
            random.shuffle(self.imids)
            self.imidx = 0
        imid = self.imids[self.imidx]
        self.imidx += 1

        gt = self.gt_labels[imid]
        if gt['sub_boxes'].shape[0]<=0:
            return self.get_minibatch()
        impath = C.get_sg_vrd_path_train(imid)
        #glog.info(impath)
        random_scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES),
                                              size=1)
        im_blob, im_scales = _get_image_blob(impath, random_scale_inds)
        #glog.info(im_scales)
        # blobs['data']=im_blob

        #rfcn_cls_reduced = np.zeros((rfcn_cls.shape[0],100,rfcn_cls.shape[2],rfcn_cls.shape[3]),np.float32)
        #for i in xrange(1,101):
        #    rfcn_cls_reduced[:,i-1,:,:] = np.average(rfcn_cls[:,i*49:i*49+49,:,:],axis=1).astype(np.float32)

        sub_boxes = gt['sub_boxes']*im_scales[0]
        obj_boxes = gt['obj_boxes']*im_scales[0]
        boxes = union_np(sub_boxes,obj_boxes)
        blobs['data'] = im_blob#self.cache['train/%s/conv_new_1'%imid][...]
        blobs['boxes'] =boxes
        blobs['labels'] = gt['rlp_labels'][:,1]
        return blobs

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        self._cur_idx = 0
        self.gt_labels = {}
        self.meta = h5py.File('data/sg_vrd_meta.h5', 'r')
        self.cache = h5py.File('output/sg_vrd_cache.h5','r')
        if os.path.exists('output/cache/sg_vrd_gt.pkl'):
            self.gt_labels = zl.load('output/cache/sg_vrd_gt.pkl')
            glog.info('loaded gt data from cache')
        else:
            glog.info( 'Preloading gt')
            zl.tic()
            for k in self.meta['gt/train'].keys():
                rlp_labels = self.meta['gt/train/%s/rlp_labels'%k][...]
                sub_boxes = self.meta['gt/train/%s/sub_boxes'%k][...].astype(np.float32)
                obj_boxes = self.meta['gt/train/%s/obj_boxes'%k][...].astype(np.float32)
                if sub_boxes.shape[0]>0:
                    zeros = np.zeros((sub_boxes.shape[0],1), dtype=np.float)
                    # first index is always zero since we do one image by one image
                    sub_boxes = np.concatenate((zeros, sub_boxes),axis=1)
                    obj_boxes = np.concatenate((zeros, obj_boxes),axis=1)
                self.gt_labels[k] = {}
                self.gt_labels[k]['rlp_labels']=rlp_labels
                self.gt_labels[k]['sub_boxes']=sub_boxes
                self.gt_labels[k]['obj_boxes']=obj_boxes
            glog.info('done preloading gt %f'%zl.toc())
            zl.save('output/cache/sg_vrd_gt.pkl',self.gt_labels)

        self.imids = []
        for k in self.gt_labels.keys():
            self.imids.append(k)
        self.imidx =0
        random.shuffle(self.imids)
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
                         50,50)
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(1, 5, 1, 1)
        self._name_to_top_map['boxes'] = idx
        idx += 1

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[idx].reshape(1, 1, 1, 1)
        self._name_to_top_map['labels'] = idx


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # todo: modify mini_batch
        blobs = self.get_minibatch()

        labels = blobs['labels']
        boxes = blobs['boxes']

        top[0].reshape(*blobs['data'].shape)
        top[0].data[...] = blobs['data']

        top[1].reshape(boxes.shape[0],5,1,1)
        top[1].data[...] = boxes.reshape((-1,5,1,1))
        top[2].reshape(labels.shape[0],1)
        top[2].data[...] = labels.reshape((-1,1))


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
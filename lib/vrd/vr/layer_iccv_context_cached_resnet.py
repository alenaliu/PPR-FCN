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
    return blob, im_scales,im.shape
def add_jitter(boxes,im_shape):
    ret_boxes = []
    for i in xrange(boxes.shape[0]):
        b = boxes[i]
        w = b[3]-b[1]
        h = b[4]-b[2]
        var = 0.1
        x1off = w*np.random.uniform(-var,var)
        x2off = w*np.random.uniform(-var,var)
        y1off = h*np.random.uniform(-var,var)
        y2off = h*np.random.uniform(-var,var)
        x1 = b[1]+x1off
        x2 = b[3]+x2off
        y1 = b[2]+y1off
        y2 = b[4]+y2off
        bjit = [0,x1,y1,x2,y2]
        ret_boxes.append(bjit)

    return np.array(ret_boxes)

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

class ICCVDataLayerContextCached(caffe.Layer):
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
        im_blob, im_scales,im_shape = _get_image_blob(impath, random_scale_inds)
        #glog.info(im_scales)
        # blobs['data']=im_blob
        if imid not in self.cache_mem:
            self.cache_mem[imid] = self.cache['train/%s/res5c'%imid][...]
        blobs['res5c']=self.cache_mem[imid]
        sub_boxes = gt['sub_boxes']*im_scales[0]
        obj_boxes = gt['obj_boxes']*im_scales[0]

        union_boxes = union_np(sub_boxes,obj_boxes)
        #sub_boxes = add_jitter(sub_boxes,im_shape)
        #obj_boxes = add_jitter(obj_boxes,im_shape)
        blobs['sub_boxes'] = sub_boxes
        blobs['obj_boxes'] = obj_boxes
        blobs['union_boxes'] = union_boxes
        blobs['labels'] = gt['rlp_labels'][:,1]
        #blobs['rlp_labels'] = gt['rlp_labels']
        return blobs

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        self._cur_idx = 0
        self.gt_labels = {}
        self.meta = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r')
        self.cache = h5py.File('output/precalc/sg_vrd_cache_resnet.h5','r')
        self.cache_mem = {}
        if os.path.exists('output/cache/sg_vrd_gt.pkl'):
            self.gt_labels = zl.load('output/cache/sg_vrd_gt.pkl')
            glog.info('loaded gt data from cache')
        else:
            glog.info( 'Preloading gt')
            zl.tic()
            for k in self.meta['gt/train'].keys():
                rlp_labels = self.meta['gt/train/%s/rlp_labels'%k][...]
                sub_boxes = self.meta['gt/train/%s/sub_boxes'%k][...].astype(np.float)
                obj_boxes = self.meta['gt/train/%s/obj_boxes'%k][...].astype(np.float)
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
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 2048,
                         50,50)
        self._name_to_top_map['res5c'] = idx
        idx += 1

        top[idx].reshape(1, 5, 1, 1)
        self._name_to_top_map['sub_boxes'] = idx
        idx += 1

        top[idx].reshape(1, 5, 1, 1)
        self._name_to_top_map['obj_boxes'] = idx
        idx += 1
        top[idx].reshape(1, 5, 1, 1)
        self._name_to_top_map['union_boxes'] = idx
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
        sub_boxes = blobs['sub_boxes']
        obj_boxes = blobs['obj_boxes']
        union_boxes = blobs['union_boxes']
        #rlp_labels = blobs['rlp_labels']

        top[0].reshape(*blobs['res5c'].shape)
        top[0].data[...] = blobs['res5c']

        top[1].reshape(sub_boxes.shape[0],5,1,1)
        top[1].data[...] = sub_boxes.reshape((-1,5,1,1))
        top[2].reshape(obj_boxes.shape[0],5,1,1)
        top[2].data[...] = obj_boxes.reshape((-1,5,1,1))
        top[3].reshape(union_boxes.shape[0],5,1,1)
        top[3].data[...] = union_boxes.reshape((-1,5,1,1))
        top[4].reshape(labels.shape[0])
        top[4].data[...] = labels.astype(np.float32)
        #top[4].reshape(*rlp_labels.shape)
        #top[4].data[...] =rlp_labels
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

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

class ICCVDataLayer(caffe.Layer):
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
        blobs['data']=im_blob
        blobs['sub_boxes'] = gt['sub_boxes']*im_scales[0]
        blobs['obj_boxes'] = gt['obj_boxes']*im_scales[0]
        blobs['labels'] = gt['rlp_labels'][:,1]
        blobs['rlp_labels'] = gt['rlp_labels']
        return blobs

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        self._cur_idx = 0
        self.gt_labels = {}
        self.meta = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r')
        if os.path.exists('output/cache/sg_vrd_gt.pkl'):
            self.gt_labels = zl.load('output/cache/sg_vrd_gt.pkl')
            glog.info('loaded gt data from cache')
        else:
            glog.info( 'Preloading gt')
            zl.tick()
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
            glog.info('done preloading gt %f'%zl.tock())
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
                         max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(1, 5, 1, 1)
        self._name_to_top_map['sub_boxes'] = idx
        idx += 1

        top[idx].reshape(1, 5, 1, 1)
        self._name_to_top_map['obj_boxes'] = idx
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
        rlp_labels = blobs['rlp_labels']

        top[0].reshape(*blobs['data'].shape)
        top[0].data[...] = blobs['data']

        top[1].reshape(sub_boxes.shape[0],5,1,1)
        top[1].data[...] = sub_boxes.reshape((-1,5,1,1))
        top[2].reshape(obj_boxes.shape[0],5,1,1)
        top[2].data[...] = obj_boxes.reshape((-1,5,1,1))
        top[3].reshape(labels.shape[0],1)
        top[3].data[...] = labels.reshape((-1,1))

        top[4].reshape(*rlp_labels.shape)
        top[4].data[...] =rlp_labels
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

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
from vrd.vr.wsd.mini_batch import get_minibatch
import numpy as np
import yaml
#from multiprocessing import Process, Queue
import pdb
import zl_config as C
import glog
class RoIDataLayerEbBinaryLogLoss(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

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

        ##todo necessary?
        #top[idx].reshape(1, 3)
        #self._name_to_top_map['im_info'] = idx
        #idx += 1
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[idx].reshape(1, 5, 1, 1)
        self._name_to_top_map['gt_boxes'] = idx
        idx += 1

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[idx].reshape(1, 1, 1, 1)
        self._name_to_top_map['labels'] = idx
        idx += 1

        top[idx].reshape(1, 100, 1, 1)
        self._name_to_top_map['labels_v'] = idx
        idx += 1
        #top[idx].reshape(1, 1000, 1, 1)
        #self._name_to_top_map['word_embeddings'] = idx
        #idx += 1

        #top[idx].reshape(1, 20, 1, 1)
        #self._name_to_top_map['regression_targets'] = idx
        #idx += 1

        #print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        #assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # todo: modify mini_batch
        blobs = self._get_next_minibatch()
        labels = blobs['labels']
        gt_boxes = blobs['gt_boxes']
        im_info = blobs['im_info']
        #gt_boxes = gt_boxes
        #blobs['labels']=blobs['gt_boxes']
        #blobs['word_embeddings'] = np.zeros((1,1,1,1),np.float32)
        #blobs['regression_targets'] = np.zeros((1,20,1,1),np.float32)

        #for blob_name, blob in blobs.iteritems():
        #    top_ind = self._name_to_top_map[blob_name]
        #    shape = blob.shape
        #    top[top_ind].reshape(*(blob.shape))
        #    # Copy data into net's input blobs
        #    top[top_ind].data[...] = blob.astype(np.float32, copy=False)

        top[0].reshape(*blobs['data'].shape)
        top[0].data[...] = blobs['data']

        gt_boxes = gt_boxes.reshape((gt_boxes.shape[0],gt_boxes.shape[1],1,1))
        zeros = np.zeros((gt_boxes.shape[0],1,1, 1), dtype=gt_boxes.dtype)
        all_rois = np.concatenate((zeros, gt_boxes),axis=1)

        top[1].reshape(*all_rois.shape)
        top[1].data[...] = all_rois
        #labels = np.concatenate((labels,np.array([0,0,0])))
        labels = np.unique(labels)
        labels_v = np.zeros((100))

        labels_v[...]=-1
        labels_v[labels.astype(np.int32)-1]=1
        #labels_v = np.tile(labels_v,labels.shape[0])

        labels_v = labels_v.reshape((1,100))
        #for i in xrange(labels.shape[0]):
        #    labels_v[i,labels[i]]=1
        labels = labels.reshape((labels.shape[0],1,1,1))
        #labels[-1,0,0,0]=0
        top[2].reshape(*labels.shape)
        top[2].data[...] = labels

        top[3].reshape(*labels_v.shape)
        top[3].data[...] =labels_v# labels

        #embeddings = np.zeros((gt_boxes.shape[0],1,1, 1), dtype=gt_boxes.dtype)
        #top[3].reshape(*embeddings.shape)
        #top[3].data[...] = embeddings

        #regression_targets = np.zeros((gt_boxes.shape[0],20,1, 1), dtype=gt_boxes.dtype)
        #top[4].reshape(*regression_targets.shape)
        #top[4].data[...] = regression_targets
        #glog.info('here')
        #pdb.set_trace()

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
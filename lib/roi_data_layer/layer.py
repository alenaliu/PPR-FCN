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
import scipy.io as sio
import os
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch

import numpy as np

import yaml
from multiprocessing import Process, Queue
import random
DEV_KIT_PATH='data/ilsvrc2016/devkit/'
vid_perf={'airplane':	0.782, 'antelope':	0.702, 'bear':	0.524, 'bicycle':	0.523, 'bird':	0.541, 'bus':	0.748, 'car':	0.483, 'cattle':	0.444, 'dog':	0.338, 'domestic_cat':	0.464, 'elephant':	0.632, 'fox':	0.524, 'giant_panda':	0.728, 'hamster':	0.674, 'horse':0.496, 'lion':	0.132, 'lizard':	0.655, 'monkey':	0.268, 'motorcycle':	0.701, 'rabbit':	0.355, 'red_panda':	0.258, 'sheep':	0.507, 'snake':	0.23, 'squirrel':	0.18, 'tiger':	0.8, 'train':	0.684, 'turtle':	0.686, 'watercraft':	0.512, 'whale':	0.609, 'zebra':	0.841}
class RoIDataLayer(caffe.Layer):
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
            print inds.shape
            inds = np.reshape(inds, (-1, 2))
            print inds.shape
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
            #db_inds = self._get_next_minibatch_inds()
            #minibatch_db = [self._roidb[i] for i in db_inds]
            minibatch_db=[]
            if 'ilsvrc' in self._roidb[0]['db_name']:
                while len(minibatch_db)<cfg.TRAIN.IMS_PER_BATCH:
                    if not cfg.TRAIN.USE_PERF_PROB:
                        #uniform distribution
                        c = random.choice(self._roidb_class.keys())
                    else:
                        #distribution based on performance
                        c = np.random.choice(self._roidb_class.keys(),p=self._p)
                    #print 'grabbing class %i'%c
                    if len(self._roidb_class[c])>0:
                        minibatch_db.append(random.choice(self._roidb_class[c]))
            else:
                db_inds = self._get_next_minibatch_inds()
                minibatch_db = [self._roidb[i] for i in db_inds]


            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb

        if 'ilsvrc' in roidb[0]['db_name']:
            self._classes = ('__background__',)  # always index 0
            self._class_name = ('__background__',)  # always index 0
            self._class_ids = ('__background__',)  # always index 0
            synsets = sio.loadmat(os.path.join(DEV_KIT_PATH, 'data', 'meta_vid.mat'))
            synsets = synsets['synsets'].squeeze()
            for i in range(30):
                self._classes += (str(synsets[i][2][0]),)
                self._class_name += (str(synsets[i][2][0]),)
                self._class_ids += (str(synsets[i][1][0]),)
            self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
            self._class_to_ind.update(dict(zip(self._class_ids, xrange(self._num_classes))))
            # probability distribution for each classes
            self._p = np.zeros(31,dtype=np.float32)
            for k in vid_perf.keys():
                self._p[self._class_to_ind[k]] = 1-vid_perf[k]# chances of getting drawn is inversely related to performance
            self._p/=self._p.sum()#norm
            print 'Reorganizing roidb into classes'
            self._roidb_class = {}
            for i in xrange(31):
                self._roidb_class[i]=[]
            for l in roidb:
                for c in l['gt_classes']:
                    self._roidb_class[c].append(l)

        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']
        print self._num_classes
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1
        # im_idx used for relation sampler
        # top[idx].reshape(1)
        # self._name_to_top_map['im_idx'] = idx
        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)

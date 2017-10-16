
import caffe
import scipy.io as sio
import os
import cv2
import numpy as np
import yaml
from multiprocessing import Process, Queue
import random
import h5py
import fast_rcnn.bbox_transform

from utils.cython_bbox import bbox_overlaps
import numpy as np
import utils.zl_utils as zl

class BoxToAffine(caffe.Layer):

    def setup(self, bottom, top):
        self._cur_idx = 0
        self.rdata = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_train.mat', struct_as_record=False,
                                 squeeze_me=True)
        self.vgg_data = h5py.File("output/sg_vrd_2016_train.hdf5",'r','core')
        self.meta = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5','r','core')
        layer_params = yaml.load(self.param_str_)

        self._batch_size = layer_params['batch_size']
        self.train_data = []
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(self._batch_size, 4096 * 2 )
        top[1].reshape(self._batch_size,101*2)
        top[2].reshape(self._batch_size, 4*2)

        top[3].reshape(self._batch_size)
        self._name_to_top_map['visual'] = 0
        self._name_to_top_map['classeme'] = 1
        self._name_to_top_map['location'] = 2

        self._name_to_top_map['label'] = 3

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self.get_minibatch()
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

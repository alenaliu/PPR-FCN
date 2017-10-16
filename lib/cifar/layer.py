import caffe
import scipy.io as sio
import os
import cv2
import numpy as np
import yaml
from multiprocessing import Process, Queue
import random

from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array
def load_mean(mean_file='models/cifar10/mean.binaryproto'):
    blob = caffe_pb2.BlobProto()
    data = open(mean_file, "rb").read()
    blob.ParseFromString(data)
    nparray = blobproto_to_array(blob)
    mean_data = nparray[0]
    channel_swap = (1, 2, 0)
    mean_data = mean_data.transpose(channel_swap)
    return mean_data
def augment(im):


    arr = [1, 3, 5]
    jpg = np.arange(30, 100, 7).astype(np.float32)
    jpg_prob = jpg / jpg.sum()
    jpg = jpg.astype(np.uint8)
    gauss_stddev = [10, 30, 50, 90]
    gauss_p = [.8, .1, .05, .05]
    k = (np.random.choice(arr, p=[.8, .1, .1]), np.random.choice(arr, p=[.8, .1, .1]))
    im_p = cv2.blur(im, k).astype(np.float32)
    gauss = np.zeros(im_p.shape, im_p.dtype)
    cv2.randn(gauss, (0), (np.random.choice(gauss_stddev, p=gauss_p)))
    im_p = (im_p + np.random.uniform(-10, 10)) * np.random.uniform(.9, 1.1)
    im_p = np.clip(im_p + gauss, 0, 255)
    im_p = im_p.astype(np.uint8)
    # print  [cv2.IMWRITE_JPEG_QUALITY, np.random.choice(jpg,p=jpg_prob)]
    flag, ajpg = cv2.imencode(".jpg", im_p, [cv2.IMWRITE_JPEG_QUALITY, int(np.random.choice(jpg, p=jpg_prob))])
    im_p = cv2.imdecode(ajpg, 1)

    return im_p
class CifarDataLayer(caffe.Layer):
    def im_list_to_blob(self,ims):
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in xrange(num_images):
            im_orig = ims[i]
            #im = augment(im_orig)
            im = im_orig
            im = im.astype(np.float32, copy=True)
            #im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])
            #im_orig -=self._mean
            im*=0.00390625
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob
    def get_minibatch(self):
        blobs={}
        """Given a roidb, construct a minibatch sampled from it."""
        num_images = self._batch_size
        idx = np.random.choice(len(self.train_data), num_images)
        im_list = []
        labels_blob = np.zeros(self._batch_size,np.int32)
        for i in xrange(self._batch_size):
            if self._cur_idx>=len(self.train_data):
                self._cur_idx=0
                random.shuffle(self.train_data)

            datum = self.train_data[self._cur_idx]
            im = cv2.imread(datum['path'])
            im_list.append(im)
            labels_blob[i] = datum['cls']
            self._cur_idx+=1
        blobs['data'] = self.im_list_to_blob(im_list)
        blobs['label'] =labels_blob

        return blobs
    def setup(self, bottom, top):
        self._mean = load_mean()
        self._cur_idx=0

        layer_params = yaml.load(self.param_str_)

        self._data_root = layer_params['data_root']
        self._batch_size = layer_params['batch_size']
        self.train_data=[]
        for path, subdirs, files in os.walk(self._data_root):
            for name in files:
                fpath = os.path.join(path, name)
                cls = name.split('_')[0]
                datum ={'cls':int(cls)}
                datum['path']=fpath
                self.train_data.append(datum)
        random.shuffle(self.train_data)
        """Setup the RoIDataLayer."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)


        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(self._batch_size, 3,32,32)
        top[1].reshape(self._batch_size)
        self._name_to_top_map['data'] = 0
        self._name_to_top_map['label'] = 1

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

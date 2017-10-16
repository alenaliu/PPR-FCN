import _init_paths
import cv2
import numpy as np
import caffe
import os
import cifar.layer

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
mean_file = None
def im_list_to_blob(ims):
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]

        im_orig = im.astype(np.float32, copy=True)
        #im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])
        im_orig -=mean_file
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im_orig
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob
def start_train():
    caffe.set_device(1)
    caffe.set_mode_gpu()

    ### load the solver and create train and test nets
    solver = None

    solver = caffe.SGDSolver('models/cifar10/solver.prototxt')
    niter = 1000000
    test_interval = 25
    # losses will also be stored in the log
    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        if niter%1000:
            print solver.net.blobs['loss'].data
    pass
def prep_train():

    data_root = '/home/zawlin/data/cifar/test'
    dst = 'models/cifar10/test.txt'
    train_data=[]
    f = open(dst,'w')
    for path, subdirs, files in os.walk(data_root):
        for name in files:
            fpath = os.path.join(path, name)
            cls = name.split('_')[0]
            f.write('%s %s\n'%(fpath,cls))
    f.close()
def test():
    global mean_file
    mean_file=load_mean()
    net = caffe.Net('models/cifar10/test.prototxt',
                    'output/cifar10/cifar10_full_iter_50000.caffemodel',caffe.TEST)
    data_root = '/home/zawlin/data/cifar/train'
    train_data=[]
    for path, subdirs, files in os.walk(data_root):
        for name in files:
            fpath = os.path.join(path, name)
            cls = name.split('_')[0]
            datum ={'cls':int(cls)}
            datum['path']=fpath
            train_data.append(datum)
    for i in train_data:
        im = cv2.imread(i['path'])
        print i['path']
        blob ={}
        blob['data'] = im_list_to_blob([im])
        result = net.forward(**blob)
        print result
        cv2.imshow('im',im)
        if cv2.waitKey(0)&0xFF==27:
            exit(0)
    pass
#prep_train()
start_train()
#test()

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import h5py
import cv2


def preload():
    net = caffe.Net('models/sg_vrd/rel_pre_iccv/test_pre_psroi_context_tri_sum_deep.prototxt',caffe.TEST)
    net.copy_from('data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_70000.caffemodel')
    for k in net.params.keys():
        if ('sub_' == k[:4] or 'obj_'==k[:4] ) and k[4:] in net.params.keys():

            for i in xrange(len(net.params[k])):
                net.params[k][i].data[...] = net.params[k[4:]][i].data[...]
    net.save('data/models/vrd_rfcn/deep_tri_init_vrd_resnet50_rfcn_iter_70000.caffemodel')
    #net.save_hdf5('output/models/sg_vrd_relation_vgg_init_100000_80000.h5')

    print dir(net)

preload()
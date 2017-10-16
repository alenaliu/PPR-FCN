import _init_paths
from vrd.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import h5py
import cv2

def do_surgery():
    net = caffe.Net('models/sg_vrd/vgg16_finetune/faster_rcnn_end2end/vgg_template.prototxt','output/models/vgg16_faster_rcnn_iter_50000.caffemodel',caffe.TEST)
    net.save('output/models/vgg16_faster_rcnn_iter_50000_4finetune.caffemodel')
    pass

def preload_relation_weights():

    net = caffe.Net('models/sg_vrd/vgg16_relation/faster_rcnn_end2end/test.prototxt',caffe.TEST)
    #net = caffe.Net('models/sg_vrd/vgg16_relation/faster_rcnn_end2end/train_dbg.prototxt',caffe.TEST)
    net.copy_from('output/models/vgg16_faster_rcnn_finetune_iter_80000.caffemodel')

    #net.save_hdf5('output/models/vgg16_faster_rcnn_finetune_iter_80000.h5')
    print net.params.keys()
    print net.params['fc6'][0].data
    net.copy_from('output/models/sg_vrd_relation_vgg16_iter_100000.caffemodel')
    print net.params['fc6'][0].data
    net.save('output/models/sg_vrd_relation_vgg_init_100000_80000.caffemodel')
    #net.save_hdf5('output/models/sg_vrd_relation_vgg_init_100000_80000.h5')

    print dir(net)

def preload_relation_weights_seperated_w():

    net = caffe.Net('models/sg_vrd/vgg16_relation/faster_rcnn_end2end/test.prototxt',caffe.TEST)
    #net = caffe.Net('models/sg_vrd/vgg16_relation/faster_rcnn_end2end/train_dbg.prototxt',caffe.TEST)
    net.copy_from('output/models/vgg16_faster_rcnn_finetune_iter_80000.caffemodel')

    #net.save_hdf5('output/models/vgg16_faster_rcnn_finetune_iter_80000.h5')
    print net.params.keys()
    print net.params['fc6'][0].data
    net.copy_from('output/models/sg_vrd_relation_vgg16_iter_100000.caffemodel')
    print net.params['fc6'][0].data
    net.save('output/models/sg_vrd_relation_vgg_init_100000_80000.caffemodel')
    #net.save_hdf5('output/models/sg_vrd_relation_vgg_init_100000_80000.h5')

    print dir(net)

preload_relation_weights()
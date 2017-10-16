import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import zl_config as C
from fast_rcnn.test import im_detect
import matplotlib.pyplot as plt
from fast_rcnn.nms_wrapper import nms

from multiprocessing import Pool, TimeoutError
import cPickle
import os
from utils.timer import Timer
import utils.zl_edge_box as zl_eb
import utils.zl_utils as zl

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

def do_one_image(file_path):
    file_name = file_path.split('/')[-1]
    edge_box_path = '/media/zawlin/ssd/iccv2017/data/voc/gen_eb/train/'
    windows = zl_eb.get_windows(file_path,maxboxes=4000,minscore=0.001)
    zl.save(edge_box_path+file_name+'.eb',windows)


def do_one_image_vrd(file_path):
    file_name = file_path.split('/')[-1]
    edge_box_path = '/home/zawlin/data/vrd/edge_boxes/sg_test_images/'
    windows = zl_eb.get_windows(file_path,maxboxes=6000,minscore=0.0001)
    zl.save(edge_box_path+file_name+'.eb',windows)

def process_voc_edge_box():
    cfg_from_file('experiments/cfgs/rfcn_end2end.yml')

    imdb, roidb = combined_roidb('voc_0712_train')
    num_images = len(imdb.image_index)
    print num_images

    pool = Pool(processes=8)
    for i in xrange(num_images):
        im_path = imdb.image_path_at(i)
        #do_one_image(im_path)
        pool.apply_async(do_one_image, (im_path,))

    pool.close()
    pool.join()

def process_vrd_edge_box_train():
    root = '/home/zawlin/data/vrd/sg/Data/sg_test_images'
    eb_list = []

    pool = Pool(processes=8)
    for path, subdirs, files in os.walk(root):
        for name in files:
            fpath = os.path.join(path, name)
            #do_one_image_vrd(fpath)
            pool.apply_async(do_one_image_vrd,(fpath,))

    pool.close()
    pool.join()



#process_voc_edge_box()
process_vrd_edge_box_train()



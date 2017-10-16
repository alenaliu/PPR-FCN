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


cfg_from_file('experiments/cfgs/rfcn_end2end.yml')

imdb, roidb = combined_roidb('voc_0712_test')

import cv2

ann = roidb[9]
im = cv2.imread(ann['image'])
idx = 0
for bb in ann['boxes']:
    cv2.rectangle(im,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),1)
cv2.imwrite('/home/zawlin/data/all.jpg',im)
cv2.imshow('im2',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

net =None
cfg.TEST.HAS_RPN=False
prototxt = 'models/pascal_voc/ResNet-50/rfcn_end2end/test_iccv_rpn.prototxt'
#model = 'data/rfcn_models/resnet50_rfcn_iter_600.caffemodel'
model = 'output/rfcn_end2end/voc_0712_train/resnet50_rfcn_iter_600.caffemodel'
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(prototxt, model, caffe.TEST)

print ann['image']

#im = cv2.imread('data/demo/004545.jpg')
im = cv2.imread(ann['image'])
print ann['boxes']
im_detect(net, im,boxes=ann['boxes'])

attention_caffe = net.blobs['attention'].data.copy()

rois = net.blobs['rois'].data

attention = attention_caffe[:,0].squeeze()
ind = np.argsort(attention)[::-1]
attention = attention[ind]
rois = rois[ind]
#rois_all = np.hstack((rois[:,1:],np.zeros(rois.shape[0],np.float32)))
#rois_all =  rois_all[ind]
for i in xrange(20):
    roi = rois[i]
    ascore = '%0.3f'%attention[i]

    roi = rois[i]
    cv2.putText(im,ascore,(int(roi[1]+10),int(roi[2]+20)),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,0),1)
    #cv2.rectangle(im,(roi[1],roi[2]),(roi[3],roi[4]),(255,0,0),1)
    cv2.rectangle(im,(roi[1],roi[2]),(roi[3],roi[4]),(255,0,0),1)

cv2.imshow('im',im)
cv2.imwrite('/home/zawlin/data/bus.jpg',im)
cv2.waitKey(0)
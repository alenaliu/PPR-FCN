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
import h5py
import cPickle
import os
from utils.timer import Timer
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


cfg_from_file('experiments/cfgs/rfcn_end2end_iccv_eb.yml')

#cfg_from_file('experiments/cfgs/rfcn_end2end_iccv_eb.yml')
imdb, roidb = combined_roidb('voc_0712_test')

import cv2
h5f = h5py.File('/media/zawlin/ssd/iccv2017/data/voc/gen_eb.h5',driver='core')
h5_rois = {}

for i in h5f['test/']:

    data=h5f['test/%s'%i][...].astype(np.float32)
    idx = np.argsort(data[:,-1],axis=0)
    data_sorted = data[idx][::-1]
    #data_sorted_idx = np.where((data_sorted[:,2]-data_sorted[:,0]>20) & (data_sorted[:,3]-data_sorted[:,1]>20))
    #data_sorted = data_sorted[data_sorted_idx]
    #print data_sorted
    h5_rois[i] = data_sorted[:50,:4]

#cfg.TEST.HAS_RPN=False
net =None
prototxt = 'models/pascal_voc/ResNet-50/rfcn_end2end/test_iccv_eb_sigmoid.prototxt'
#model = 'data/rfcn_models/resnet50_rfcn_iter_1200.caffemodel'

#model = 'output/rfcn_end2end/voc_0712_train/resnet50_rfcn_iter_16000.caffemodel'
model = 'output/rfcn_end2end/voc_0712_train/resnet50_rfcn_eb_sigx_iter_100000.caffemodel'
#model = 'data/rfcn_models_iccv/eb_resnet50_rfcn_iter_600.caffemodel'
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(prototxt, model, caffe.TEST)

#prototxt = 'models/pascal_voc/ResNet-50/rfcn_end2end/test_iccv_rpn.prototxt'
#model = 'data/rfcn_models_iccv/eb_resnet50_rfcn_iter_800.caffemodel'
#model = 'output/rfcn_end2end/voc_0712_train/resnet50_rfcn_iter_1600.caffemodel'
#model = 'data/rfcn_models_iccv/eb_resnet50_rfcn_iter_800.caffemodel'
#net2 = caffe.Net(prototxt, model, caffe.TEST)
#net.params['conv_new_1_zl'][0].data[...] =  net2.params['conv_new_1_zl'][0].data[...]
#net.params['conv_new_1_zl'][1].data[...] =  net2.params['conv_new_1_zl'][1].data[...]
#net2 = None
net.name = 'resnet50_rfcn_iter_1200'
num_images = len(imdb.image_index)
#num_images = 100
#del imdb.image_index[num_images:]
#num_images = 10#len(imdb.image_index)
all_boxes = [[[] for _ in xrange(num_images)]
             for _ in xrange(imdb.num_classes)]

output_dir = get_output_dir(imdb, net)
zl.tic()
# timers
_t = {'im_detect' : Timer(), 'misc' : Timer()}

max_per_image = 50
thresh = 0.7
for i in xrange(num_images):
    # filter out any ground truth boxes
    im_path = imdb.image_path_at(i)
    im_name = im_path.split('/')[-1]
    eb_roi = h5_rois[im_name]
    im = cv2.imread(imdb.image_path_at(i))
    _t['im_detect'].tic()
    #scores, boxes = im_detect(net, im, box_proposals)
    scores, boxes = im_detect(net, im, eb_roi)
    attention = net.blobs['attention'].data.squeeze()
    #net.blobs['attention'].data
    #scores = np.multiply(scores,attention)
    _t['im_detect'].toc()

    _t['misc'].tic()
    # skip j = 0, because it's the background class
    for j in xrange(1, imdb.num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        if cfg.TEST.AGNOSTIC:
            cls_boxes = boxes[inds, 1:]
        else:
            cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS,force_cpu=True)
        cls_dets = cls_dets[keep, :]
        all_boxes[j][i] = cls_dets
        cls_str = imdb.classes[j]
        for roi in all_boxes[j][i]:
            cv2.putText(im,cls_str,(roi[0],roi[1]),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,0),1)
            cv2.rectangle(im,(roi[0],roi[1]),(roi[2],roi[3]),(0,0,255),1)

    # Limit to max_per_image detections *over all classes*

    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1]
                                  for j in xrange(1, imdb.num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, imdb.num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]
    #cv2.imshow('vis',im)
    #cv2.waitKey(0)
    _t['misc'].toc()

    print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
                _t['misc'].average_time)

det_file = os.path.join(output_dir, 'detections.pkl')
with open(det_file, 'wb') as f:
    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

print 'Evaluating detections'
imdb.evaluate_detections(all_boxes, output_dir)
print zl.toc()

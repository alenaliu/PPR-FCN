import _init_paths
from vrd.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net('models/sg_vrd/vgg16/faster_rcnn_end2end/test.prototxt','output/faster_rcnn_end2end/sg_vrd_2016_train/vgg16_finetune_faster_rcnn_iter_20000.caffemodel', caffe.TEST)
#net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
net.name='sgvrd'
imdb = get_imdb('sg_vrd_2016_train')
imdb.competition_mode(0)
if not cfg.TEST.HAS_RPN:
    imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

test_net(net, imdb, max_per_image=1, vis=0)
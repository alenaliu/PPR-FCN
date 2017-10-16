import _init_paths
import cv2
import numpy as np
import caffe
import os
import cifar.layer

from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array

def start_train():
    caffe.set_device(0)
    caffe.set_mode_gpu()

    ### load the solver and create train and test nets
    solver = None

    #solver = caffe.SGDSolver('models/sg_vrd/rel_iccv/solver_opsroi.prototxt')
    #solver = caffe.SGDSolver('models/sg_vrd/rel_iccv/solver_jointbox_cached.prototxt')
    #solver = caffe.SGDSolver('models/sg_vrd/rel_pre_iccv/solver_pre_psroi_context_tri_sum_cached.prototxt')
    #solver = caffe.SGDSolver('models/sg_vrd/rel_pre_iccv/solver_pre_psroi_context_tri_sum.prototxt')
    #solver = caffe.SGDSolver('models/sg_vrd/rel_pre_iccv/solver_pre_psroi_context_tri_sum_deep.prototxt')
    solver = caffe.SGDSolver('models/sg_vrd/rel_pre_iccv/solver_exp10.prototxt')
    #solver = caffe.SGDSolver('models/sg_vrd/rel_iccv/solver_classeme_gt_cached.prototxt')
    #solver = caffe.SGDSolver('models/sg_vrd/rel_iccv/solver_visual_fc_cached.prototxt')
    #solver.net.copy_from('data/imagenet/ResNet-50-model.caffemodel')
    #solver.net.copy_from('output/rfcn_end2end/sg_vrd_2016_train/vrd_resnet50_rfcn_iter_11500.caffemodel')
    #solver.net.copy_from('output/sg_vrd_rfcn/resnet50_rfcn_iccv_gt_iter_10500.caffemodel')
    #solver.net.copy_from('data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_70000.caffemodel')
    solver.net.copy_from('data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_64000.caffemodel')
    #solver.net.copy_from('data/models/vrd_rfcn/deep_tri_init_vrd_resnet50_rfcn_iter_70000.caffemodel')
    #solver.net.copy_from('output/rel_iccv/pre_psroi_context_tri_sum_iter_10500.caffemodel')

    niter = 1000000
    test_interval = 25
    # losses will also be stored in the log
    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        if niter%1000:
            print solver.net.blobs['loss'].data

start_train()
#test()

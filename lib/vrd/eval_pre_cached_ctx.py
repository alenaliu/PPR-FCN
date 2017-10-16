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
import scipy.io as sio

from numpy.core.records import fromarrays
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
import h5py
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
import zl_config as C
import utils.zl_utils as zl
import glog
from numpy import linalg as LA

def union_np(a,b):
    x = np.minimum(a[:,1], b[:,1])
    y = np.minimum(a[:,2], b[:,2])
    x2 = np.maximum(a[:,3],b[:,3])
    y2 = np.maximum(a[:,4],b[:,4])
    return np.concatenate((a[:,0,np.newaxis],x[:,np.newaxis],y[:,np.newaxis],x2[:,np.newaxis],y2[:,np.newaxis]),axis=1)

def _get_image_blob(im_path):
    im = cv2.imread(im_path)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def im_detect(net, im_path, conv_new_1,sub_boxes_gt,obj_boxes_gt):

    zeros = np.zeros((sub_boxes_gt.shape[0],1), dtype=np.float)
    # first index is always zero since we do one image by one image
    sub_boxes = np.concatenate((zeros, sub_boxes_gt),axis=1)
    obj_boxes = np.concatenate((zeros, obj_boxes_gt),axis=1)
    #blobs['data'], im_scales = _get_image_blob(im_path)
    _, im_scales = _get_image_blob(im_path)
    sub_boxes = (sub_boxes*im_scales[0]).reshape(-1,5,1,1)
    obj_boxes = (obj_boxes*im_scales[0]).reshape(-1,5,1,1)
    union_boxes = union_np(sub_boxes,obj_boxes)
    # reshape network inputs
    net.blobs['conv_new_1'].reshape(*conv_new_1.shape)
    net.blobs['sub_boxes'].reshape(*sub_boxes.shape)
    net.blobs['obj_boxes'].reshape(*obj_boxes.shape)
    net.blobs['union_boxes'].reshape(*union_boxes.shape)

    forward_kwargs = {'conv_new_1': conv_new_1,
                      'sub_boxes':sub_boxes,
                      'obj_boxes':obj_boxes,
                      'union_boxes':union_boxes,
                      }

    blobs_out = net.forward(**forward_kwargs)
    return net.blobs['relation_prob'].data[...]

def run_relation(model_type,iteration):
    cache_h5= h5py.File('output/sg_vrd_cache.h5')['test']
    result = h5py.File('output/sg_vrd_2016_result_'+model_type+'_'+iteration+'.hdf5')
    m = h5py.File('data/sg_vrd_meta.h5')
    data_root='data/sg_vrd_2016/Data/sg_test_images/'
    keep = 100
    thresh = 0.0001
    prototxt = 'models/sg_vrd/rel_pre_iccv/test_'+model_type+'.prototxt'
    model_path ='output/rel_iccv/'+model_type+'_iter_'+iteration+'.caffemodel'
    net = caffe.Net(prototxt,
                    model_path,
                    caffe.TEST)
    # sio.savemat('output/'+model_type+'.mat',{'params_'+model_type:net.params['relation'][0].data})
    # exit(0)
    #net = caffe.Net('models/sg_vrd/relation/test.prototxt','output/models/sg_vrd_relation_vgg16_iter_264000.caffemodel',caffe.TEST)
    cnt =0
    for imid in cache_h5.keys():
        cnt+=1
        if cnt%10==0:
            glog.info(cnt)

        conv_new_1  = cache_h5[imid]['conv_new_1'][...]
        obj_boxes_gt = m['gt/test'][imid]['obj_boxes'][...]
        sub_boxes_gt = m['gt/test'][imid]['sub_boxes'][...]
        rlp_labels_gt = m['gt/test'][imid]['rlp_labels'][...]
        rlp_labels = []
        rlp_confs = []
        sub_boxes=[]
        obj_boxes=[]
        for s in xrange(sub_boxes_gt.shape[0]):
            im_path = C.get_sg_vrd_path_test(imid)
            im_detect(net,im_path,conv_new_1,sub_boxes_gt,obj_boxes_gt)
            relation_prob = net.blobs['relation_prob'].data[...]
            sub_cls = rlp_labels_gt[:,0]
            obj_cls = rlp_labels_gt[:,2]
            argmax = np.argmax(relation_prob[s,...])
            gt_pre = rlp_labels_gt[s,1]
            rs = relation_prob[s,argmax]

            predicate = argmax
            #rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
            rlp_label = np.array([sub_cls[s],predicate,obj_cls[s]]).astype(np.int32)
            #print '%s %s %s %f'%(m['meta/cls/idx2name/'+str(rlp_label[0])][...],m['meta/pre/idx2name/'+str(rlp_label[1])][...],m['meta/cls/idx2name/'+str(rlp_label[2])][...],rs)
            rlp_conf = rs#+sub_score+obj_score#relation_score[predicate]

            rlp_confs.append(rlp_conf)
            rlp_labels.append(rlp_label)
            sub_boxes.append(sub_boxes_gt[s])
            obj_boxes.append(obj_boxes_gt[s])
            # for i in xrange(70):
            # rs = relation_score[i]
            # if rs>0.0:
            # predicate =i
            # #print relation_score[predicate]
            # rlp_label = np.array([sub_cls[s],predicate,obj_cls[s]]).astype(np.int32)
            # #print '%s %s %s %f'%(m['meta/cls/idx2name/'+str(rlp_label[0])][...],m['meta/pre/idx2name/'+str(rlp_label[1])][...],m['meta/cls/idx2name/'+str(rlp_label[2])][...],rs)
            # rlp_conf = rs#+sub_score+obj_score#relation_score[predicate]
            # rlp_confs.append(rlp_conf)
            # rlp_labels.append(rlp_label)
            # sub_boxes.append(sub_boxes_gt[s])
            # obj_boxes.append(obj_boxes_gt[s])

        result.create_dataset(imid+'/rlp_confs',dtype='float16', data=np.array(rlp_confs).astype(np.float16))
        result.create_dataset(imid+'/sub_boxes',dtype='float16', data=np.array(sub_boxes).astype(np.float16))
        result.create_dataset(imid+'/obj_boxes',dtype='float16', data=np.array(obj_boxes).astype(np.float16))
        result.create_dataset(imid+'/rlp_labels',dtype='float16', data=np.array(rlp_labels).astype(np.float16))


def make_relation_result(model_type,iteration):
    m = h5py.File('data/sg_vrd_meta.h5')
    result_str = 'sg_vrd_2016_result_'+model_type+'_'+iteration
    result = h5py.File('output/'+result_str+'.hdf5')
    rlp_confs=[]
    rlp_labels=[]
    subbox=[]
    objbox=[]
    for i in xrange(1000):
        imid = str(m['db/testidx/'+str(i)][...])
        if imid in result:
            objbox.append(result[imid+'/obj_boxes'][...])#\.reshape(-1,4).T)
            subbox.append(result[imid+'/sub_boxes'][...])
            rlp_labels.append(result[imid+'/rlp_labels'][...])
            rlp_confs.append(result[imid+'/rlp_confs'][...].T)
        else:
            rlp_confs.append([])
            rlp_labels.append([])
            subbox.append([])
            objbox.append([])

    #print objbox
    #objboxx=np.array(objbox)#.astype(np.float64)
    sio.savemat('output/'+result_str+'.mat', {'obj_bboxes_ours': objbox,'sub_bboxes_ours':subbox,
                                              'rlp_labels_ours':rlp_labels,'rlp_confs_ours':rlp_confs})
    pass

def print_mappings():
    m = h5py.File('data/sg_vrd_meta.h5')
    #print m['meta/pre/name2idx'].keys()
    keys = []
    for k in m['meta/pre/idx2name'].keys():
        keys.append(int(k))
    keys = sorted(keys)
    for k in keys:
        print str(k)+','+m['meta/pre/idx2name/'+str(k)][...]
        #print sorted(keys)

caffe.set_mode_gpu()
caffe.set_device(0)
#make_meta()
#exit(0)
model_type = 'exp15'
iteration = '5000'
run_relation(model_type,iteration)
#run_relation_diff(model_type,iteration)
#run_relation_diff_classeme()
make_relation_result(model_type,iteration)

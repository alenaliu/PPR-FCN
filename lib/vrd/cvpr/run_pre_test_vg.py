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
import utils.zl_utils as zl

from numpy import linalg as LA

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #x = x/np.max(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_relation(model_type,iteration):
    vgg_data = h5py.File('output/precalc/vg1_2_2016_predicate_exp_test.hdf5')
    result = h5py.File('output/vg1_2_2016_result_'+model_type+'_'+iteration+'.hdf5')
    #if os.path.exists('output/sg_vrd_2016_result.hdf5'):
    #    os.remove('output/sg_vrd_2016_result.hdf5')
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/vg1_2_2016/Data/test/'
    keep = 100
    thresh = 0.0001
    net = caffe.Net('models/vg1_2/relation/test_'+model_type+'.prototxt','output/relation/vg/relation_vgg16_'+model_type+'_iter_'+iteration+'.caffemodel',caffe.TEST)

    cnt =0
    zl.tick()
    for imid in vgg_data.keys():
        if cnt%100==0:
            print cnt,zl.tock()
            zl.tick()
        cnt+=1
        obj_boxes_gt = vgg_data[imid]['obj_boxes']
        sub_boxes_gt = vgg_data[imid]['sub_boxes']
        sub_visual = vgg_data[imid]['sub_visual']
        obj_visual = vgg_data[imid]['obj_visual']
        joint_visual = vgg_data[imid]['joint_visual']
        sub_cls = vgg_data[imid]['sub_cls']
        obj_cls = vgg_data[imid]['obj_cls']

        rlp_labels = []
        rlp_confs = []
        sub_boxes=[]
        obj_boxes=[]
        for s in xrange(sub_boxes_gt.shape[0]):
            if model_type=='pre_diff':
                blob = {
                    'visual_s':np.array(sub_visual[s]).reshape(1,4096),
                    'visual_o':np.array(obj_visual[s]).reshape(1,4096),
                    }
            elif model_type=='pre_jointbox':
                blob = {
                    'visual':np.array(joint_visual[s]).reshape(1,4096),
                    }
                pass
            elif model_type=='pre_concat':
                visual = np.hstack((sub_visual[s], obj_visual[s])).reshape(1,8192)
                blob = {
                    'visual':visual
                    }
            #print visual.shape
            net.forward_all(**blob)
            relation_score =net.blobs['relation'].data[0]
            #l2_norm = relation_score/LA.norm(relation_score)
            relation_score=softmax(relation_score)
            #relation_score/=LA.norm(relation_score)
            argmax = np.argmax(relation_score)
            rs = relation_score[argmax]

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

def make_meta():
    data = sio.loadmat('/home/zawlin/g/Visual-Relationship-Detection/data/imagePath.mat', struct_as_record=False, squeeze_me=True)
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    for i in xrange(len(data['imagePath'])):
        m['db/testidx/'+str(i)]=data['imagePath'][i].split('.')[0]
    pass

def make_relation_result(model_type,iteration):
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    result_str = 'vg1_2_2016_result_'+model_type+'_'+iteration
    result = h5py.File('output/'+result_str+'.hdf5')
    rlp_confs=[]
    rlp_labels=[]
    subbox=[]
    objbox=[]
    imids = sorted(m['gt/test'].keys())
    for imid in imids:
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
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    #print m['meta/pre/name2idx'].keys()
    keys = []
    for k in m['meta/pre/idx2name'].keys():
        keys.append(int(k))
    keys = sorted(keys)
    for k in keys:
        print str(k)+','+m['meta/pre/idx2name/'+str(k)][...]

caffe.set_mode_gpu()
caffe.set_device(0)
#make_meta()
#exit(0)
model_type = 'pre_jointbox'
iteration = '80000'
run_relation(model_type,iteration)
#run_relation_diff(model_type,iteration)
#run_relation_diff_classeme()
make_relation_result(model_type,iteration)

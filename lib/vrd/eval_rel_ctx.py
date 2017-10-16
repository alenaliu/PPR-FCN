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


def im_detect(net, im_path, sub_boxes_gt,obj_boxes_gt):
    try:
        zeros = np.zeros((len(sub_boxes_gt),1), dtype=np.float)
        # first index is always zero since we do one image by one image
        sub_boxes = np.concatenate((zeros, np.array(sub_boxes_gt)),axis=1)
        obj_boxes = np.concatenate((zeros, np.array(obj_boxes_gt)),axis=1)
        #blobs['data'], im_scales = _get_image_blob(im_path)
        im_blob, im_scales = _get_image_blob(im_path)
        sub_boxes = (sub_boxes*im_scales[0]).reshape(-1,5,1,1)
        obj_boxes = (obj_boxes*im_scales[0]).reshape(-1,5,1,1)
        union_boxes = union_np(sub_boxes,obj_boxes)
        # reshape network inputs
        net.blobs['data'].reshape(*im_blob.shape)
        net.blobs['sub_boxes'].reshape(*sub_boxes.shape)
        net.blobs['obj_boxes'].reshape(*obj_boxes.shape)
        net.blobs['union_boxes'].reshape(*union_boxes.shape)

        forward_kwargs = {'data': im_blob,
                          'sub_boxes':sub_boxes,
                          'obj_boxes':obj_boxes,
                          'union_boxes':union_boxes,
                          }

        blobs_out = net.forward(**forward_kwargs)
    except Exception as ex:
        glog.info("error @ im_path")
        glog.info(ex)
    return net.blobs['relation_prob'].data[...]

def run_relation(model_type,iteration):
    cache_h5= h5py.File('output/sg_vrd_cache.h5')['test']
    result = h5py.File('output/sg_vrd_2016_result_'+model_type+'_'+iteration+'.hdf5')
    m = h5py.File('data/sg_vrd_meta.h5')
    #--------------cache boxes-----------------
    h5_boxes = h5py.File('output/precalc/sg_vrd_objs.hdf5')
    cache_boxes = {}
    if os.path.exists('output/cache/sg_vrd_objs_test.pkl'):
        cache_boxes= zl.load('output/cache/sg_vrd_objs_test.pkl')
        glog.info('loaded obj data from cache')
    else:
        glog.info( 'Preloading obj')
        zl.tic()
        for k in h5_boxes['test'].keys():
            boxes = h5_boxes['test/%s/boxes'%k][...]
            cache_boxes[k]=boxes
        glog.info('done preloading obj %f'%zl.toc())
        zl.save('output/cache/sg_vrd_obj_test.pkl',cache_boxes)
    #--------------cache boxes-----------------

    #--------------cache old boxes-------------
    h5_boxes = h5py.File('output/sg_vrd_2016_test.hdf5')
    cache_old_boxes = {}
    if os.path.exists('output/cache/sg_vrd_objs_test_vgg.pkl'):
        cache_old_boxes= zl.load('output/cache/sg_vrd_objs_test_vgg.pkl')
        glog.info('loaded obj data from cache')
    else:
        glog.info( 'Preloading obj')
        zl.tic()
        for k in h5_boxes.keys():
            locations = h5_boxes['%s/locations'%k][...]
            cls_confs = h5_boxes['%s/cls_confs'%k][...]
            boxes=np.concatenate((locations,cls_confs[:,1,np.newaxis],cls_confs[:,0,np.newaxis]),axis=1)
            cache_old_boxes[k]=boxes
        glog.info('done preloading obj %f'%zl.toc())
        zl.save('output/cache/sg_vrd_obj_test_vgg.pkl',cache_old_boxes)

    #--------------cache old boxes-------------
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
        obj_boxes_gt = m['gt/test'][imid]['obj_boxes'][...]
        sub_boxes_gt = m['gt/test'][imid]['sub_boxes'][...]
        rlp_labels_gt = m['gt/test'][imid]['rlp_labels'][...]
        rlp_labels = []
        rlp_confs = []
        sub_boxes=[]
        obj_boxes=[]
        #boxes = cache_boxes[imid]
        boxes = cache_old_boxes[imid]

        if boxes.shape[0]>=2:
            for s in xrange(boxes[:20].shape[0]):
                for o in xrange(boxes[:20].shape[0]):
                    if s == o:continue
                    if boxes[s][4]<0.01:continue
                    if boxes[o][4]<0.01:continue
                    sbox = boxes[s][:4]
                    obox = boxes[o][:4]
                    rlp_labels.append([boxes[s,5],-1,boxes[o,5]])
                    rlp_confs.append(boxes[s,4]+boxes[o,4])
                    sub_boxes.append(sbox)
                    obj_boxes.append(obox)
            if len(sub_boxes)<=0:continue

            #sub_box = np.array(sub_box)
            #obj_box = np.array(obj_box)
            im_path = C.get_sg_vrd_path_test(imid)
            im_detect(net,im_path,sub_boxes,obj_boxes)
            relation_prob = net.blobs['relation_prob'].data[...]
            for r in xrange(relation_prob.shape[0]):
                argmax = np.argmax(relation_prob[r,...])
                rs = relation_prob[r,argmax].squeeze()

                rlp_labels[r][1] = argmax
                rlp_confs[r] = rlp_confs[r] + rs


        im = cv2.imread(data_root+imid+'.jpg')
        for i in xrange(len(sub_boxes)):
            sb = sub_boxes[i]
            cv2.rectangle(im,(int(sb[0]),int(sb[1])),(int(sb[2]),int(sb[3])),(255,0,0),2  )
            pass
        cv2.imshow('im',im)
        #if cv2.waitKey(0)==27:
        #    exit(0)

        #rlp_confs.append(rlp_conf)
        #rlp_labels.append(rlp_label)
        #sub_boxes.append(sub_box)
        #obj_boxes.append(obj_box)
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
            ob = result[imid+'/obj_boxes'][...]#\.reshape(-1,4).T)
            objbox.append(ob)
            subbox.append(result[imid+'/sub_boxes'][...])
            rlp_labels.append(result[imid+'/rlp_labels'][...])
            rlp_confs.append(result[imid+'/rlp_confs'][...].T)
        else:
            rlp_confs.append([])
            rlp_labels.append([])
            subbox.append([])
            objbox.append([])
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
caffe.set_device(1)
#make_meta()
#exit(0)
model_type = 'pre_psroi_context_tri_sum'
iteration = '17400'
run_relation(model_type,iteration)
#run_relation_diff(model_type,iteration)
#run_relation_diff_classeme()
make_relation_result(model_type,iteration)
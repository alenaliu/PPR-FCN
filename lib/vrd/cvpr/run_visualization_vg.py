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

from fast_rcnn.config import cfg, get_output_dir
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
import operator
import random


def random_color():
    b = np.random.randint(0,255)
    g = np.random.randint(0,255)
    r = np.random.randint(0,255)
    return (b,g,r)
def run_visualization():
    #h5f= h5py.File('output/sg_vrd_2016_result_pre_diff_5000.hdf5')

    h5f= h5py.File('output/sg_vrd_2016_result_pre_jointbox_15000.hdf5')
    #h5f= h5py.File('output/sg_vrd_2016_result_pre_concat_14000.hdf5')
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/Data/sg_test_images/'
    keep = 100
    thresh = 0.0001
    cnt =0
    pre =54
    thresh = 0.001
    rois = []
    images = {}
    for imid in h5f.keys():
        cnt+=1
        if cnt%100==0:
            print cnt
        obj_boxes = h5f[imid+'/obj_boxes'][...]
        sub_boxes = h5f[imid+'/sub_boxes'][...]
        rlp_labels = h5f[imid+'/rlp_labels']
        rlp_confs = h5f[imid+'/rlp_confs'][...]
        inds = np.where(rlp_labels[:,1]==pre)

        obj_boxes = obj_boxes[inds]
        sub_boxes = sub_boxes[inds]
        rlp_confs = rlp_confs[inds]
        inds = np.where(rlp_confs>thresh)
        obj_boxes = obj_boxes[inds]
        sub_boxes = sub_boxes[inds]
        rlp_confs = rlp_confs[inds]
        if rlp_confs.shape[0]<=0: continue
        im_orig = cv2.imread(data_root+imid+'.jpg')
        for i in xrange(obj_boxes.shape[0]):
            im = im_orig.copy()
            o_box = obj_boxes[i]
            s_box = sub_boxes[i]
            conf =  rlp_confs[i]
            u_box = [min(s_box[0],o_box[0]),min(s_box[1],o_box[1]),max(s_box[2],o_box[2]),max(s_box[3],o_box[3])]
            # roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            # roi = 255-roi
            # roi=cv2.applyColorMap(roi,cv2.COLORMAP_JET)
            #roi = cv2.resize(roi,(300,300))

            # roi_avg = np.average(np.array(rois),axis=0).astype(np.uint8)
            # cv2.imshow('avg',roi_avg)
            #cv2.rectangle(im,

            cv2.rectangle(im,(s_box[0],s_box[1]),(s_box[2],s_box[3]),(0,200,0),1)
            cv2.rectangle(im,(o_box[0],o_box[1]),(o_box[2],o_box[3]),(0,0,200),1)
            #cv2.rectangle(im,(u_box[0],u_box[1]),(u_box[2],u_box[3]),(0,0,255),2)
            #cv2.rectangle(im,(u_box[0],u_box[1]),(u_box[2],u_box[3]),(0,0,255),2)
            #cv2.putText(im,'%0.4f'%conf,(int(u_box[0])+30,int(u_box[1])+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
            pad_sz = 15
            px = pad_sz if u_box[0]-pad_sz>=0 else u_box[0]-1
            py = pad_sz if u_box[1]-pad_sz>=0 else u_box[1]-1
            px1 = pad_sz if u_box[2]+pad_sz<im.shape[1] else im.shape[1]-u_box[2]
            py1 = pad_sz if u_box[3]+pad_sz<im.shape[0] else im.shape[0]-u_box[3]
            pad = [-px,-py,px1,py1]
            roi = im[u_box[1]+pad[1]:u_box[3]+pad[3],u_box[0]+pad[0]:u_box[2]+pad[2],:].copy()
            rois.append(roi)
            images[conf]= roi
        cv2.imshow('im',im)
        # if len(rois)<5:
            # cv2.waitKey(0)
        # if cv2.waitKey(0)==27:
            # exit(0)
    images_sorted = sorted(images.items(), key=operator.itemgetter(0),reverse=True)
    cnt = 0
    for k,v in images_sorted:
        cv2.imshow('roi',v)

        cv2.imwrite( 'output/results/examples/%d_%0.5f_%s.jpg'%(cnt,k,'jointbox') ,v)
        cnt+=1
        if cnt >20:
            exit(0)
        if cv2.waitKey(1)==27:
            exit(0)
    print len(rois)
    # roi_avg = np.average(np.array(rois),axis=0).astype(np.uint8)
    # cv2.imwrite('output/results/roi_avg.jpg',roi_avg)

def nothing(x):
    pass

def run_relation_visualization():
    h5f_ours = h5py.File('output/vg_results/vg1_2_2016_result_all_100000.all.hdf5','r')
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/vg1_2_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/vg1_2_2016/Data/test/'
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5', 'r', 'core')
    thresh = 0.0
    filters = []
    rois = []
    images = {}
    imids = h5f_ours.keys()
    #random.shuffle(imids)
    rel_types = {}
    rel_types['p']=[]
    rel_types['s']=[]
    rel_types['v']=[]
    rel_types['c']=[]
    for k in m['meta/pre/name2idx'].keys():
        idx = int(str(m['meta/pre/name2idx/'+k][...]))
        r_type = m['meta/pre/name2idx/'+k].attrs['type']
        rel_types[r_type].append(idx)
    for imid in imids:
        obj_boxes_ours = h5f_ours[imid+'/obj_boxes'][...]
        sub_boxes_ours = h5f_ours[imid+'/sub_boxes'][...]
        rlp_labels_ours = h5f_ours[imid+'/rlp_labels'][...].astype(np.int32)
        rlp_confs_ours = h5f_ours[imid+'/rlp_confs'][...]
        sorted_ind = np.argsort(rlp_confs_ours)[::-1]
        rlp_confs_ours = rlp_confs_ours[sorted_ind]
        obj_boxes_ours = obj_boxes_ours[sorted_ind]
        sub_boxes_ours = sub_boxes_ours[sorted_ind]
        rlp_labels_ours = rlp_labels_ours[sorted_ind]
        if rlp_confs_ours.shape[0]<=0: continue

        ours_indices = {}
        ours_indices['p']=[]
        ours_indices['s']=[]
        ours_indices['v']=[]
        ours_indices['c']=[]
        #map indices to type
        for i in xrange(rlp_confs_ours.shape[0]):
            pre_label = rlp_labels_ours[i][1]
            if pre_label in rel_types['p']: ours_indices['p'].append(i)
            if pre_label in rel_types['s']: ours_indices['s'].append(i)
            if pre_label in rel_types['v']: ours_indices['v'].append(i)
            if pre_label in rel_types['c']: ours_indices['c'].append(i)

        im_path = zl.imid2path(m,imid)
        im_orig = cv2.imread(data_root+im_path)
        cv2.namedWindow('ctrl')
        cv2.destroyWindow('ctrl')
        cv2.namedWindow('ctrl')

        ours_p_len = len(ours_indices['p'])-1
        ours_c_len = len(ours_indices['c'])-1
        ours_v_len = len(ours_indices['v'])-1
        ours_s_len = len(ours_indices['s'])-1
        ours_len = len(rlp_labels_ours)-1

        if ours_len>0 :cv2.createTrackbar('idx_ours','ctrl',0,ours_len,nothing)
        if ours_p_len>0 :cv2.createTrackbar('idx_ours_p','ctrl',0,ours_p_len,nothing)
        if ours_c_len>0: cv2.createTrackbar('idx_ours_c','ctrl',0,ours_c_len,nothing)
        if ours_v_len>0:cv2.createTrackbar('idx_ours_v','ctrl',0, ours_v_len,nothing)
        if ours_s_len>0:cv2.createTrackbar('idx_ours_s','ctrl',0, ours_s_len,nothing)


        cnt = 0
        while True:
            cv2.imshow('orig', im_orig)
            if ours_len>=0:
                idx_ours = cv2.getTrackbarPos('idx_ours','ctrl')
                im_ours = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours',im_ours)

            if ours_p_len>=0:
                idx_ours_p = cv2.getTrackbarPos('idx_ours_p','ctrl')
                idx_ours = ours_indices['p'][idx_ours_p]
                im_ours_p = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours_p,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours_p,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours_p,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours_p',im_ours_p)

            if ours_v_len>=0:
                idx_ours_v = cv2.getTrackbarPos('idx_ours_v','ctrl')
                idx_ours = ours_indices['v'][idx_ours_v]
                im_ours_v = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours_v,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours_v,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours_v,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours_v',im_ours_v)

            if ours_c_len>=0:
                idx_ours_c = cv2.getTrackbarPos('idx_ours_c','ctrl')
                idx_ours = ours_indices['c'][idx_ours_c]
                im_ours_c = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours_c,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours_c,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours_c,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours_c',im_ours_c)

            if ours_s_len>=0:
                idx_ours_s = cv2.getTrackbarPos('idx_ours_s','ctrl')
                idx_ours = ours_indices['s'][idx_ours_s]
                im_ours_s = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours_s,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours_s,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours_s,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours_s',im_ours_s)


            c = cv2.waitKey(1)&0xFF
            if c == ord(' '):
                break
            if c == ord('s'):
                im_folder = 'output/results/examples/'+imid
                if not os.path.exists('output/results/examples/'+imid):
                    os.makedirs('output/results/examples/'+imid)
                if not os.path.exists('output/results/examples/'+imid+'/orig_'+imid+'.jpg'):
                    cv2.imwrite('output/results/examples/'+imid+'/orig_'+imid+'.jpg',im_orig)

                if ours_v_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_v_'+imid+'_'+str(idx_ours)+'.jpg',im_ours_v)
                if ours_p_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_p_'+imid+'_'+str(idx_ours)+'.jpg',im_ours_p)
                if ours_c_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_c_'+imid+'_'+str(idx_ours)+'.jpg',im_ours_c)
                if ours_s_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_s_'+imid+'_'+str(idx_ours)+'.jpg',im_ours_s)
                cnt+=1
            if c == 27:
                exit(0)

def run_relation_visualization_zeroshot():
    #h5f= h5py.File('output/sg_vrd_2016_result.classeme.hdf5')
    h5f_lu = h5py.File('output/results/lu_method_results.hdf5')
    #h5f_ours = h5py.File('output/sg_vrd_2016_result_diff_all_5000.hdf5')
    h5f_ours = h5py.File('output/results/ours_retr.hdf5')

    #h5f= h5py.File('output/sg_vrd_2016_result_pre_jointbox_15000.hdf5')
    #h5f= h5py.File('output/sg_vrd_2016_result_pre_concat_14000.hdf5')
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/Data/sg_test_images/'
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    zeroshots = m['meta/zeroshots'][...].astype(np.int32)
    thresh = 0.0
    filters = []
    rois = []
    images = {}
    imids = h5f_lu.keys()
    random.shuffle(imids)
    rel_types = {}
    rel_types['p']=[]
    rel_types['s']=[]
    rel_types['v']=[]
    rel_types['c']=[]
    for k in m['meta/pre/name2idx'].keys():
        idx = int(str(m['meta/pre/name2idx/'+k][...]))
        r_type = m['meta/pre/name2idx/'+k].attrs['type']
        rel_types[r_type].append(idx)
    imcnt=0
    for imid in imids:
        imcnt+=1
        if imcnt%10==0:
            print imcnt
        obj_boxes_lu = h5f_lu[imid+'/obj_boxes'][...]
        sub_boxes_lu = h5f_lu[imid+'/sub_boxes'][...]
        rlp_labels_lu = h5f_lu[imid+'/rlp_labels'][...].astype(np.int32)
        rlp_confs_lu = h5f_lu[imid+'/rlp_confs'][...]
        sorted_ind = np.argsort(rlp_confs_lu)[::-1]
        rlp_confs_lu = rlp_confs_lu[sorted_ind]
        obj_boxes_lu = obj_boxes_lu[sorted_ind]
        sub_boxes_lu = sub_boxes_lu[sorted_ind]
        rlp_labels_lu = rlp_labels_lu[sorted_ind]
        # inds = range(0,len(obj_boxes_lu))
        # obj_boxes_lu = obj_boxes_lu[inds]
        # sub_boxes_lu = sub_boxes_lu[inds]
        # rlp_confs_lu = rlp_confs_lu[inds]
        # inds = np.where(rlp_confs_lu>thresh)
        # obj_boxes_lu = obj_boxes_lu[inds]
        # sub_boxes_lu = sub_boxes_lu[inds]
        # rlp_confs_lu = rlp_confs_lu[inds]
        # rlp_labels_lu = rlp_labels_lu[ind]
        if rlp_confs_lu.shape[0]<=0: continue


        obj_boxes_ours = h5f_ours[imid+'/obj_boxes'][...]
        sub_boxes_ours = h5f_ours[imid+'/sub_boxes'][...]
        rlp_labels_ours = h5f_ours[imid+'/rlp_labels'][...].astype(np.int32)
        rlp_confs_ours = h5f_ours[imid+'/rlp_confs'][...]
        sorted_ind = np.argsort(rlp_confs_ours)[::-1]
        rlp_confs_ours = rlp_confs_ours[sorted_ind]
        obj_boxes_ours = obj_boxes_ours[sorted_ind]
        sub_boxes_ours = sub_boxes_ours[sorted_ind]
        rlp_labels_ours = rlp_labels_ours[sorted_ind]

        if rlp_confs_ours.shape[0]<=0: continue

        zeroshotind_ours = []
        for i in xrange(rlp_labels_ours.shape[0]):
            ind = np.all(zeroshots==rlp_labels_ours[i],axis=1)
            if np.any(ind):
                zeroshotind_ours.append(i)
        if len(zeroshotind_ours)==0:continue
        rlp_confs_ours = rlp_confs_ours[zeroshotind_ours]
        obj_boxes_ours = obj_boxes_ours[zeroshotind_ours]
        sub_boxes_ours = sub_boxes_ours[zeroshotind_ours]
        rlp_labels_ours = rlp_labels_ours[zeroshotind_ours]

        zeroshotind_lu = []
        for i in xrange(rlp_labels_lu.shape[0]):
            ind = np.all(zeroshots==rlp_labels_lu[i],axis=1)
            if np.any(ind):
                zeroshotind_lu.append(i)
        if len(zeroshotind_lu)==0:continue
        rlp_confs_lu = rlp_confs_lu[zeroshotind_lu]
        obj_boxes_lu = obj_boxes_lu[zeroshotind_lu]
        sub_boxes_lu = sub_boxes_lu[zeroshotind_lu]
        rlp_labels_lu = rlp_labels_lu[zeroshotind_lu]

        ours_indices = {}
        ours_indices['p']=[]
        ours_indices['s']=[]
        ours_indices['v']=[]
        ours_indices['c']=[]
        #map indices to type
        for i in xrange(rlp_confs_ours.shape[0]):
            pre_label = rlp_labels_ours[i][1]
            if pre_label in rel_types['p']: ours_indices['p'].append(i)
            if pre_label in rel_types['s']: ours_indices['s'].append(i)
            if pre_label in rel_types['v']: ours_indices['v'].append(i)
            if pre_label in rel_types['c']: ours_indices['c'].append(i)

        lu_indices = {}
        lu_indices['p']=[]
        lu_indices['s']=[]
        lu_indices['v']=[]
        lu_indices['c']=[]
        #map indices to type
        for i in xrange(rlp_confs_lu.shape[0]):
            pre_label = rlp_labels_lu[i][1]
            if pre_label in rel_types['p']: lu_indices['p'].append(i)
            if pre_label in rel_types['s']: lu_indices['s'].append(i)
            if pre_label in rel_types['v']: lu_indices['v'].append(i)
            if pre_label in rel_types['c']: lu_indices['c'].append(i)

        im_orig = cv2.imread(data_root+imid+'.jpg')
        cv2.namedWindow('ctrl')
        cv2.destroyWindow('ctrl')
        cv2.namedWindow('ctrl')

        lu_p_len = len(lu_indices['p'])-1
        lu_c_len = len(lu_indices['c'])-1
        lu_v_len = len(lu_indices['v'])-1
        lu_s_len = len(lu_indices['s'])-1
        lu_len = len(rlp_labels_lu)-1

        if lu_len>0 :cv2.createTrackbar('idx_lu','ctrl',0,lu_len,nothing)
        if lu_p_len>0 :cv2.createTrackbar('idx_lu_p','ctrl',0,lu_p_len,nothing)
        if lu_c_len>0: cv2.createTrackbar('idx_lu_c','ctrl',0,lu_c_len,nothing)
        if lu_v_len>0:cv2.createTrackbar('idx_lu_v','ctrl',0, lu_v_len,nothing)
        if lu_s_len>0:cv2.createTrackbar('idx_lu_s','ctrl',0, lu_s_len,nothing)

        ours_p_len = len(ours_indices['p'])-1
        ours_c_len = len(ours_indices['c'])-1
        ours_v_len = len(ours_indices['v'])-1
        ours_s_len = len(ours_indices['s'])-1
        ours_len = len(rlp_labels_ours)-1

        if ours_len>0 :cv2.createTrackbar('idx_ours','ctrl',0,ours_len,nothing)
        if ours_p_len>0 :cv2.createTrackbar('idx_ours_p','ctrl',0,ours_p_len,nothing)
        if ours_c_len>0: cv2.createTrackbar('idx_ours_c','ctrl',0,ours_c_len,nothing)
        if ours_v_len>0:cv2.createTrackbar('idx_ours_v','ctrl',0, ours_v_len,nothing)
        if ours_s_len>0:cv2.createTrackbar('idx_ours_s','ctrl',0, ours_s_len,nothing)

        # if lu_c_len<0:continue
        # if ours_c_len<0:continue

        cnt = 0
        while True:
            if lu_len>=0:
                idx_lu = cv2.getTrackbarPos('idx_lu','ctrl')
                im_lu = im_orig.copy()
                s_box_lu = sub_boxes_lu[idx_lu]
                o_box_lu = obj_boxes_lu[idx_lu]
                conf =  rlp_confs_lu[idx_lu]
                sub_label_lu = m['meta/cls/idx2name/'+str(rlp_labels_lu[idx_lu][0])][...]
                obj_label_lu = m['meta/cls/idx2name/'+str(rlp_labels_lu[idx_lu][2])][...]
                pre_label_lu = m['meta/pre/idx2name/'+str(rlp_labels_lu[idx_lu][1])][...]
                r_label_lu = '%s %s %s'%(sub_label_lu,pre_label_lu,obj_label_lu)
                cv2.putText(im_lu,r_label_lu,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_lu,(s_box_lu[0],s_box_lu[1]),(s_box_lu[2],s_box_lu[3]),(0,200,0),2)
                cv2.rectangle(im_lu,(o_box_lu[0],o_box_lu[1]),(o_box_lu[2],o_box_lu[3]),(0,0,200),2)
                cv2.imshow('im_lu',im_lu)

            if ours_len>=0:
                idx_ours = cv2.getTrackbarPos('idx_ours','ctrl')
                im_ours = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours',im_ours)

            if ours_p_len>=0:
                idx_ours_p = cv2.getTrackbarPos('idx_ours_p','ctrl')
                idx_ours = ours_indices['p'][idx_ours_p]
                im_ours_p = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours_p,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours_p,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours_p,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours_p',im_ours_p)

            if ours_v_len>=0:
                idx_ours_v = cv2.getTrackbarPos('idx_ours_v','ctrl')
                idx_ours = ours_indices['v'][idx_ours_v]
                im_ours_v = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours_v,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours_v,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours_v,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours_v',im_ours_v)

            if ours_c_len>=0:
                idx_ours_c = cv2.getTrackbarPos('idx_ours_c','ctrl')
                idx_ours = ours_indices['c'][idx_ours_c]
                im_ours_c = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours_c,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours_c,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours_c,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours_c',im_ours_c)

            if ours_s_len>=0:
                idx_ours_s = cv2.getTrackbarPos('idx_ours_s','ctrl')
                idx_ours = ours_indices['s'][idx_ours_s]
                im_ours_s = im_orig.copy()
                s_box_ours = sub_boxes_ours[idx_ours]
                o_box_ours = obj_boxes_ours[idx_ours]
                conf =  rlp_confs_ours[idx_ours]
                sub_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][0])][...]
                obj_label_ours = m['meta/cls/idx2name/'+str(rlp_labels_ours[idx_ours][2])][...]
                pre_label_ours = m['meta/pre/idx2name/'+str(rlp_labels_ours[idx_ours][1])][...]
                r_label_ours = '%s %s %s'%(sub_label_ours,pre_label_ours,obj_label_ours)
                cv2.putText(im_ours_s,r_label_ours,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.rectangle(im_ours_s,(s_box_ours[0],s_box_ours[1]),(s_box_ours[2],s_box_ours[3]),(0,200,0),2)
                cv2.rectangle(im_ours_s,(o_box_ours[0],o_box_ours[1]),(o_box_ours[2],o_box_ours[3]),(0,0,200),2)
                cv2.imshow('im_ours_s',im_ours_s)

            c = cv2.waitKey(1)&0xFF
            if c == ord(' '):
                break
            if c == ord('s'):
                im_folder = 'output/results/examples/'+imid
                if not os.path.exists('output/results/examples/'+imid):
                    os.makedirs('output/results/examples/'+imid)
                if not os.path.exists('output/results/examples/'+imid+'/orig_'+imid+'.jpg'):
                    cv2.imwrite('output/results/examples/'+imid+'/orig_'+imid+'.jpg',im_orig)


                if ours_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_'+imid+str(idx_ours)+'.jpg',im_ours)

                # if lu_v_len>=0:cv2.imwrite('output/results/examples/'+imid+'/lu_v_'+imid+str(idx_lu)+'.jpg',im_lu_v)
                # if lu_p_len>=0:cv2.imwrite('output/results/examples/'+imid+'/lu_p_'+imid+str(idx_lu)+'.jpg',im_lu_p)
                # if lu_c_len>=0:cv2.imwrite('output/results/examples/'+imid+'/lu_c_'+imid+str(idx_lu)+'.jpg',im_lu_c)
                # if lu_s_len>=0:cv2.imwrite('output/results/examples/'+imid+'/lu_s_'+imid+str(idx_lu)+'.jpg',im_lu_s)
                if ours_v_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_v_'+imid+str(idx_ours)+'.jpg',im_ours_v)
                if ours_p_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_p_'+imid+str(idx_ours)+'.jpg',im_ours_p)
                if ours_c_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_c_'+imid+str(idx_ours)+'.jpg',im_ours_c)
                if ours_s_len>=0:cv2.imwrite('output/results/examples/'+imid+'/ours_s_'+imid+str(idx_ours)+'.jpg',im_ours_s)
                # cv2.imwrite('output/results/examples/'+imid+'/lu_'+imid+str(idx_lu)+'.jpg',im_lu)
                # cv2.imwrite('output/results/examples/'+imid+'/ours_'+imid+str(idx_ours)+'.jpg',im_ours)
                # cv2.imwrite('output/results/examples/'+imid+'/ours_v_'+imid+str(idx_ours)+'.jpg',im_ours_v)
                # cv2.imwrite('output/results/examples/'+imid+'/ours_p_'+imid+str(idx_ours)+'.jpg',im_ours_p)
                # cv2.imwrite('output/results/examples/'+imid+'/ours_c_'+imid+str(idx_ours)+'.jpg',im_ours_c)
                # cv2.imwrite('output/results/examples/'+imid+'/ours_s_'+imid+str(idx_ours)+'.jpg',im_ours_s)

                # cv2.imwrite('output/results/examples/'+imid+'/lu_v_'+imid+str(idx_lu)+'.jpg',im_lu_v)
                # cv2.imwrite('output/results/examples/'+imid+'/lu_p_'+imid+str(idx_lu)+'.jpg',im_lu_p)
                # cv2.imwrite('output/results/examples/'+imid+'/lu_c_'+imid+str(idx_lu)+'.jpg',im_lu_c)
                # cv2.imwrite('output/results/examples/'+imid+'/lu_s_'+imid+str(idx_lu)+'.jpg',im_lu_s)
                cnt+=1
            if c == 27:
                exit(0)

def convert_result_mat_to_hdf5():
    data = sio.loadmat('output/relationship_det_result.mat', struct_as_record=False, squeeze_me=True)
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    result = h5py.File('output/results/lu_method_results.hdf5')
    rlp_confs=[]
    rlp_labels=[]
    subbox=[]
    objbox=[]
    for i in xrange(1000):
        sub_bboxes = data['sub_bboxes_ours'][i]
        obj_bboxes = data['obj_bboxes_ours'][i]
        rlp_confs_ours = data['rlp_confs_ours'][i]
        rlp_labels_ours = data['rlp_labels_ours'][i]
        if rlp_labels_ours.shape[0]>0:
            rlp_labels_ours[:,1]-=1
            # rlp_labels_ours[:,0]-=1
            # rlp_labels_ours[:,2]-=1
        #rlp_labels_ours[:,1]-=1
        imid = str(m['db/testidx/'+str(i)][...])
        result.create_dataset(imid+'/rlp_confs',dtype='float16', data=np.array(rlp_confs_ours).astype(np.float16))
        result.create_dataset(imid+'/sub_boxes',dtype='float16', data=np.array(sub_bboxes).astype(np.float16))
        result.create_dataset(imid+'/obj_boxes',dtype='float16', data=np.array(obj_bboxes).astype(np.float16))
        result.create_dataset(imid+'/rlp_labels',dtype='float16', data=np.array(rlp_labels_ours).astype(np.float16))

#convert_result_mat_to_hdf5()
#run_visualization()
run_relation_visualization()
#run_relation_visualization_zeroshot()

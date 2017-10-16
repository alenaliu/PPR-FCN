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
import utils.zl_utils as zl

from utils.cython_bbox import bbox_overlaps
from numpy import linalg as LA
import operator

def get_triplet_stats(split):
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    triplets = {}
    triplets_idx = {}
    for imid in m['gt/%s/'%split].keys():
        rlp_labels = m['gt/%s/%s/rlp_labels'%(split,imid)][...]
        sub_boxes= m['gt/%s/%s/sub_boxes'%(split,imid)][...]
        obj_boxes= m['gt/%s/%s/obj_boxes'%(split,imid)][...]
        for i in xrange(rlp_labels.shape[0]):
            p_idx = rlp_labels[i][1]
            s_idx = rlp_labels[i][0]
            o_idx = rlp_labels[i][2]
            p_str = zl.idx2name_pre(m,p_idx)
            s_str = zl.idx2name_cls(m,s_idx)
            o_str = zl.idx2name_cls(m,o_idx)
            sb = sub_boxes[i]
            ob = obj_boxes[i]
            #triplet = str(sub_cls_idx)+'_'+str(pre_idx)+'_'+str(obj_cls_idx)
            triplet_idx = str(s_idx)+'_'+ str(p_idx)+'_'+str(o_idx)
            triplet = s_str+'_'+p_str+'_'+o_str
            if triplet not in triplets:
                triplets[triplet] = 0
            if triplet_idx not in triplets_idx:
               triplets_idx [triplet_idx] = 0
            triplets[triplet] += 1
            triplets_idx[triplet_idx] += 1

    triplets_sorted = sorted(triplets.items(), key=operator.itemgetter(1),reverse=True)

    triplets_idx_sorted = sorted(triplets_idx.items(), key=operator.itemgetter(1),reverse=True)
    return triplets,triplets_sorted


def save_zeroshots_to_meta():
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    train_triplets,train_triplets_sorted = get_triplet_stats('train')
    test_triplets,test_triplets_sorted = get_triplet_stats('test')
    triplets,triplets_sorted = test_triplets,test_triplets_sorted
    #triplets,triplets_sorted = train_triplets,train_triplets_sorted
    zeroshots=[]
    for i in xrange(len(triplets_sorted)):
        if triplets_sorted[i][0] in train_triplets: continue

        spo = triplets_sorted[i][0].split('_')
        s = zl.name2idx_cls(m,spo[0])
        p = zl.name2idx_pre(m,spo[1])
        o = zl.name2idx_cls(m,spo[2])
        zeroshots.append([s,p,o])
        spo_str = '%s_%s_%s'%(str(s),str(p),str(o))
        print '%s;%s;%s'%(triplets_sorted[i][0],spo_str,triplets_sorted[i][1])
        # if triplets_sorted[i][0] not in train_triplets:
            # print triplets_sorted[i][0],triplets_idx_sorted[i][0],triplets_sorted[i][1]
        # if triplets_sorted[i][0] not in triplets:
            # print test_triplets_sorted[i][0],test_triplets_sorted[i][1]
    m.create_dataset('meta/zeroshots',data=np.array(zeroshots).astype(np.int16))

def visualize_gt_data():
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    split = 'test'
    triplets = {}
    root = 'data/sg_vrd_2016/Data/sg_%s_images/'%split
    for imid in m['gt/%s/'%split].keys():
        im_path = root+imid+'.jpg'
        im = cv2.imread(im_path)
        rlp_labels = m['gt/%s/%s/rlp_labels'%(split,imid)][...]
        sub_boxes= m['gt/%s/%s/sub_boxes'%(split,imid)][...]
        obj_boxes= m['gt/%s/%s/obj_boxes'%(split,imid)][...]
        for i in xrange(rlp_labels.shape[0]):
            p_idx = rlp_labels[i][1]
            s_idx = rlp_labels[i][0]
            o_idx = rlp_labels[i][2]
            p_str = zl.idx2name_pre(m,p_idx)
            s_str = zl.idx2name_cls(m,s_idx)
            o_str = zl.idx2name_cls(m,o_idx)
            sb = sub_boxes[i]
            ob = obj_boxes[i]
            cv2.rectangle(im,(sb[0],sb[1]),(sb[2],sb[3]),(0,200,0),2)
            cv2.rectangle(im,(ob[0],ob[1]),(ob[2],ob[3]),(0,0,200),2)
            #triplet = str(sub_cls_idx)+'_'+str(pre_idx)+'_'+str(obj_cls_idx)
            triplet = s_str+'_'+p_str+'_'+o_str
            print triplet
            if triplet not in triplets:
                triplets[triplet] = 0
            triplets[triplet] += 1
        cv2.imshow('im',im)
        c = cv2.waitKey(0)&0xFF
        if c == 27:
            exit(0)

def count_meta_triplet():
    pass

def get_unique_triplets():
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    triplets = {}
    triplets_idx = {}
    for imid in m['gt/%s/'%split].keys():
        rlp_labels = m['gt/%s/%s/rlp_labels'%(split,imid)][...]
        sub_boxes= m['gt/%s/%s/sub_boxes'%(split,imid)][...]
        obj_boxes= m['gt/%s/%s/obj_boxes'%(split,imid)][...]
        for i in xrange(rlp_labels.shape[0]):
            p_idx = rlp_labels[i][1]
            s_idx = rlp_labels[i][0]
            o_idx = rlp_labels[i][2]
            p_str = zl.idx2name_pre(m,p_idx)
            s_str = zl.idx2name_cls(m,s_idx)
            o_str = zl.idx2name_cls(m,o_idx)
            sb = sub_boxes[i]
            ob = obj_boxes[i]
            #triplet = str(sub_cls_idx)+'_'+str(pre_idx)+'_'+str(obj_cls_idx)
            triplet_idx = str(s_idx)+'_'+ str(p_idx)+'_'+str(o_idx)
            triplet = s_str+'_'+p_str+'_'+o_str
            if triplet not in triplets:
                triplets[triplet] = 0
            if triplet_idx not in triplets_idx:
               triplets_idx [triplet_idx] = 0
            triplets[triplet] += 1
            triplets_idx[triplet_idx] += 1

    triplets_sorted = sorted(triplets.items(), key=operator.itemgetter(1),reverse=True)

    triplets_idx_sorted = sorted(triplets_idx.items(), key=operator.itemgetter(1),reverse=True)
    return triplets,triplets_sorted

def run_retrieval_vp():
    h5f = h5py.File('output/precalc/ours_retr.hdf5')
    retr_meta = zl.load('output/pkl/vr_retr_meta.pkl')
    # print h5f[h5f.keys()[0]].keys()
    # exit(0)
    #h5f = h5py.file('output/results/lu_method_results.hdf5')
    # data_root='/home/zawlin/g/py-faster-rcnn/data/vg1_2_2016/Data/test/'
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    thresh = 0.0
    filters = []
    rois = []
    images = {}
    imids = h5f.keys()
    results = {}
    cnt = 0
    pre = 'teddy bear_sit on_blanket'
    # pre = 'paw_in front of_cat'
    # pre = 'racket_hold by_player'
    r_idx = zl.name2idx_tri(m_vp,pre)
    # sub_lbl = pre.split('_')[0]
    # pre_lbl = pre.split('_')[1]
    # obj_lbl = pre.split('_')[2]
    # sub_idx = zl.name2idx_cls(m,sub_lbl)
    # pre_idx = zl.name2idx_pre(m,pre_lbl)
    # obj_idx = zl.name2idx_cls(m,obj_lbl)
    # rlp_label = np.array([sub_idx,pre_idx,obj_idx]).astype(np.int16)
    results = {}

    for imid in imids:
        if cnt%100==0:
            print cnt,zl.toc()
            zl.tic()
        cnt+=1
        # rlp_labels = h5f[imid+'/labels'][...]
        rlp_confs = h5f[imid+'/confs'][:,r_idx]
        ind = np.argsort(rlp_confs)[::-1]
        rlp_confs = rlp_confs[ind[:5]]
        results[imid] = np.average(rlp_confs)
        if rlp_confs.shape[0]==0:
            results[imid]=0.0
            continue
        zl.tic()
        indexor = np.arange(rlp_labels.shape[0])
        # ind = indexor[np.all(rlp_labels==rlp_label,axis=1)]
        ind = np.where(rlp_labels == r_idx)[0]
        if ind.shape[0]==0:
            results[imid]=0.0
            continue
        confs = rlp_confs[ind]
        results[imid] = np.average(confs)
    results_sorted = zl.sort_dict_by_val(results)
    example_folder = 'output/examples_retr_vg_vp/%s/'%pre
    zl.make_dirs(example_folder)
    cnt = 0
    for imid,v in results_sorted[:200]:
        boxes = h5f[imid+'/boxes'][...]
        rlp_labels = h5f[imid+'/labels'][...]
        rlp_confs = h5f[imid+'/confs'][...]

        # indexor = np.arange(rlp_labels.shape[0])
        # ind = indexor[np.all(rlp_labels==rlp_label,axis=1)]
        ind = np.where(rlp_labels == r_idx)[0]
        if ind.shape[0]!=0:
            boxes = boxes[ind]
            sb = boxes [0]
            # ob = obj_boxes[0]
            im = cv2.imread(data_root+zl.imid2path(m,imid))
            cv2.rectangle(im,(sb[0],sb[1]),(sb[2],sb[3]),(0,200,0),1)
            # cv2.rectangle(im,(ob[0],ob[1]),(ob[2],ob[3]),(0,0,200),1)
            cv2.imwrite(example_folder+str(cnt)+'_%s.jpg'%imid,im)
            #cv2.imshow('im',im)
            cnt += 1
            # if cv2.waitKey(0) & 0xFF ==27:
                # exit(0)

def gen_meta_for_retrieval():
    out_pkl = 'output/pkl/vr_retr_meta.pkl'
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5','r')
    data_root = 'data/sg_vrd_2016/Data/sg_test_images/'

    rlp_labels = []
    files = []
    counts = []
    sub_boxes = []
    obj_boxes = []

    for k in m['gt/test']:
        gt_rlp_labels = m['gt/test'][k]['rlp_labels'][...]
        gt_sub_boxes= m['gt/test'][k]['sub_boxes'][...]
        gt_obj_boxes= m['gt/test'][k]['obj_boxes'][...]
        for i in xrange(gt_rlp_labels.shape[0]):
            gt_rlp_label = gt_rlp_labels[i]
            gt_sub_box = gt_sub_boxes[i]
            gt_obj_box = gt_obj_boxes[i]
            if len(rlp_labels)==0:
                rlp_labels.append(gt_rlp_label.tolist())
                files.append([k])
                sub_boxes.append([gt_sub_box.tolist()])
                obj_boxes.append([gt_obj_box.tolist()])
                counts.append(1)
                continue
            bInd = np.all(gt_rlp_label == rlp_labels,axis=1)
            ind = np.arange(len(rlp_labels))[bInd]
            if len(ind)==0:
                rlp_labels.append(gt_rlp_label.tolist())
                files.append([k])
                counts.append(1)
                sub_boxes.append([gt_sub_box.tolist()])
                obj_boxes.append([gt_obj_box.tolist()])
            else:
                files[ind].append(k)
                counts[ind] = counts[ind]+1
                sub_boxes[ind].append(gt_sub_box.tolist())
                obj_boxes[ind].append(gt_obj_box.tolist())
                # rlp_labels.append(gt_rlp_label.tolist())
                # files.append([k])
                # counts.append(1)
    rlp_labels = np.array(rlp_labels)
    files = np.array(files)
    counts = np.array(counts)
    sub_boxes = np.array(sub_boxes)
    obj_boxes = np.array(obj_boxes)
    ind = np.argsort(counts)[::-1]

    counts = counts[ind]
    files = files[ind]
    rlp_labels = rlp_labels[ind]
    sub_boxes = sub_boxes[ind]
    obj_boxes = obj_boxes[ind]
    print sub_boxes[:4]
    for i in xrange(20):
        rlp_label = rlp_labels[i]
        print files[i]
        s_lbl =zl.idx2name_cls(m,rlp_label[0])
        p_lbl =zl.idx2name_pre(m,rlp_label[1])
        o_lbl =zl.idx2name_cls(m,rlp_label[2])
        print s_lbl,p_lbl,o_lbl
        for j in xrange(len(files[i])):
            s_box = sub_boxes[i][j]
            o_box = obj_boxes[i][j]
            fpath = files[i][j]
            impath = data_root+fpath+'.jpg'
            im = cv2.imread(impath)
            cv2.rectangle(im,(s_box[0],s_box[1]),(s_box[2],s_box[3]),(0,255,0),1)
            cv2.rectangle(im,(o_box[0],o_box[1]),(o_box[2],o_box[3]),(255,0,0),1)
            cv2.imshow('im',im)
            cv2.waitKey(0)
    retr_meta = {'counts':counts,'files':files,'rlp_labels':rlp_labels,'sub_boxes':sub_boxes,
    'obj_boxes':obj_boxes}
    zl.save(out_pkl,retr_meta)
    # for i in xrange(20):
        # s_lbl = zl.idx2name_cls(m,rlp_labels[i][0])
        # p_lbl = zl.idx2name_pre(m,rlp_labels[i][1])
        # o_lbl = zl.idx2name_cls(m,rlp_labels[i][2])
        # print '%s_%s_%s=%d'%(s_lbl,p_lbl,o_lbl,counts[i])
    # print files[19]
def has_overlap(ov_sub,ov_obj):
    for ii in xrange(ov_sub.shape[0]):
        for jj in xrange(ov_sub.shape[1]):
            if ov_sub[ii][jj]>0.5 and ov_obj[ii][jj]>0.5:
                return True
    return False

def run_retrieval_n2():
    #h5_path = 'output/sg_vrd_2016_result_all_19500.hdf5'
    h5_path = 'output/sg_vrd_2016_result_diff_all_5000.hdf5'
    # h5_path= 'output/results/lu_method_results.hdf5'
    # h5_path = 'output/sg_vrd_2016_result.hdf5.dd'
    # h5_path = 'output/results/lu_method_results_max.hdf5'
    #h5_path = 'output/results/lu_visual_method_results.hdf5'

    data_root = 'data/sg_vrd_2016/Data/sg_test_images/'
    m = h5py.File('data/sg_vrd_meta.h5', 'r', 'core')
    gt_cache_path = 'output/cache/sg_vrd_gt_cache.pkl'
    gt_h5f = {}
    if os.path.exists(gt_cache_path):
        print 'load gt from cache'
        gt_h5f = zl.load(gt_cache_path)
    else:
        print 'cacheing gt'
        for k in m['gt/test']:
            gt_h5f[k] = {}
            sub_boxes = m['gt/test'][k]['sub_boxes'][...]
            rlp_labels = m['gt/test'][k]['rlp_labels'][...]
            obj_boxes = m['gt/test'][k]['obj_boxes'][...]
            gt_h5f[k]['sub_boxes'] = sub_boxes
            gt_h5f[k]['obj_boxes'] = obj_boxes
            gt_h5f[k]['rlp_labels'] = rlp_labels
        print 'caching gt done'
        zl.save(gt_cache_path,gt_h5f)
    cache_path = 'output/cache/%s.pkl'%h5_path.split('/')[-1]
    # h5f = h5py.File('output/sg_vrd_2016_result.classeme.hdf5')
    if os.path.exists(cache_path):
        print 'load from cache'
        h5f = zl.load(cache_path)
    else:
        h5_in = h5py.File(h5_path,'r')
        h5f = {}
        print 'preloading data'
        for i in h5_in:
            h5f[i] = {}
            h5f[i]['rlp_labels'] = h5_in[i]['rlp_labels'][...]
            h5f[i]['rlp_confs'] = h5_in[i]['rlp_confs'][...]
            h5f[i]['sub_boxes'] = h5_in[i]['sub_boxes'][...]
            h5f[i]['obj_boxes'] = h5_in[i]['obj_boxes'][...]
        zl.save(cache_path,h5f)
        print 'preloading data done'
    #h5f = h5py.file('output/results/lu_method_results.hdf5')
    retr_meta = zl.load('output/pkl/vr_retr_meta.pkl')
    thresh = 0.0
    images = {}
    imids = h5f.keys()
    results = {}
    cnt = 0
    r_acc_100 = 0
    r_acc_50 = 0

    tp_total = 0
    gt_total = 0
    median = []
    for k in xrange(len(retr_meta['rlp_labels'])):
        if k>1000:
            break
        rlp_label = retr_meta['rlp_labels'][k]
        gt_files = retr_meta['files'][k]

        cnt+=1
        results = {}
        zl.tic()
        for imid in imids:
            rlp_labels = h5f[imid]['rlp_labels']
            rlp_confs = h5f[imid]['rlp_confs']
            if rlp_confs.shape[0]==0:
                results[imid]=0.0
                continue
            indexor = np.arange(rlp_labels.shape[0])
            ind = indexor[np.all(rlp_labels==rlp_label,axis=1)]
            if ind.shape[0]==0:
                results[imid]=0.0
                continue
            confs = rlp_confs[ind]
            results[imid] = np.average(confs)

        results_sorted = zl.sort_dict_by_val(results)
        total_gt = len(gt_files)+0.0
        gt_total+=total_gt+0.0
        tp_50=0.
        tp_100=0.
        found = False
        delay = 0
        s_lbl = zl.idx2name_cls(m,rlp_label[0])
        p_lbl = zl.idx2name_pre(m,rlp_label[1])
        o_lbl = zl.idx2name_cls(m,rlp_label[2])
        lbl_str = '%s_%s_%s'%(s_lbl,p_lbl,o_lbl)
        r_at_k = 5

        ex_cnt = 0
        for i in xrange(len(results_sorted)):
            imid,v = results_sorted[i]
            if found and ex_cnt >=100 and i>=r_at_k:
                break
            cor = imid in gt_files
            if cor:
                if not found:
                    found = True
                    median.append(i)
                if i<r_at_k:
                    tp_100+=1
                    tp_total+=1
                    if i<50:tp_50+=1
            if True:
                cor_or_not = str(cor)
                zl.make_dirs('output/examples_vr_insufficient_gt/'+lbl_str+'/')
                outpath = 'output/examples_vr_insufficient_gt/'+lbl_str+'/'+str(i)+'.jpg'
                if cor :delay=0
                im = cv2.imread(data_root+imid+'.jpg')
                if not cor and ex_cnt<100:
                    cv2.imwrite(outpath,im)
                    ex_cnt+=1
                if delay ==0:
                    cv2.putText(im, cor_or_not, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(im, lbl_str, (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(im, str(i), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.imshow('im',im)
                #c = cv2.waitKey(1)&0xFF
                #if c ==27:
                #    exit(0)
                #if c == ord('s'):
                #    delay = 1-delay
                #if c == ord('c'):
                #    delay = 1
        r_50 = tp_50/r_at_k#total_gt
        r_100 = tp_100/r_at_k#total_gt
        r_acc_50+=r_50
        r_acc_100+=r_100
        med = np.median(median)
        print '%d %f %f %f %f %d %f'%(cnt,r_50,r_100,r_acc_50/cnt,r_acc_100/cnt,med,zl.toc())

def run_retrieval_zeroshot():
    # h5_path = 'output/sg_vrd_2016_result_all_19500.hdf5'
    # h5_path = 'output/sg_vrd_2016_result_diff_all_5000.hdf5'
    # h5_path= 'output/results/lu_method_results.hdf5'
    # h5_path = 'output/sg_vrd_2016_result.hdf5.dd'
    # h5_path = 'output/results/lu_method_results_max.hdf5'
    h5_path = 'output/results/lu_visual_method_results.hdf5'

    data_root = 'data/sg_vrd_2016/Data/sg_test_images/'
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    zeroshots = m['meta/zeroshots'][...]
    gt_cache_path = 'output/cache/sg_vrd_gt_cache.pkl'
    gt_h5f = {}
    np.random.seed(76)
    if os.path.exists(gt_cache_path):
        print 'load gt from cache'
        gt_h5f = zl.load(gt_cache_path)
    else:
        print 'cacheing gt'
        for k in m['gt/test']:
            gt_h5f[k] = {}
            sub_boxes = m['gt/test'][k]['sub_boxes'][...]
            rlp_labels = m['gt/test'][k]['rlp_labels'][...]
            obj_boxes = m['gt/test'][k]['obj_boxes'][...]
            gt_h5f[k]['sub_boxes'] = sub_boxes
            gt_h5f[k]['obj_boxes'] = obj_boxes
            gt_h5f[k]['rlp_labels'] = rlp_labels
        print 'caching gt done'
        zl.save(gt_cache_path,gt_h5f)
    cache_path = 'output/cache/%s.pkl'%h5_path.split('/')[-1]
    # h5f = h5py.File('output/sg_vrd_2016_result.classeme.hdf5')
    if os.path.exists(cache_path):
        print 'load from cache'
        h5f = zl.load(cache_path)
    else:
        h5_in = h5py.File(h5_path,'r')
        h5f = {}
        print 'preloading data'
        for i in h5_in:
            h5f[i] = {}
            h5f[i]['rlp_labels'] = h5_in[i]['rlp_labels'][...]
            h5f[i]['rlp_confs'] = h5_in[i]['rlp_confs'][...]
            h5f[i]['sub_boxes'] = h5_in[i]['sub_boxes'][...]
            h5f[i]['obj_boxes'] = h5_in[i]['obj_boxes'][...]
        zl.save(cache_path,h5f)
        print 'preloading data done'
    #h5f = h5py.file('output/results/lu_method_results.hdf5')
    retr_meta = zl.load('output/pkl/vr_retr_meta.pkl')
    thresh = 0.0
    images = {}
    imids = h5f.keys()
    results = {}
    cnt = 0
    r_acc_100 = 0
    r_acc_50 = 0

    tp_total = 0
    gt_total = 0
    median = []
    for k in xrange(len(retr_meta['rlp_labels'])):
        # if k>1000:
            # break
        rlp_label = retr_meta['rlp_labels'][k]
        if not np.any(np.all(zeroshots==rlp_label,axis=1)): continue
        gt_files = retr_meta['files'][k]

        cnt+=1
        results = {}
        zl.tic()
        for imid in imids:
            rlp_labels = h5f[imid]['rlp_labels']
            rlp_confs = h5f[imid]['rlp_confs']
            if rlp_confs.shape[0]==0:
                results[imid]=0.0
                continue
            indexor = np.arange(rlp_labels.shape[0])
            ind = indexor[np.all(rlp_labels==rlp_label,axis=1)]
            if ind.shape[0]==0:
                results[imid]=0.0
                continue
            confs = rlp_confs[ind]
            results[imid] = np.random.uniform(-1,1)#np.average(confs)

        results_sorted = zl.sort_dict_by_val(results)
        total_gt = len(gt_files)+0.0
        gt_total+=total_gt+0.0
        tp_50=0.
        tp_100=0.
        found = False
        delay = 0
        s_lbl = zl.idx2name_cls(m,rlp_label[0])
        p_lbl = zl.idx2name_pre(m,rlp_label[1])
        o_lbl = zl.idx2name_cls(m,rlp_label[2])
        lbl_str = '%s_%s_%s'%(s_lbl,p_lbl,o_lbl)
        r_at_k = 5
        for i in xrange(len(results_sorted)):
            imid,v = results_sorted[i]
            if found and i>=r_at_k:
                break
            cor = imid in gt_files
            if cor:
                if not found:
                    found = True
                    median.append(i)
                if i<r_at_k:
                    tp_100+=1
                    tp_total+=1
                    if i<50:tp_50+=1
            # if True:
                # cor_or_not = str(cor)
                # if cor :delay=0
                # if delay ==0:
                    # im = cv2.imread(data_root+imid+'.jpg')
                    # cv2.putText(im, cor_or_not, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    # cv2.putText(im, lbl_str, (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    # cv2.putText(im, str(i), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    # cv2.imshow('im',im)
                # c = cv2.waitKey(delay)&0xFF
                # if c ==27:
                    # exit(0)
                # if c == ord('s'):
                    # delay = 1-delay

                # if c == ord('c'):
                    # delay = 1
        r_50 = tp_50/r_at_k#total_gt
        r_100 = tp_100/r_at_k#total_gt
        r_acc_50+=r_50
        r_acc_100+=r_100
        med = np.median(median)
        print '%d %f %f %f %f %d %f'%(cnt,r_50,r_100,r_acc_50/cnt,r_acc_100/cnt,med,zl.toc())

def run_retrieval_vp_v2():
    # h5_path = 'output/sg_vrd_2016_result.classeme.hdf5'
    h5_path = 'output/precalc/sg_vrd_2016_test_nms.7.hdf5'
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
    m_vp = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_vp_meta.h5', 'r', 'core')
    cache_path = 'output/cache/%s.pkl'%h5_path.split('/')[-1]
    data_root='/home/zawlin/data/data_vrd/vrd/sg/Data/sg_test_images/'
    rlp_map = []
    for i in xrange(1,len(m_vp['meta/tri/idx2name'].keys())):
        tri = str(m_vp['meta/tri/idx2name'][str(i)][...])
        s_lbl = tri.split('_')[0]
        p_lbl = tri.split('_')[1]
        o_lbl = tri.split('_')[2]
        rlp_label = [zl.name2idx_cls(m,s_lbl),zl.name2idx_pre(m,p_lbl),zl.name2idx_cls(m,o_lbl)]
        rlp_map.append(rlp_label)
    rlp_map = np.array(rlp_map)
    if os.path.exists(cache_path):
        print 'load from cache'
        h5f = zl.load(cache_path)
    else:
        h5_in = h5py.File(h5_path,'r')
        h5f = {}
        print 'preloading data'
        for i in h5_in:
            h5f[i] = {}
            h5f[i]['labels'] = h5_in[i]['labels'][...]
            h5f[i]['confs'] = h5_in[i]['confs'][...]
            h5f[i]['boxes'] = h5_in[i]['boxes'][...]
        zl.save(cache_path,h5f)
        print 'preloading data done'
    retr_meta = zl.load('output/pkl/vr_retr_meta.pkl')
    thresh = 0.0
    images = {}
    imids = h5f.keys()
    results = {}
    cnt = 0
    r_acc_100 = 0
    r_acc_50 = 0

    tp_total = 0
    gt_total = 0
    median = []
    for k in xrange(len(retr_meta['rlp_labels'])):
        if k>1000:
            break
        cnt+=1
        rlp_label = retr_meta['rlp_labels'][k]
        gt_files = retr_meta['files'][k]
        # print gt_files
        # exit(0)
        # for f in gt_files:
            # impath= zl.imid2path(m,f)
            # print impath
            # im= cv2.imread(data_root+impath)
            # cv2.imshow('im',im)
            # cv2.waitKey(0)
        results = {}
        zl.tic()
        ranks = []
        for imid in imids:
            labels = h5f[imid]['labels']-1
            rlp_confs= h5f[imid]['confs']
            rlp_labels = rlp_map[labels]
            if rlp_labels.shape[0]==0:
                results[imid]=0.0
                continue
            indexor = np.arange(rlp_labels.shape[0])
            ind = indexor[np.all(rlp_labels==rlp_label,axis=1)]
            if ind.shape[0]==0:
                results[imid]=0.0
                continue
            confs = rlp_confs[ind]
            results[imid] = np.average(confs)

        results_sorted = zl.sort_dict_by_val(results)
        total_gt = len(gt_files)+0.0
        gt_total+=total_gt+0.0
        tp_50=0.
        tp_100=0.
        found = False
        s_lbl = zl.idx2name_cls(m,rlp_label[0])
        p_lbl = zl.idx2name_pre(m,rlp_label[1])
        o_lbl = zl.idx2name_cls(m,rlp_label[2])
        lbl_str = '%s_%s_%s'%(s_lbl,p_lbl,o_lbl)

        delay = 0
        for i in xrange(len(results_sorted)):
            imid,v = results_sorted[i]
            impath = zl.imid2path(m,imid)
            if found and i>=5:
                break
            # print gt_files
            cor = imid in gt_files
            if cor :
                if not found:
                    found = True
                    median.append(i)
                if i<5:
                    tp_100+=1
                    tp_total+=1
                    if i<50:tp_50+=1
            if True:
                cor_or_not = str(cor)
                if cor :delay=0
                if delay ==0:
                    im = cv2.imread(data_root+impath)
                    cv2.putText(im, cor_or_not, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(im, lbl_str, (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(im, str(i), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.imshow('im',im)
                c = cv2.waitKey(delay)&0xFF
                if c ==27:
                    exit(0)
                if c == ord('s'):
                    delay = 1-delay
                if c == ord('c'):
                    delay = 1
        r_50 = tp_50/5#total_gt
        r_100 = tp_100/5#total_gt
        r_acc_50+=r_50
        r_acc_100+=r_100
        med = np.median(median)
        print '%d %f %f %f %f %d %f'%(cnt,r_50,r_100,r_acc_50/cnt,r_acc_100/cnt,med,zl.toc())

# run_retrieval_zeroshot()
m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
zeroshots = m['meta/zeroshots'][...]
total_box = 0.
cnt = 0.
for k in m['gt/test']:
    if cnt %1000==0:
        print cnt
    cnt +=1
    sub_boxes = m['gt/test'][k]['sub_boxes'][...]
    obj_boxes = m['gt/test'][k]['obj_boxes'][...]
    rlp_labels = m['gt/test'][k]['rlp_labels'][...]
    for i in xrange(rlp_labels.shape[0]):
        rlp_label = rlp_labels[i]
        if np.any(np.all(zeroshots==rlp_label,axis=1)):
            total_box +=2


    #unique = zl.unique_arr(np.vstack((sub_boxes,obj_boxes)))
    # total_box +=boxes.shape[0]

#print total_box/cnt
run_retrieval_n2()
#run_retrieval_vp_v2()
# m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
# zeroshots = m['meta/zeroshots'][...]
# ta = [35,10,2]
# print np.any(np.all(zeroshots==ta,axis=1))
#run_retrieval_n2()
#run_retrieval_vp_v2()
#gen_meta_for_retrieval()
#run_retrieval_n2()
#visualize_gt_data()
#print_zero_shots()
#run_retrieval()
#print_triplet_info()
#save_zeroshots_to_meta()
#run_retrieval()
#show_retrieval_results()
#print_triplet_info()
#run_retrieval()
# x = np.array([[1,1],[1,2],[2,2],[5,25],[1,1]])
# y = np.bincount(x)
# rlp_labels = [[1,2],[2,3]]
# test = np.array([1,2])
# print np.all(rlp_labels==test,axis=1)
# retr_meta = zl.load('output/pkl/vr_retr_meta.pkl')
# for i in xrange(2000):
    # if retr_meta['counts'][i]<50:
        # print  i
        # print len(retr_meta['counts'])
        # exit(0)


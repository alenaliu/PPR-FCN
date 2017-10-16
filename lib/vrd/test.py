
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
import cv2
from utils.blob import prep_im_for_blob,im_list_to_blob
import utils.zl_utils as zl
import os
def get_vis(feat):
    im_vis = vis_square(feat, index)
    im_vis *= 255
    im_vis = im_vis.astype(np.uint8)
    im_vis_color = cv2.applyColorMap(im_vis, cv2.COLORMAP_JET)
    im_vis_color = cv2.resize(im_vis_color,(500,500))
    return im_vis_color

def union_np(a,b):
    x = np.minimum(a[:,1], b[:,1])
    y = np.minimum(a[:,2], b[:,2])
    x2 = np.maximum(a[:,3],b[:,3])
    y2 = np.maximum(a[:,4],b[:,4])
    return np.concatenate((a[:,0,np.newaxis],x[:,np.newaxis],y[:,np.newaxis],x2[:,np.newaxis],y2[:,np.newaxis]),axis=1)
def vis_square(data, index):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    #dbg
    # normalize data for display, from 1 ~ 0
    data = (data - data.min()) / (data.max() - data.min())
    copydata = data

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)

    # pad with ones (white)
    data = np.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data
    # plt.imshow(data)
    # plt.show()
    # plt.axis('off')
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


def im_detect(net, im_path, sub_boxes,obj_boxes):
    blobs = {'data' : None}
    blobs['data'], im_scales = _get_image_blob(im_path)
    blobs['sub_boxes'] = sub_boxes
    blobs['obj_boxes'] = obj_boxes
    sub_boxes = sub_boxes.reshape((-1,5,1,1))
    obj_boxes = obj_boxes.reshape((-1,5,1,1))
    union_boxes = union_np(sub_boxes,obj_boxes)
    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['sub_boxes'].reshape(sub_boxes.shape[0],5,1,1)
    net.blobs['obj_boxes'].reshape(obj_boxes.shape[0],5,1,1)
    net.blobs['union_boxes'].reshape(union_boxes.shape[0],5,1,1)
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False),
                      'sub_boxes': sub_boxes,
                      'obj_boxes': obj_boxes,
                      'union_boxes':union_boxes
                      }

    blobs_out = net.forward(**forward_kwargs)
    return net.blobs['relation_prob'].data[...]

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    x2 = max(a[2],b[2])
    y2 = max(a[3],b[3])
    return (int(x), int(y), int(x2), int(y2))

def run():
    prototxt = 'models/sg_vrd/rel_iccv/test_iccv_gt.prototxt'
    obj_detector_model='data/models/vrd_rfcn/vrd_resnet50_rfcn_iter_70000.caffemodel'
    relation_model='output/sg_vrd_rfcn/psroi_context_tri_sum_cached_iter_75500.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffe.TEST)
    net.copy_from(obj_detector_model)
    net.copy_from(relation_model)

    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r')
    cnt = 0
    root_img = '/home/zawlin/Dropbox/iccv17_hw/_results_from_zl/img/'
    root_cls = '/home/zawlin/Dropbox/iccv17_hw/_results_from_zl/cls/'
    import glog
    for k in m['gt/test'].keys():
        if not os.path.exists(root_img+k):
            os.makedirs(root_img+k)
        cnt += 1
        glog.info(cnt)
        if cnt >80:
            break
        rlp_labels = m['gt/test/%s/rlp_labels'%k][...]
        sub_boxes  = m['gt/test/%s/sub_boxes'%k][...].astype(np.float)
        obj_boxes  = m['gt/test/%s/obj_boxes'%k][...].astype(np.float)

        if sub_boxes.shape[0]>0:
            zeros = np.zeros((sub_boxes.shape[0],1), dtype=np.float)
            # first index is always zero since we do one image by one image
            sub_boxes = np.concatenate((zeros, sub_boxes),axis=1)
            obj_boxes = np.concatenate((zeros, obj_boxes),axis=1)
            im_path = C.get_sg_vrd_path_test(k)
            im = cv2.imread(im_path)
            for i in xrange(sub_boxes.shape[0]):
                # sb = sub_boxes[i][1:].astype(np.int)
                # ob = obj_boxes[i][1:].astype(np.int)
                rlp = rlp_labels[i]
                rel = zl.idx2name_cls(m,rlp[0])+' '+zl.idx2name_pre(m,rlp[1])+' '+zl.idx2name_cls(m,rlp[2])
                x1,y1,x2,y2 = union(sub_boxes[i][1:],obj_boxes[i][1:])
                cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(im,rel,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                # cv2.rectangle(im,(sb[0],sb[1]),(sb[2],sb[3]),(255,0,0),2)
                # cv2.rectangle(im,(ob[0],ob[1]),(ob[2],ob[3]),(255,0,0),2)
            cv2.imshow('im',im)
            cv2.imwrite(root_img+k+'/_.orig.jpg',im)
            im_detect(net,im_path,sub_boxes,obj_boxes)
            rfcn_sub_rel = net.blobs['rfcn_sub_rel'].data[0]
            rfcn_obj_rel = net.blobs['rfcn_obj_rel'].data[0]
            rfcn_union_rel = net.blobs['rfcn_union_rel'].data[0]

            for pi in xrange(70):
                index = pi
                head, last = index * 9, (index + 1)*9
                feat_sub = rfcn_sub_rel[head:last]
                feat_obj= rfcn_obj_rel[head:last]
                feat_union= rfcn_union_rel[head:last]
                im_vis_sub = get_vis(feat_sub)
                im_vis_obj = get_vis(feat_obj)
                im_vis_union = get_vis(feat_union)

                pre = zl.idx2name_pre(m,pi)
                cv2.imwrite(root_img+k+'/%s_sub.jpg'%pre,im_vis_sub)
                cv2.imwrite(root_img+k+'/%s_obj.jpg'%pre,im_vis_obj)
                cv2.imwrite(root_img+k+'/%s_union.jpg'%pre,im_vis_union)
                if not os.path.exists(root_cls+pre):
                    os.makedirs(root_cls+pre)
                cv2.imwrite(root_cls+pre+'/%s_sub.jpg'%k,im_vis_sub)
                cv2.imwrite(root_cls+pre+'/%s_obj.jpg'%k,im_vis_obj)
                cv2.imwrite(root_cls+pre+'/%s_union.jpg'%k,im_vis_union)

                #cv2.imshow(pre+'sub',im_vis_sub)
                #cv2.imshow(pre+'obj',im_vis_obj)
                #cv2.imshow(pre+'union',im_vis_union)

                #if cv2.waitKey(0)==27:
                #    exit(0)
        else:
            #todo
            #print nothing
            pass

    cv2.waitKey(0)

a = []
a.append([1,2,3])
a[0][1]=323
print a


import caffe
import scipy.io as sio
import os
import cv2
import numpy as np
import yaml
from multiprocessing import Process, Queue
import random
import h5py
import fast_rcnn.bbox_transform

from utils.cython_bbox import bbox_overlaps
import numpy as np
import utils.zl_utils as zl
import glog

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


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


class RelationDatalayer(caffe.Layer):
    def get_minibatch(self):
        blobs = {}
        idx = np.random.choice(len(self.rdata['annotation_train']), self._batch_size)

        # labels_blob = np.zeros(self._batch_size,np.int32)
        visuals = []
        labels = []
        cnt = 0
        while cnt < self._batch_size:
            idx = np.random.choice(len(self.rdata['annotation_train']), 1)
            anno = self.rdata['annotation_train'][idx[0]]
            objs = []
            im_id = anno.filename.split('.')[0]
            if im_id not in self.vgg_data:
                continue

            #r_idx = np.random.choice(self.vgg_data[im_id+'/sub_visual'].shape[0], 1)[0]
            rlp_labels = self.gt_labels[im_id]['rlp_labels']
            for i in xrange(self.vgg_data[im_id]['sub_visual'].shape[0]):
                r_idx = i
                sub_visual = self.vgg_data[im_id]['sub_visual'][r_idx]
                obj_visual = self.vgg_data[im_id]['obj_visual'][r_idx]
                visuals.append(np.hstack((sub_visual, obj_visual)))
                labels.append(self.vgg_data[im_id]['pre_label'][r_idx])

            cnt+=1

        # blobs['visual'] = np.array(visual)
        blobs['visual'] = np.array(visuals)
        #print blobs['visual_s']
        # blobs['classeme'] = np.array(classeme)
        # blobs['location'] = np.array(location)
        blobs['label'] = np.array(labels)

        return blobs

    def setup(self, bottom, top):
        self._cur_idx = 0
        self.rdata = sio.loadmat('data/meta/vrd/annotation_train.mat', struct_as_record=False,squeeze_me=True)
        self.meta = h5py.File('data/sg_vrd_meta.h5', 'r')
        self.gt_labels = {}
        if os.path.exists('output/cache/sg_vrd_gt.pkl'):
            self.gt_labels = zl.load('output/cache/sg_vrd_gt.pkl')
            glog.info('loaded gt data from cache')
        else:
            glog.info( 'Preloading gt')
            zl.tic()
            for k in self.meta['gt/train'].keys():
                rlp_labels = self.meta['gt/train/%s/rlp_labels'%k][...]
                sub_boxes = self.meta['gt/train/%s/sub_boxes'%k][...].astype(np.float)
                obj_boxes = self.meta['gt/train/%s/obj_boxes'%k][...].astype(np.float)
                if sub_boxes.shape[0]>0:
                    zeros = np.zeros((sub_boxes.shape[0],1), dtype=np.float)
                    # first index is always zero since we do one image by one image
                    sub_boxes = np.concatenate((zeros, sub_boxes),axis=1)
                    obj_boxes = np.concatenate((zeros, obj_boxes),axis=1)
                self.gt_labels[k] = {}
                self.gt_labels[k]['rlp_labels']=rlp_labels
                self.gt_labels[k]['sub_boxes']=sub_boxes
                self.gt_labels[k]['obj_boxes']=obj_boxes
            glog.info('done preloading gt %f'%zl.toc())
        vgg_h5 = h5py.File("output/sg_vrd_2016_train_predicate_exp_train.hdf5", 'r', 'core')

        if os.path.exists('output/cache/sg_vrd_2016_train.pkl'):
            self.vgg_data = zl.load('output/cache/sg_vrd_2016_train.pkl')
            glog.info('loaded train data from cache')
        else:
            glog.info('Preloading training data')
            zl.tic()
            self.vgg_data= {}
            for k in vgg_h5.keys():
                sub_visual = vgg_h5[k]['sub_classeme'][...]
                obj_visual = vgg_h5[k]['obj_classeme'][...]
                pre_label = vgg_h5[k]['pre_label'][...]
                self.vgg_data[k]={}
                self.vgg_data[k]['sub_visual']=sub_visual
                self.vgg_data[k]['obj_visual']=obj_visual
                self.vgg_data[k]['pre_label']=pre_label
            glog.info('done preloading training data %f'%zl.toc())
            zl.save('output/cache/sg_vrd_2016_train.pkl',self.vgg_data)
            vgg_h5.close()
        self.meta = h5py.File('data/sg_vrd_meta.h5', 'r', 'core')
        layer_params = yaml.load(self.param_str)

        self._batch_size = layer_params['batch_size']
        self.train_data = []
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        # top[0].reshape(self._batch_size, 4096 * 2 )

        top[0].reshape(self._batch_size, 2*101)
        top[1].reshape(self._batch_size)
        # self._name_to_top_map['visual'] = 0
        # self._name_to_top_map['classeme'] = 0
        self._name_to_top_map['visual'] = 0
        # self._name_to_top_map['location'] = 1
        self._name_to_top_map['label'] = 1

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self.get_minibatch()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

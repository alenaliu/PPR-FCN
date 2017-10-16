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
        # labels_blob = np.zeros(self._batch_size,np.int32)
        visuals = []
        labels = []
        cnt = 0
        while cnt < self._batch_size:

            if self.imidx >=len(self.imids):
                random.shuffle(self.imids)
                self.imidx = 0
            imid = self.imids[self.imidx]
            self.imidx += 1
            objs = []

            r_idx = np.random.choice(self.vgg_data[imid]['sub_visual'].shape[0], 1)[0]
            sub_visual = self.vgg_data[imid]['sub_visual'][r_idx]
            obj_visual = self.vgg_data[imid]['obj_visual'][r_idx]
            visuals.append(np.hstack((sub_visual, obj_visual)))
            labels.append(self.vgg_data[imid]['pre_label'][r_idx])

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
        self.vgg_data = {}
        vgg_h5 = h5py.File("output/precalc/vg1_2_2016_predicate_exp_train.hdf5", 'r' )
        layer_params = yaml.load(self.param_str_)

        self.imids = []
        for k in vgg_h5.keys():
            self.imids.append(k)
        self.imidx = 0

        if os.path.exists('output/cache/vg1_2_2016_pre_train_concat.pkl'):
            self.vgg_data = zl.load('output/cache/vg1_2_2016_pre_train_concat.pkl')
            print 'loaded train data from cache'
        else:
            print 'Preloading training data'
            zl.tick()
            for k in vgg_h5.keys():
                sub_visual  = vgg_h5[k]['sub_visual'][...]
                obj_visual  = vgg_h5[k]['obj_visual'][...]
                pre_label  = vgg_h5[k]['pre_label'][...]
                self.vgg_data[k]={}
                self.vgg_data[k]['sub_visual']=sub_visual
                self.vgg_data[k]['obj_visual']=obj_visual
                self.vgg_data[k]['pre_label']=pre_label
            print 'done preloading training data %f'%zl.tock()
            zl.save('output/cache/vg1_2_2016_pre_train_concat.pkl',self.vgg_data)
            vgg_h5.close()


        self._batch_size = layer_params['batch_size']
        self.train_data = []
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        # top[0].reshape(self._batch_size, 4096 * 2 )

        top[0].reshape(self._batch_size, 2*4096)
        top[1].reshape(self._batch_size)
        self._name_to_top_map['visual'] = 0
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

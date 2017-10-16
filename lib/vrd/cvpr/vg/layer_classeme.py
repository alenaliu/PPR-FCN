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
import random

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

        data = []
        visual = []
        classeme = []
        classeme_s = []
        classeme_o = []
        visual_s = []
        visual_o = []
        loc_s = []
        loc_o = []
        location = []
        labels = []
        cnt = 0
        while cnt < self._batch_size:
            if self.imidx >=len(self.imids):
                random.shuffle(self.imids)
                self.imidx = 0
            imid = self.imids[self.imidx]
            self.imidx += 1
            gt_rlp_labels = self.gt_labels[imid]['rlp_labels']
            gt_sub_boxes= self.gt_labels[imid]['sub_boxes']
            gt_obj_boxes = self.gt_labels[imid]['obj_boxes']

            classemes = self.vgg_data[imid]['classemes']
            visuals = self.vgg_data[imid]['visuals']
            locations = self.vgg_data[imid]['locations']
            cls_confs = self.vgg_data[imid]['cls_confs']
            for i in xrange(gt_rlp_labels.shape[0]):
                gt_rlp_label = gt_rlp_labels[i]
                gt_sub_box = gt_sub_boxes[i]
                gt_obj_box= gt_obj_boxes[i]
                overlaps = bbox_overlaps(
                    np.array([gt_sub_box, gt_obj_box]),
                    locations.astype(np.float))

                if overlaps.shape[0] == 0:
                    continue
                sub_sorted = overlaps[0].argsort()[-30:][::-1]
                obj_sorted = overlaps[1].argsort()[-30:][::-1]
                while len(sub_sorted) > 0 and overlaps[0][sub_sorted[-1]] < .7: sub_sorted = sub_sorted[:-1]
                while len(obj_sorted) > 0 and overlaps[1][obj_sorted[-1]] < .7: obj_sorted = obj_sorted[:-1]
                if len(sub_sorted) <= 0 or len(obj_sorted) <= 0:
                    continue

                for s in sub_sorted[:1]:
                    for o in obj_sorted[:1]:
                        if s != o and cnt < self._batch_size:
                            sub_visual = visuals[s]
                            obj_visual = visuals[o]
                            sub_clsmemes = classemes[s]
                            obj_clsmemes = classemes[o]
                            sub_box_encoded = bbox_transform(np.array([locations[o]]), np.array([locations[s]]))[0]
                            obj_box_encoded = bbox_transform(np.array([locations[s]]), np.array([locations[o]]))[0]
                            pre_lbl = gt_rlp_label[1]
                            labels.append(np.float32(pre_lbl))
                            classeme_s.append(sub_clsmemes)
                            classeme_o.append(obj_clsmemes)
                            visual_s.append(sub_visual)
                            visual_o.append(obj_visual)
                            loc_s.append(sub_box_encoded)
                            loc_o.append(obj_box_encoded)
                            visual.append(np.hstack((sub_visual, obj_visual)))
                            classeme.append(np.hstack((sub_clsmemes, obj_clsmemes)))
                            location.append(np.hstack((sub_box_encoded, obj_box_encoded)))
                            cnt += 1
                if cnt >= self._batch_size:
                    break

        blobs['classeme'] = np.array(classeme)
        # blobs['visual'] = np.array(visual)
        # blobs['location'] = np.array(location)
        blobs['label'] = np.array(labels)

        return blobs

    def setup(self, bottom, top):
        self._cur_idx = 0
        self.vgg_data = {}
        self.gt_labels = {}
        vgg_h5 = h5py.File("output/precalc/vg1_2_2016_train.hdf5", 'r')
        if os.path.exists('output/cache/vg1_2_2016_train.pkl'):
            self.vgg_data = zl.load('output/cache/vg1_2_2016_train.pkl')
            print 'loaded train data from cache'
        else:
            print 'Preloading training data'
            zl.tick()
            for k in vgg_h5.keys():
                classemes = vgg_h5[k]['classemes'][...]
                visuals = vgg_h5[k]['visuals'][...]
                locations = vgg_h5[k]['locations'][...]
                cls_confs = vgg_h5[k]['cls_confs'][...]
                self.vgg_data[k]={}
                self.vgg_data[k]['classemes']=classemes
                self.vgg_data[k]['visuals']=visuals
                self.vgg_data[k]['cls_confs']=cls_confs
                self.vgg_data[k]['locations']=locations
            print 'done preloading training data %f'%zl.tock()
            zl.save('output/cache/vg1_2_2016_train.pkl',self.vgg_data)
            vgg_h5.close()

        self.meta = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5', 'r')
        if os.path.exists('output/cache/vg1_2_2016_gt.pkl'):
            self.gt_labels = zl.load('output/cache/vg1_2_2016_gt.pkl')
            print 'loaded gt data from cache'
        else:
            print 'Preloading gt'
            zl.tick()
            for k in self.meta['gt/train'].keys():
                rlp_labels = self.meta['gt/train/%s/rlp_labels'%k][...]
                sub_boxes = self.meta['gt/train/%s/sub_boxes'%k][...].astype(np.float)
                obj_boxes = self.meta['gt/train/%s/obj_boxes'%k][...].astype(np.float)
                self.gt_labels[k] = {}
                self.gt_labels[k]['rlp_labels']=rlp_labels
                self.gt_labels[k]['sub_boxes']=sub_boxes
                self.gt_labels[k]['obj_boxes']=obj_boxes
            print 'done preloading gt %f'%zl.tock()
            zl.save('output/cache/vg1_2_2016_gt.pkl',self.gt_labels)

        self.imids = []
        for k in self.vgg_data.keys():
            self.imids.append(k)
        self.imidx = 0
        random.shuffle(self.imids)
        layer_params = yaml.load(self.param_str_)

        self._batch_size = layer_params['batch_size']
        self.train_data = []
        self._name_to_top_map = {}

        top[0].reshape(self._batch_size, 201*2)
        # top[1].reshape(self._batch_size, 4096*2)
        # top[2].reshape(self._batch_size, 4*2)
        top[1].reshape(self._batch_size)

        self._name_to_top_map['classeme'] = 0
        # self._name_to_top_map['visual'] = 1
        # self._name_to_top_map['location'] = 2
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

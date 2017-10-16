import pycocotools.coco as coco
import cv2
import numpy as np
import pdb
import os
from multiprocessing import Pool, TimeoutError

import lib
import lib.utils.zl_edge_box as zl_eb
import lib.utils.zl_utils as zl

def do_one_image(file_path):
    root_path = '/media/zawlin/ssd/coco/train2014/'
    edge_box_path = '/media/zawlin/ssd/coco/gen_edgebox/'
    windows = zl_eb.get_windows(root_path+file_path,maxboxes=4000,minscore=0.001)
    zl.save(edge_box_path+file_path+'.eb',windows)

def process_coco_edge_box():
    pool = Pool(processes=8)
    db = coco.COCO('/media/zawlin/ssd/coco/annotations/captions_train2014.json')
    cnt =0
    for k in db.imgs:
        cnt += 1
        im = db.imgs[k]
        pool.apply_async(do_one_image, (im['file_name'],))
    pool.close()
    pool.join()

process_coco_edge_box()

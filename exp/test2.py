import _init_paths
from pycocotools.coco import COCO
import cv2
import numpy as np
import pdb
import os
from multiprocessing import Pool, TimeoutError

import utils.zl_edge_box as zl_eb
import utils.zl_utils as zl
import zl_config as C

def test_coco_api():
    coco = COCO('/media/zawlin/ssd/coco/annotations/instances_train2014.json')
    cats = coco.loadCats(coco.getCatIds())
    print dir(cats)
    nms=[cat['name'] for cat in cats]

    # catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
    # imgIds = coco.getImgIds(catIds=catIds )
    imgs = coco.loadImgs()
    for img in imgs:
        im_path = C.coco_dir+'train2014/'+img['file_name']
        im = cv2.imread(im_path)

        annIds = coco.getAnnIds(imgIds=img['id'],)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            bbox = np.array(ann['bbox']).astype(np.int32)
            zl.tic()
            cats = coco.loadCats(ann['category_id'])[0]['name']
            print zl.toc()

            bbox[2],bbox[3] = bbox[0]+bbox[2],bbox[1]+bbox[3]
            cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),2)
            cv2.putText(im,str(cats),(bbox[0],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
        cv2.imshow('im',im)
        if cv2.waitKey(0)==27:
            exit(0)
    # zl.save(C.cache_path+'coco.pkl',coco)
    #print cats


zl.tic()
test_coco_api()
#coco =zl.load(C.cache_path+'coco.pkl')
print zl.toc()

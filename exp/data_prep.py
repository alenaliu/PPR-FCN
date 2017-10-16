import _init_paths
import utils.zl_utils as zl
import h5py
import zl_config as C
import numpy as np
def merge_pickled_files():
    import os
    h5f = h5py.File(C.coco_eb_h5_path,'w')
    cnt = 0
    zl.tick()
    for path, subdirs, files in os.walk(C.coco_eb_dir):
        for name in files:
            cnt += 1
            if cnt %1000==0:
                print cnt,zl.tock()
                zl.tick()
            fpath = os.path.join(path, name)
            fid = name.replace('.eb','')
            bbs = np.array(zl.load(fpath)).astype(np.float16)
            h5f[fid]=bbs

merge_pickled_files()

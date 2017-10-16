import _init_paths
import h5py as h5
import utils.zl_utils as zl
import h5py
import os
import numpy as np
def prep_train():
    cnt = 0
    root = 'data/sg_vrd_2016/EB/sg_train_images'
    h5path =  'data/sg_vrd_2016/EB/eb2.h5'
    h5f  = h5py.File(h5path,driver='core')
    data = {}
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt += 1
            if cnt % 100==0:
                print cnt
            fpath = os.path.join(path, name)
            fname = name[:-3]

            eb_data = zl.load(fpath)
            eb_data = np.array(eb_data).astype(np.float16)
            data[fname]=eb_data
            # h5f.create_dataset('train/%s'%fname,dtype='float1', data=eb_data)
            # exit(0)
    for fname in data.keys():
        h5f['train/%s'%fname] = data[fname]

def prep_test():
    root = 'data/sg_vrd_2016/EB/sg_test_images'
    h5path =  'data/sg_vrd_2016/EB/eb2.h5'
    h5f  = h5py.File(h5path,driver='core')
    cnt = 0

    data = {}
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt += 1
            if cnt % 100==0:
                print cnt
            fpath = os.path.join(path, name)
            fname = name[:-3]

            eb_data = zl.load(fpath)
            eb_data = np.array(eb_data).astype(np.float16)
            data[fname] = eb_data
            # h5f.create_dataset('train/%s'%fname,dtype='float1', data=eb_data)
            # exit(0)

    for fname in data.keys():
        h5f['test/%s'%fname] = data[fname]
#prep_train()
prep_test()
def load_test():
    h5path = '/media/zawlin/ssd/iccv2017/data/voc/gen_eb.h5'
    h5f  = h5py.File(h5path,driver='core')
    data = {}
    cnt = 0
    zl.tic()

    for i in h5f['train/']:
        cnt += 1
        if cnt %100==0:
            print zl.toc(),cnt
            zl.tic()

        data=h5f['train/%s'%i][...].astype(np.float32)
        print data[:,-2]
        idx = np.argsort(data[:,-2],axis=0)
        print data[idx][::-1]
        exit(0)

#load_test()

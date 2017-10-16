import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

repo_root = '/home/zawlin/g/iccv2017'
add_path('%s/edges/cpp/build'%repo_root)
add_path('%s/lib/'%repo_root)
add_path('%s/tools/'%repo_root)
# add_path('%s/externals/py-faster-rcnn/'%repo_root)
# add_path('%s/externals/py-faster-rcnn/caffe-fast-rcnn/python'%repo_root)
add_path('%s/externals/py-R-FCN/'%repo_root)
add_path('%s/externals/py-R-FCN/lib/'%repo_root)
add_path('%s/externals/ms_caffe/python'%repo_root)

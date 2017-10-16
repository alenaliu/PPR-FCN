import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# add_path('%s/externals/py-faster-rcnn/'%repo_root)
# add_path('%s/externals/py-faster-rcnn/caffe-fast-rcnn/python'%repo_root)
add_path('/home/zawlin/g/py-faster-rcnn/lib')
add_path('/home/zawlin/g/py-faster-rcnn/ms_caffe/python')


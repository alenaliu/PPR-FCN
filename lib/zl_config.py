import os

root_dir = '/media/zawlin/ssd/iccv2017/'
coco_dir = root_dir+'data/coco/'
coco_eb_dir = coco_dir + '_gen/eb/train2014/'
coco_eb_h5_path = root_dir + 'meta/coco/train2014_eb.hf5'
word2vec_wiki_path = root_dir + 'models/word2vec/en_1000_no_stem/en.model'
word2vec_news_path = root_dir + 'models/word2vec/GoogleNews-vectors-negative300.bin'
cache_path = root_dir + '_cache/'
imdb_name = ''

sg_vrd_im_root_train = 'data/sg_vrd_2016/Data/sg_train_images/'
sg_vrd_im_root_test= 'data/sg_vrd_2016/Data/sg_test_images/'

vg_im_root_train = 'data/vg1_2_2016/Data/train/'
vg_im_root_test= 'data/vg1_2_2016/Data/test/'
def get_roidb_cache_path(imdb_name):
    return cache_path + imdb_name + '.roidb.pkl'

def get_roidb_normed_cache_path(imdb_name):
    return cache_path + imdb_name + '.roidb_normed.pkl'

def get_imdb_cache_path(imdb_name):
    return cache_path + imdb_name + '.imdb.pkl'

def get_sg_vrd_path_train(imid):
    return sg_vrd_im_root_train+imid+'.jpg'

def get_sg_vrd_path_test(imid):
    return sg_vrd_im_root_test+imid+'.jpg'

def get_vg_path_train(imid):
    return vg_im_root_train+imid

def get_vg_path_test(imid):
    return vg_im_root_test+imid


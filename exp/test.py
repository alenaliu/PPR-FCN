import gensim
import _init_paths
import utils.zl_utils as zl
from numpy.random import randn
from sklearn.decomposition import sparse_encode
import numpy as np
from sklearn.decomposition import SparseCoder
import pycocotools.coco as coco
import random
import zl_config as C
n = m = 100 # dimensions of our input
input_x = randn(n, m)

def get_ilsvrc_word_list():
    ret = []
    lines = [line.strip() for line in open('ilsvrc_det_wordlist.txt')]
    for l in lines:
        ret.append(l.split(' ')[2])
    return ret

def save_ilsvrc_vectors_to_pickle():
    ilsvrc_words = get_ilsvrc_word_list()
    model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    word2vec = {}
    for w in ilsvrc_words:
        w = w.replace('_',' ')
        if w in model:
            word2vec[w] = model[w]
        else:
            print '%s not in model'%w
    zl.save('ilsvrc_word2vec',word2vec)

def save_test_words_to_pickle():
    ilsvrc_words =  ['antelope','cup','ball','cooking']
    model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    word2vec = {}
    for w in ilsvrc_words:
        w = w.replace('_',' ')
        word2vec[w] = model[w]
    zl.save('test_word2vec',word2vec)

def test_sparse_coder():
    db = coco.COCO('/media/zawlin/ssd/coco/annotations/captions_train2014.json')
    ilsvrc_word2vec = zl.load('ilsvrc_word2vec')
    test_word2vec = zl.load('test_word2vec')
    D = []
    idx2word = []
    for i in xrange(len(ilsvrc_word2vec.keys())):
        idx2word.append(ilsvrc_word2vec.keys()[i])
        D.append(ilsvrc_word2vec[ilsvrc_word2vec.keys()[i]])
    idx2word = np.array(idx2word)
    D = np.array(D)
    print 'loading word2vec'
    model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=5,
                            transform_alpha=None, transform_algorithm='lasso_cd')
    #random.shuffle(db.anns)
    for k in db.anns:
        cap = db.anns[k]['caption']
        splited = cap.split(' ')
        for s in splited:
            if s.lower() in model:
                y = model[s.lower()]
                x  = coder.transform(y.reshape(1,-1))[0]
                print '%s = %s'%(s.lower(),idx2word[np.argsort(x)[::-1]])
                print x[np.argsort(x)[::-1]]
        c = raw_input('press q to quit')
        if c== 'q':
            exit(0)

def test_coco():
    db = coco.COCO('/media/zawlin/ssd/coco/annotations/captions_train2014.json')
    for k in db.imgs:
        print db.imgs[k]
        exit(0)
    pass

def test_word2vec_wiki():
    ilsvrc_words = get_ilsvrc_word_list()
    model = gensim.models.Word2Vec.load(C.word2vec_wiki_path)
    print dir(model.vocab.keys())
    print len(model.vocab.keys())
    word2vec = {}
    for w in ilsvrc_words:
        w = w.replace('_',' ')
        if w in model:
            word2vec[w] = model[w]
        else:
            print '%s not in model'%w

#test_coco()
#test_sparse_coder()
#test_word2vec_wiki()
#save_test_words_to_pickle()
#save_ilsvrc_vectors_to_pickle()
# loaded = zl.load('ilsvrc_word2vec')
# print loaded.keys()

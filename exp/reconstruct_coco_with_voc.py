import _init_paths
from pycocotools.coco import COCO
import zl_config as C
def recon():
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

    coco = COCO('/media/zawlin/ssd/coco/annotations/instances_train2014.json')
    cats = coco.loadCats(coco.getCatIds())
    coco_classes=[cat['name'] for cat in cats]

    model = gensim.models.Word2Vec.load_word2vec_format(C.word2vec_news_path)
    voc_v = {}
    coco_v = {}

    for n in voc_classes:
        voc_v[n] = model[n]
    for n in coco_classes:
        coco_v[n] = model[n]
    zl.save(C.cache_path+'word2vec',{'voc':voc_v,'coco':coco_v})

recon()

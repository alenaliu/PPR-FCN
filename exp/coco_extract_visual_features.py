import caffe
def extract_vis_features_coco():
    net = caffe.Net('models/sg_vrd/vgg16/faster_rcnn_end2end/test_jointbox.prototxt',
                    'output/faster_rcnn_end2end/sg_vrd_2016_train/vgg16_faster_rcnn_finetune_iter_40000.caffemodel',caffe.TEST)
    # net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    net.name = 'sgvrd'
    pass

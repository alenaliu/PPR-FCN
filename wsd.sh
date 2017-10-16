#!/bin/sh
#python tools/train_net.py --gpu 0 --solver models/sg_vrd/wsd/solver_eb_binary_log.prototxt --imdb voc_0712_train   --iters 700000   --cfg experiments/cfgs/rfcn_end2end_iccv_eb.yml --weights data/rfcn_models/resnet50_rfcn_final.caffemodel
python tools/train_net.py --gpu 1 --solver models/sg_vrd/wsd/solver_eb_wsddn.prototxt --imdb voc_0712_train   --iters 700000   --cfg experiments/cfgs/rfcn_end2end_iccv_eb.yml --weights data/imagenet/ResNet-50-model.caffemodel 2>&1|tee -a logs/wsd.log
#python tools/train_net.py --gpu 0 --solver models/sg_vrd/wsd/solver_eb_wsddn.prototxt --imdb voc_0712_train   --iters 700000   --cfg experiments/cfgs/rfcn_end2end_iccv_eb.yml --weights data/rfcn_models/resnet50_rfcn_final.caffemodel 2>&1|tee -a logs/wsd.log

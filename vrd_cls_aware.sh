#!/bin/sh
python -u tools/train_net.py --gpu 0 --solver models/sg_vrd/resnet50/solver_cls_aware.prototxt --weights data/imagenet/ResNet-50-model.caffemodel --imdb  sg_vrd_2016_train --iters 700000   --cfg experiments/cfgs/rfcn_end2end_cls_aware.yml 2>&1|tee -a logs/vrd_cls_aware.log

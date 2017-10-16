#!/bin/sh
python -u tools/train_net.py --gpu 0 --solver models/sg_vrd/resnet50/solver.prototxt --weights data/imagenet/ResNet-50-model.caffemodel --imdb  sg_vrd_2016_train --iters 700000   --cfg experiments/cfgs/rfcn_end2end.yml 2>&1|tee -a logs/vrd.log
#python -u tools/train_net.py --gpu 0 --solver models/sg_vrd/resnet50/solver.prototxt --weights output/rfcn_end2end/sg_vrd_2016_train/vrd_resnet50_rfcn_iter_65000.caffemodel --imdb  sg_vrd_2016_train --iters 700000   --cfg experiments/cfgs/rfcn_end2end.yml 2>&1|tee -a logs/vrd.log

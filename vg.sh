#!/bin/sh
python -u tools/train_net.py --gpu 0 --solver models/vg1_2/resnet50/solver.prototxt --weights data/imagenet/ResNet-50-model.caffemodel --imdb  vg1_2_2016_train --iters 700000   --cfg experiments/cfgs/rfcn_end2end.yml 2>&1|tee -a logs/vg.log

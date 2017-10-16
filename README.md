# PPR-FCN
> This is the source code for the paper: [PPR-FCN: Weakly Supervised Visual Relation Detection via Parallel Pairwise R-FCN](https://arxiv.org/abs/1708.01956)


### Dependencies
This code is built on top of R-FCN. Please carefully read through [py-R-FCN](https://github.com/YuwenXiong/py-R-FCN) and make sure py-R-FCN can run within your enviornment. 
Note: The evaluation scripts of our model are written in MATLAB. 

### Dataset/models dependencies
As discussed in the paper, we used VG and VRD as our two datasets. They can be found here: [VG](http://visualgenome.org/) and [VRD](http://cs.stanford.edu/people/ranjaykrishna/vrd/). 

Certain models are used for training. Please review the readme file under ./data

### Usage
#### Training
Prior to training, go to ./lib and modify zl_config.py. Change the file paths to your correct paths.

Then, simply call wsd.sh, vg.sh, vrd.sh, vrd_cls_aware.sh and subsequence batch files under the root directory, by following the steps discussed in the paper.

#### Evaluate
We provide evaluation in MATLAB. Navigate to ./_evaluation folder. Then under/vg(or /vr). Run zl_eval.m to get the evaluation result.

### License

PPR-FCN is released under the MIT License (refer to the LICENSE file for details).

### Citing PPR-FCN

If you find PPR-FCN useful in your research, please consider citing:

```
@article{DBLP:journals/corr/abs-1708-01956,
  author    = {Hanwang Zhang and
               Zawlin Kyaw and
               Jinyang Yu and
               Shih{-}Fu Chang},
  title     = {{PPR-FCN:} Weakly Supervised Visual Relation Detection via Parallel
               Pairwise {R-FCN}},
  journal   = {CoRR},
  volume    = {abs/1708.01956},
  year      = {2017},
  url       = {http://arxiv.org/abs/1708.01956}
}
```

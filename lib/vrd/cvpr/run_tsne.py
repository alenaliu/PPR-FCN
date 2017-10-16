import numpy as np
from skdata.mnist.views import OfficialImageClassification
from matplotlib import pyplot as plt
from tsne import bh_sne

import _init_paths
import caffe
import h5py
m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
#print m['meta/pre/name2idx'].keys()
keys = []
for k in m['meta/pre/idx2name'].keys():
    keys.append(int(k))
keys = sorted(keys)
for k in keys:
    print str(k)+','+m['meta/pre/idx2name/'+str(k)][...]
labels = []
for k in keys:
    labels.append(m['meta/pre/idx2name/'+str(k)][...])

model_type = 'pre_diff'
iteration = '5000'
# model_type = 'pre_diff'
# iteration = '5000'
net = caffe.Net('models/sg_vrd/relation/test_'+model_type+'.prototxt','output/relation/vr/sg_vrd_relation_vgg16_'+model_type+'_iter_'+iteration+'.caffemodel',caffe.TEST)

embeddings = net.params['relation'][0].data
# load up data
#data = OfficialImageClassification(x_dtype="float32")
x_data = embeddings
y_data = np.array(range(0,70))
# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))

# For speed of computation, only run on a subset
x_data = x_data
y_data = y_data

# perform t-SNE embedding
vis_data = bh_sne(x_data,perplexity=1.5)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

fig, ax = plt.subplots()
ax.scatter(vis_x,vis_y,color='red',s=25)
# plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)


# fig, ax = plt.subplots()

for i, txt in enumerate(labels):
    ax.annotate(txt, (vis_x[i],vis_y[i]),fontsize=20)
plt.show()

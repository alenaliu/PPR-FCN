#include <vector>

#include "caffe/layers/binary_log_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BinaryLogLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Backward_cpu(top,propagate_down,bottom);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(BinaryLogLossLayer);


}  // namespace caffe

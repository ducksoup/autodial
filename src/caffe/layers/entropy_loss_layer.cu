#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EntropyLossForwardGPU(const int nthreads,
          const Dtype* prob_data, Dtype* entropy, const int num, const int dim,
          const int spatial_dim) {
  const int channels = dim / spatial_dim;
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    for (int c = 0; c < channels; ++c) {
      const Dtype p = prob_data[n * dim + c * spatial_dim + s];
      entropy[n * spatial_dim + s] -= p * log(max(p, Dtype(FLT_MIN)));
    }
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  Dtype* entropy_data = entropy_.mutable_gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Init buffers
  caffe_gpu_set(nthreads, Dtype(0), entropy_data);
  // Compute element-wise entropy
  // NOLINT_NEXT_LINE(whitespace/operators)
  EntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, entropy_data, outer_num_,
      dim, inner_num_);
  Dtype loss = 0;
  caffe_gpu_asum(nthreads, entropy_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_);
}

template <typename Dtype>
__global__ void EntropyLossBackwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* entropy_data, Dtype* bottom_diff,
          const int num, const int dim, const int spatial_dim) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const Dtype H = entropy_data[n * spatial_dim + s];
    for (int c = 0; c < channels; ++c) {
      const Dtype p = prob_data[n * dim + c * spatial_dim + s];
      bottom_diff[n * dim + c * spatial_dim + s] = Dtype(-1.) * p * (
          H + log(max(p, Dtype(FLT_MIN))));
    }
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* entropy_data = entropy_.gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    EntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, entropy_data,
        bottom_diff, outer_num_, dim, inner_num_);
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_);
    caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyLossLayer);

}  // namespace caffe

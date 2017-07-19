#include <algorithm>
#include <vector>

#include "caffe/layers/dial_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  
  // Clip alpha in the range [0.5, 1]
  if (this->blobs_[0]->cpu_data()[0] < 0.5)
    this->blobs_[0]->mutable_cpu_data()[0] = 0.5;
  if (this->blobs_[0]->cpu_data()[0] > 1)
    this->blobs_[0]->mutable_cpu_data()[0] = 1;
  const Dtype alpha = this->blobs_[0]->cpu_data()[0];
  
  if (use_global_stats_) {
    if (bottom[0] != top[0]) {
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }
    
    // Retrieve stored stats
    const Dtype scale_factor = this->blobs_[3]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[3]->cpu_data()[0];
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->gpu_data(), mean_.mutable_gpu_data());
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[2]->gpu_data(), variance_.mutable_gpu_data());
    
    // Select between source / target stats
    const Dtype* mean = mean_.gpu_data() +
        (test_stats_ == DialParameter_TestStats_TARGET) * channels_;
    Dtype* variance = variance_.mutable_gpu_data() +
        (test_stats_ == DialParameter_TestStats_TARGET) * channels_;
    
    // Subtract mean
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.gpu_data(), mean, 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, -1, num_by_chans_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 1., top_data);
    
    // Normalize variance
    caffe_gpu_add_scalar(channels_, eps_, variance);
    caffe_gpu_powx(channels_, variance, Dtype(0.5), variance);

    // replicate variance to input size and do division
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.gpu_data(), variance, 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, 1., num_by_chans_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
    caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
    return;
  }
  
  // copy everything to x_norm variables
  caffe_copy(bottom[0]->count(), bottom_data, x_norm_src_.mutable_gpu_data());
  caffe_copy(bottom[0]->count(), bottom_data, x_norm_tgt_.mutable_gpu_data());

  // Find pointers to src and tgt data
  int offset = slice_point_ * channels_ * spatial_dim;
  const Dtype* src_data = bottom_data;
  const Dtype* tgt_data = src_data + offset;
  Dtype* src_top = top_data;
  Dtype* tgt_top = src_top + offset;
  int src_num = slice_point_;
  int tgt_num = num - src_num;
  Dtype* src_mean = mean_.mutable_gpu_data();
  Dtype* tgt_mean = src_mean + channels_;
  Dtype* src_variance = variance_.mutable_gpu_data();
  Dtype* tgt_variance = src_variance + channels_;
  
  // compute means
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * src_num, spatial_dim,
      1. / (src_num * spatial_dim), src_data,
      spatial_sum_multiplier_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, src_num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0., src_mean);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * tgt_num, spatial_dim,
      1. / (tgt_num * spatial_dim), tgt_data,
      spatial_sum_multiplier_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, tgt_num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0., tgt_mean);
  
  // combine with weight
  caffe_copy(mean_.count(), src_mean, src_variance);
  caffe_gpu_axpby(channels_, Dtype(1. - alpha), tgt_variance, alpha, src_mean);
  caffe_gpu_axpby(channels_, Dtype(1. - alpha), src_variance, alpha, tgt_mean);
  
  // subtract src mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), src_mean, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., x_norm_src_.mutable_gpu_data());
  
  // subtract tgt mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), tgt_mean, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., x_norm_tgt_.mutable_gpu_data());
  
  // compute src variance
  caffe_gpu_powx(temp_.count(), x_norm_src_.gpu_data(), Dtype(2),
      temp_.mutable_gpu_data());
  caffe_gpu_scal<Dtype>(src_num * channels_ * spatial_dim, alpha /
      (src_num * spatial_dim), temp_.mutable_gpu_data());
  caffe_gpu_scal<Dtype>(tgt_num * channels_ * spatial_dim, (1. - alpha) /
      (tgt_num * spatial_dim), temp_.mutable_gpu_data() + offset);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
      1., temp_.gpu_data(), spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      src_variance);
  
  // compute tgt variance
  caffe_gpu_powx(temp_.count(), x_norm_tgt_.gpu_data(), Dtype(2),
      temp_.mutable_gpu_data());
  caffe_gpu_scal<Dtype>(src_num * channels_ * spatial_dim, (1. - alpha) /
      (src_num * spatial_dim), temp_.mutable_gpu_data());
  caffe_gpu_scal<Dtype>(tgt_num * channels_ * spatial_dim, alpha /
      (tgt_num * spatial_dim), temp_.mutable_gpu_data() + offset);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
      1., temp_.gpu_data(), spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      tgt_variance);
  
  // compute and save moving average
  this->blobs_[3]->mutable_cpu_data()[0] *= moving_average_fraction_;
  this->blobs_[3]->mutable_cpu_data()[0] += 1;
  caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
      moving_average_fraction_, this->blobs_[1]->mutable_gpu_data());
  int m = bottom[0]->count()/channels_;
  Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
  caffe_gpu_axpby(variance_.count(), bias_correction_factor,
      variance_.gpu_data(), moving_average_fraction_,
      this->blobs_[2]->mutable_gpu_data());
  
  // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
             variance_.mutable_gpu_data());
  
  // divide by src variance
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), src_variance, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), x_norm_src_.gpu_data(), temp_.gpu_data(),
      x_norm_src_.mutable_gpu_data());
  
  // divide by tgt variance
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), tgt_variance, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), x_norm_tgt_.gpu_data(), temp_.gpu_data(),
      x_norm_tgt_.mutable_gpu_data());
  
  // copy to output
  caffe_copy(src_num * channels_ * spatial_dim,
             x_norm_src_.gpu_data(), src_top);
  caffe_copy(tgt_num * channels_ * spatial_dim,
             x_norm_tgt_.gpu_data() + offset, tgt_top);
}

template <typename Dtype>
__global__ void DialWeightBackwardGPU(const int nthreads,
          const Dtype* x_norm_src, const Dtype* x_norm_tgt,
          const Dtype* dys, const Dtype* ys_dys,
          const Dtype* dyt, const Dtype* yt_dyt, Dtype* out,
          int src_num, int tgt_num, int channels) {
  int spatial_dim = nthreads / (src_num + tgt_num);
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / spatial_dim;
    const int j = index % spatial_dim;
    
    for (int c = 0; c < channels; ++c) {
      const Dtype ys = x_norm_src[(i * channels + c) * spatial_dim + j];
      const Dtype yt = x_norm_tgt[(i * channels + c) * spatial_dim + j];
      Dtype dw = 0;
      if (i < src_num) {
        dw += -1. / (src_num * spatial_dim) * ys * dys[c];
        dw += -0.5 / (src_num * spatial_dim) * ys * ys * ys_dys[c];
        dw += 1. / (src_num * spatial_dim) * yt * dyt[c];
        dw += 0.5 / (src_num * spatial_dim) * yt * yt * yt_dyt[c];
      } else {
        dw += 1. / (tgt_num * spatial_dim) * ys * dys[c];
        dw += 0.5 / (tgt_num * spatial_dim) * ys * ys * ys_dys[c];
        dw += -1. / (tgt_num * spatial_dim) * yt * dyt[c];
        dw += -0.5 / (tgt_num * spatial_dim) * yt * yt * yt_dyt[c];
      }
      out[(i * channels + c) * spatial_dim + j] = dw;
    }
  }
}

template <typename Dtype>
void DialLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  
  if (use_global_stats_) {
    caffe_gpu_div(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
    return;
  }
  
  // Dimensions
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  int offset = slice_point_ * channels_ * spatial_dim;
  int src_num = slice_point_;
  int tgt_num = num - src_num;
  
  // Pointers
  const Dtype* x_norm_src = x_norm_src_.gpu_data();
  const Dtype* x_norm_tgt = x_norm_tgt_.gpu_data();
  Dtype* x_norm_src_d = x_norm_src_.mutable_gpu_diff();
  Dtype* x_norm_tgt_d = x_norm_tgt_.mutable_gpu_diff();
  const Dtype* src_top_diff = top_diff;
  const Dtype* tgt_top_diff = top_diff + offset;
  Dtype* src_mean = mean_.mutable_gpu_data();
  Dtype* tgt_mean = src_mean + channels_;
  Dtype* src_mean_d = mean_.mutable_gpu_diff();
  Dtype* tgt_mean_d = src_mean_d + channels_;
  Dtype* src_variance = variance_.mutable_gpu_data();
  Dtype* tgt_variance = src_variance + channels_;
  
  const Dtype alpha = this->blobs_[0]->cpu_data()[0];
  
  // source part
  // src_mean <- sum(ys .* dl_dys)  
  caffe_gpu_mul(src_num * channels_ * spatial_dim, x_norm_src_.gpu_data(),
      src_top_diff, temp_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * src_num, spatial_dim, 1.,
      temp_.gpu_data(), spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, src_num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      src_mean);
  // x_norm_src_diff <- broadcast(sum(ys .* dl_dys))
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), src_mean, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., x_norm_src_d);
  // x_norm_src_diff <- broadcast(sum(ys .* dl_dys)) .* [ys; yts]
  caffe_gpu_mul(temp_.count(), x_norm_src, x_norm_src_d, x_norm_src_d);
  // src_mean_d <- sum(dl_dys)
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * src_num, spatial_dim, 1.,
      src_top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, src_num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      src_mean_d);
  // x_norm_src_diff <- broadcast(sum(ys .* dl_dys)) .* [ys; yts]
  //                    + broadcast(sum(dl_dys))
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), src_mean_d, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., x_norm_src_d);
  // scale by [-w/ns; -(1-w)/nt]
  caffe_gpu_scal<Dtype>(src_num * channels_ * spatial_dim,
      -alpha / (src_num * spatial_dim), x_norm_src_d);
  caffe_gpu_scal<Dtype>(tgt_num * channels_ * spatial_dim,
      -(1. - alpha) / (tgt_num * spatial_dim), x_norm_src_d + offset);
  // sum dl_dys to src part of x_norm_src_diff
  caffe_gpu_axpy(src_num * channels_ * spatial_dim, Dtype(1.),
      src_top_diff, x_norm_src_d);
  // divide everything by sqrt(eps + sigma_s)
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), src_variance, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), x_norm_src_d, temp_.gpu_data(), x_norm_src_d);
  
  // target part
  // tgt_mean <- sum(yt .* dl_dyt)
  caffe_gpu_mul(tgt_num * channels_ * spatial_dim, x_norm_tgt_.gpu_data() +
      offset, tgt_top_diff, temp_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * tgt_num, spatial_dim, 1.,
      temp_.gpu_data(), spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, tgt_num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      tgt_mean);
  // x_norm_tgt_diff <- broadcast(sum(yt .* dl_dyt))
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), tgt_mean, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., x_norm_tgt_d);
  // x_norm_tgt_diff <- broadcast(sum(yt .* dl_dyt)) .* [yst; yt]
  caffe_gpu_mul(temp_.count(), x_norm_tgt, x_norm_tgt_d, x_norm_tgt_d);
  // tgt_mean <- sum(dl_dyt)
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * tgt_num, spatial_dim, 1.,
      tgt_top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, tgt_num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      tgt_mean_d);
  // x_norm_tgt_diff <- broadcast(sum(yt .* dl_dyt)) .* [yst; yt]
  //                    + broadcast(sum(dl_dyt))
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), tgt_mean_d, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., x_norm_tgt_d);
  // scale by [-(1-w)/ns; -w/nt]
  caffe_gpu_scal<Dtype>(src_num * channels_ * spatial_dim,
      -(1. - alpha) / (src_num * spatial_dim), x_norm_tgt_d);
  caffe_gpu_scal<Dtype>(tgt_num * channels_ * spatial_dim,
      -alpha / (tgt_num * spatial_dim), x_norm_tgt_d + offset);
  // sum dl_dyt to tgt part of x_norm_tgt_diff
  caffe_gpu_axpy(tgt_num * channels_ * spatial_dim, Dtype(1.),
      tgt_top_diff, x_norm_tgt_d + offset);
  // divide everything by sqrt(eps + sigma_t)
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), tgt_variance, 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), x_norm_tgt_d, temp_.gpu_data(), x_norm_tgt_d);
  
  // Compose output
  caffe_gpu_add(temp_.count(), x_norm_src_d, x_norm_tgt_d, bottom_diff);
  
  // Gradient w.r.t weight
  int nthreads = num * spatial_dim;
  DialWeightBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, x_norm_src, x_norm_tgt, src_mean_d,
      src_mean, tgt_mean_d, tgt_mean, temp_.mutable_gpu_data(), src_num,
      tgt_num, channels_);
  caffe_gpu_set(temp_.count(), Dtype(1.), temp_.mutable_gpu_diff());
  caffe_gpu_dot(temp_.count(), temp_.mutable_gpu_diff(),
      temp_.mutable_gpu_data(), this->blobs_[0]->mutable_cpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(DialLayer);

}  // namespace caffe

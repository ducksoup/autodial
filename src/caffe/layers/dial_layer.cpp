#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/dial_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DialLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DialParameter param = this->layer_param_.dial_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  test_stats_ = param.test_stats();
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (!param.has_slice_point() || param.slice_point() == -1)
    slice_point_ = bottom[0]->shape(0) / 2;
  else
    slice_point_ = param.slice_point();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    // Weighting parameter
    vector<int> sz;
    sz.push_back(1);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.dial_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // Moving averages
    sz[0] = 2;
    sz.push_back(channels_);
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    sz.resize(1);
    sz[0] = 1;
    this->blobs_[3].reset(new Blob<Dtype>(sz));
    for (int i = 1; i < 4; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
  // Mask statistics from optimization by setting local learning rates
  // for mean, variance, and the bias correction to zero.
  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      if (i == 0)
        fixed_param_spec->set_lr_mult(1.f);
      else
        fixed_param_spec->set_lr_mult(0.f);
    } else {
      if (i != 0) {
        CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
            << "Cannot configure batch normalization statistics as layer "
            << "parameters.";
      }
    }
  }
}

template <typename Dtype>
void DialLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(2);
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  x_norm_src_.ReshapeLike(*bottom[0]);
  x_norm_tgt_.ReshapeLike(*bottom[0]);
  sz.resize(1);
  sz[0] = bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
  }
  
  if (!use_global_stats_)
    CHECK_LE(slice_point_, bottom[0]->shape(0));
}

template <typename Dtype>
void DialLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
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
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[2]->cpu_data(), variance_.mutable_cpu_data());
    
    // Select between source / target stats
    const Dtype* mean = mean_.cpu_data() +
        (test_stats_ == DialParameter_TestStats_TARGET) * channels_;
    Dtype* variance = variance_.mutable_cpu_data() +
        (test_stats_ == DialParameter_TestStats_TARGET) * channels_;
    
    // Subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.cpu_data(), mean, 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, -1, num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 1., top_data);
    
    // Normalize variance
    caffe_add_scalar(channels_, eps_, variance);
    caffe_powx(channels_, variance, Dtype(0.5), variance);

    // replicate variance to input size and do division
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.cpu_data(), variance, 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, 1., num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
    return;
  }
  
  // copy everything to x_norm variables
  caffe_copy(bottom[0]->count(), bottom_data, x_norm_src_.mutable_cpu_data());
  caffe_copy(bottom[0]->count(), bottom_data, x_norm_tgt_.mutable_cpu_data());

  // Find pointers to src and tgt data
  int offset = slice_point_ * channels_ * spatial_dim;
  const Dtype* src_data = bottom_data;
  const Dtype* tgt_data = src_data + offset;
  Dtype* src_top = top_data;
  Dtype* tgt_top = src_top + offset;
  int src_num = slice_point_;
  int tgt_num = num - src_num;
  Dtype* src_mean = mean_.mutable_cpu_data();
  Dtype* tgt_mean = src_mean + channels_;
  Dtype* src_variance = variance_.mutable_cpu_data();
  Dtype* tgt_variance = src_variance + channels_;
  
  // compute means
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * src_num, spatial_dim,
      1. / (src_num * spatial_dim), src_data,
      spatial_sum_multiplier_.cpu_data(), 0., num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, src_num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0., src_mean);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * tgt_num, spatial_dim,
      1. / (tgt_num * spatial_dim), tgt_data,
      spatial_sum_multiplier_.cpu_data(), 0., num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, tgt_num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0., tgt_mean);
  
  // combine with weight
  caffe_copy(mean_.count(), src_mean, src_variance);
  caffe_cpu_axpby(channels_, Dtype(1. - alpha), tgt_variance, alpha, src_mean);
  caffe_cpu_axpby(channels_, Dtype(1. - alpha), src_variance, alpha, tgt_mean);
  
  // subtract src mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), src_mean, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., x_norm_src_.mutable_cpu_data());
  
  // subtract tgt mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), tgt_mean, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., x_norm_tgt_.mutable_cpu_data());
  
  // compute src variance
  caffe_powx(temp_.count(), x_norm_src_.cpu_data(), Dtype(2),
      temp_.mutable_cpu_data());
  caffe_scal<Dtype>(src_num * channels_ * spatial_dim, alpha /
      (src_num * spatial_dim), temp_.mutable_cpu_data());
  caffe_scal<Dtype>(tgt_num * channels_ * spatial_dim, (1. - alpha) /
      (tgt_num * spatial_dim), temp_.mutable_cpu_data() + offset);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
      1., temp_.cpu_data(), spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      src_variance);
  
  // compute tgt variance
  caffe_powx(temp_.count(), x_norm_tgt_.cpu_data(), Dtype(2),
      temp_.mutable_cpu_data());
  caffe_scal<Dtype>(src_num * channels_ * spatial_dim, (1. - alpha) /
      (src_num * spatial_dim), temp_.mutable_cpu_data());
  caffe_scal<Dtype>(tgt_num * channels_ * spatial_dim, alpha /
      (tgt_num * spatial_dim), temp_.mutable_cpu_data() + offset);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
      1., temp_.cpu_data(), spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      tgt_variance);
  
  // compute and save moving average
  this->blobs_[3]->mutable_cpu_data()[0] *= moving_average_fraction_;
  this->blobs_[3]->mutable_cpu_data()[0] += 1;
  caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
      moving_average_fraction_, this->blobs_[1]->mutable_cpu_data());
  int m = bottom[0]->count()/channels_;
  Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
  caffe_cpu_axpby(variance_.count(), bias_correction_factor,
      variance_.cpu_data(), moving_average_fraction_,
      this->blobs_[2]->mutable_cpu_data());
  
  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
             variance_.mutable_cpu_data());
  
  // divide by src variance
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), src_variance, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), x_norm_src_.cpu_data(), temp_.cpu_data(),
      x_norm_src_.mutable_cpu_data());
  
  // divide by tgt variance
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), tgt_variance, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), x_norm_tgt_.cpu_data(), temp_.cpu_data(),
      x_norm_tgt_.mutable_cpu_data());
  
  // copy to output
  caffe_copy(src_num * channels_ * spatial_dim,
             x_norm_src_.cpu_data(), src_top);
  caffe_copy(tgt_num * channels_ * spatial_dim,
             x_norm_tgt_.cpu_data() + offset, tgt_top);
}

template <typename Dtype>
void DialLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  if (use_global_stats_) {
    caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
    return;
  }
  
  // Dimensions
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  int offset = slice_point_ * channels_ * spatial_dim;
  int src_num = slice_point_;
  int tgt_num = num - src_num;
  
  // Pointers
  const Dtype* x_norm_src = x_norm_src_.cpu_data();
  const Dtype* x_norm_tgt = x_norm_tgt_.cpu_data();
  Dtype* x_norm_src_d = x_norm_src_.mutable_cpu_diff();
  Dtype* x_norm_tgt_d = x_norm_tgt_.mutable_cpu_diff();
  const Dtype* src_top_diff = top_diff;
  const Dtype* tgt_top_diff = top_diff + offset;
  Dtype* src_mean = mean_.mutable_cpu_data();
  Dtype* tgt_mean = src_mean + channels_;
  Dtype* src_mean_d = mean_.mutable_cpu_diff();
  Dtype* tgt_mean_d = src_mean_d + channels_;
  Dtype* src_variance = variance_.mutable_cpu_data();
  Dtype* tgt_variance = src_variance + channels_;
  
  const Dtype alpha = this->blobs_[0]->cpu_data()[0];
  
  // source part
  // src_mean <- sum(ys .* dl_dys)  
  caffe_mul(src_num * channels_ * spatial_dim, x_norm_src_.cpu_data(),
      src_top_diff, temp_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * src_num, spatial_dim, 1.,
      temp_.cpu_data(), spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, src_num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      src_mean);
  // x_norm_src_diff <- broadcast(sum(ys .* dl_dys))
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), src_mean, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., x_norm_src_d);
  // x_norm_src_diff <- broadcast(sum(ys .* dl_dys)) .* [ys; yts]
  caffe_mul(temp_.count(), x_norm_src, x_norm_src_d, x_norm_src_d);
  // src_mean_d <- sum(dl_dys)
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * src_num, spatial_dim, 1.,
      src_top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, src_num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      src_mean_d);
  // x_norm_src_diff <- broadcast(sum(ys .* dl_dys)) .* [ys; yts]
  //                    + broadcast(sum(dl_dys))
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), src_mean_d, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., x_norm_src_d);
  // scale by [-w/ns; -(1-w)/nt]
  caffe_scal<Dtype>(src_num * channels_ * spatial_dim,
      -alpha / (src_num * spatial_dim), x_norm_src_d);
  caffe_scal<Dtype>(tgt_num * channels_ * spatial_dim,
      -(1. - alpha) / (tgt_num * spatial_dim), x_norm_src_d + offset);
  // sum dl_dys to src part of x_norm_src_diff
  caffe_axpy(src_num * channels_ * spatial_dim, Dtype(1.),
      src_top_diff, x_norm_src_d);
  // divide everything by sqrt(eps + sigma_s)
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), src_variance, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), x_norm_src_d, temp_.cpu_data(), x_norm_src_d);
  
  // target part
  // tgt_mean <- sum(yt .* dl_dyt)
  caffe_mul(tgt_num * channels_ * spatial_dim, x_norm_tgt_.cpu_data() + offset,
      tgt_top_diff, temp_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * tgt_num, spatial_dim, 1.,
      temp_.cpu_data(), spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, tgt_num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      tgt_mean);
  // x_norm_tgt_diff <- broadcast(sum(yt .* dl_dyt))
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), tgt_mean, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., x_norm_tgt_d);
  // x_norm_tgt_diff <- broadcast(sum(yt .* dl_dyt)) .* [yst; yt]
  caffe_mul(temp_.count(), x_norm_tgt, x_norm_tgt_d, x_norm_tgt_d);
  // tgt_mean <- sum(dl_dyt)
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * tgt_num, spatial_dim, 1.,
      tgt_top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, tgt_num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      tgt_mean_d);
  // x_norm_tgt_diff <- broadcast(sum(yt .* dl_dyt)) .* [yst; yt]
  //                    + broadcast(sum(dl_dyt))
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), tgt_mean_d, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., x_norm_tgt_d);
  // scale by [-(1-w)/ns; -w/nt]
  caffe_scal<Dtype>(src_num * channels_ * spatial_dim,
      -(1. - alpha) / (src_num * spatial_dim), x_norm_tgt_d);
  caffe_scal<Dtype>(tgt_num * channels_ * spatial_dim,
      -alpha / (tgt_num * spatial_dim), x_norm_tgt_d + offset);
  // sum dl_dyt to tgt part of x_norm_tgt_diff
  caffe_axpy(tgt_num * channels_ * spatial_dim, Dtype(1.),
      tgt_top_diff, x_norm_tgt_d + offset);
  // divide everything by sqrt(eps + sigma_t)
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), tgt_variance, 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), x_norm_tgt_d, temp_.cpu_data(), x_norm_tgt_d);
  
  // Compose output
  caffe_add(temp_.count(), x_norm_src_d, x_norm_tgt_d, bottom_diff);
  
  // Gradient w.r.t weight
  Dtype dw = 0;
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < channels_; ++c) {
      for (int j = 0; j < spatial_dim; ++j) {
        Dtype ys = x_norm_src[(i * channels_ + c) * spatial_dim + j];
        Dtype yt = x_norm_tgt[(i * channels_ + c) * spatial_dim + j];
        if (i < slice_point_) {
          dw += -1. / (src_num * spatial_dim) * ys * src_mean_d[c];
          dw += -0.5 / (src_num * spatial_dim) * ys * ys * src_mean[c];
          dw += 1. / (src_num * spatial_dim) * yt * tgt_mean_d[c];
          dw += 0.5 / (src_num * spatial_dim) * yt * yt * tgt_mean[c];
        } else {
          dw += 1. / (tgt_num * spatial_dim) * ys * src_mean_d[c];
          dw += 0.5 / (tgt_num * spatial_dim) * ys * ys * src_mean[c];
          dw += -1. / (tgt_num * spatial_dim) * yt * tgt_mean_d[c];
          dw += -0.5 / (tgt_num * spatial_dim) * yt * yt * tgt_mean[c];
        }
      }
    }
  }
  this->blobs_[0]->mutable_cpu_diff()[0] = dw;
}

#ifdef CPU_ONLY
STUB_GPU(DialLayer);
#endif

INSTANTIATE_CLASS(DialLayer);
REGISTER_LAYER_CLASS(Dial);
}  // namespace caffe

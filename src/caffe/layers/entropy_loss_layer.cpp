#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  // Reshape entropy blob
  vector<int> entropy_shape = prob_.shape();
  entropy_shape[softmax_axis_] = 1;
  entropy_.Reshape(entropy_shape);
}

template <typename Dtype>
Dtype EntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_VALID:
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* entropy_data = entropy_.mutable_cpu_data();
  int cls = bottom[0]->shape(softmax_axis_);
  int dim = cls * inner_num_;
  // Compute entropy
  caffe_set(entropy_.count(), Dtype(0), entropy_data);
  for (int n = 0; n < outer_num_; ++n) {
    for (int s = 0; s < inner_num_; ++s) {
      for (int c = 0; c < cls; ++c) {
        const Dtype p = prob_data[n * dim + c * inner_num_ + s];
        entropy_data[n * inner_num_ + s] -= p * std::log(std::max(p, Dtype(FLT_MIN)));
      }
    }
  }
  Dtype loss = caffe_cpu_asum(entropy_.count(), entropy_data);
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* entropy_data = entropy_.cpu_data();
    int cls = bottom[0]->shape(softmax_axis_);
    int dim = cls * inner_num_;
    for (int n = 0; n < outer_num_; ++n) {
      for (int s = 0; s < inner_num_; ++s) {
        const Dtype H = entropy_data[n * inner_num_ + s];
        for (int c = 0; c < cls; ++c) {
          const Dtype p = prob_data[n * dim + c * inner_num_ + s];          
          bottom_diff[n * dim + c * inner_num_ + s] = Dtype(-1) * p * (H +
              std::log(std::max(p, Dtype(FLT_MIN))));
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(EntropyLossLayer);
#endif

INSTANTIATE_CLASS(EntropyLossLayer);
REGISTER_LAYER_CLASS(EntropyLoss);

}  // namespace caffe

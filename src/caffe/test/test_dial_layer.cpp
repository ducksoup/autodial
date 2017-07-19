#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dial_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

  template <typename TypeParam>
  class DialLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
   protected:
    DialLayerTest()
        : blob_bottom_(new Blob<Dtype>(6, 2, 3, 4)),
          blob_top_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~DialLayerTest() { delete blob_bottom_; delete blob_top_; }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  TYPED_TEST_CASE(DialLayerTest, TestDtypesAndDevices);

  TYPED_TEST(DialLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    DialLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-4);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  TYPED_TEST(DialLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    Dtype alpha = 0.8;
    
    LayerParameter layer_param;
    layer_param.mutable_dial_param()->mutable_weight_filler()->set_type("constant");
    layer_param.mutable_dial_param()->mutable_weight_filler()->set_value(alpha);

    DialLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      // Compute input stats
      Dtype sum_src = 0, sum_tgt = 0;
      int n_src = 0, n_tgt = 0;
      for (int i = 0; i < num; ++i) {
        for (int l = 0; l < height; ++l) {
          for (int m = 0; m < width; ++m) {
            if (i < num/2) {
              n_src++;
              sum_src += this->blob_bottom_->data_at(i, j, l, m);
            } else {
              n_tgt++;
              sum_tgt += this->blob_bottom_->data_at(i, j, l, m);
            }
          }
        }
      }
      Dtype mean_src = alpha / n_src * sum_src + (1-alpha) / n_tgt * sum_tgt;
      Dtype mean_tgt = (1-alpha) / n_src * sum_src + alpha / n_tgt * sum_tgt;
      
      Dtype var_src = 0, var_tgt = 0;
      for (int i = 0; i < num; ++i) {
        for (int l = 0; l < height; ++l) {
          for (int m = 0; m < width; ++m) {
            Dtype diff_src = this->blob_bottom_->data_at(i, j, l, m) - mean_src;
            Dtype diff_tgt = this->blob_bottom_->data_at(i, j, l, m) - mean_tgt;
            if (i < num/2) {
              var_src += diff_src * diff_src * alpha / n_src;
              var_tgt += diff_tgt * diff_tgt * (1 - alpha) / n_tgt;
            } else {
              var_src += diff_src * diff_src * (1 - alpha) / n_src;
              var_tgt += diff_tgt * diff_tgt * alpha / n_tgt;
            }
          }
        }
      }
      
      // Check output
      const Dtype kErrorBound = 0.001;
      for (int i = 0; i < num; ++i) {
        for (int l = 0; l < height; ++l) {
          for (int m = 0; m < width; ++m) {
            if (i < num/2) {
              Dtype x = this->blob_bottom_->data_at(i, j, l, m);
              Dtype y = this->blob_top_->data_at(i, j, l, m);
              EXPECT_NEAR(y, (x - mean_src) / sqrt(1e-5 + var_src), kErrorBound);
            } else {
              Dtype x = this->blob_bottom_->data_at(i, j, l, m);
              Dtype y = this->blob_top_->data_at(i, j, l, m);
              EXPECT_NEAR(y, (x - mean_tgt) / sqrt(1e-5 + var_tgt), kErrorBound);
            }
          }
        }
      }
    }
  }
}  // namespace caffe

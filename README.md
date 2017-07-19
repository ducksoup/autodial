# AutoDIAL Caffe Implementation

This is the official Caffe implementation of [AutoDIAL: Automatic DomaIn Alignment Layers](https://arxiv.org/pdf/1704.08082.pdf)

This code is forked from [BVLC/caffe](https://github.com/BVLC/caffe).
For any issue not directly related to our components (listed in the following), please refer to the upstream repository.

## Contents

We provide two additional layers and some example configuration files to train AlexNet-DIAL on the Office-31 dataset:

* DialLayer: implements the AutoDIAL layer described in the paper.
* EntropyLossLayer: a simple entropy loss implementation with integrated softmax computation.
* AlexNet-DIAL: model and train *.prototxt files to train AlexNet-DIAL on Office-31 are available under models/alexnet_dial. They assume the Office-31 images are formatted in a way that is compatible with Caffe's ImageDataLayer.

### DialLayer

At training time, DialLayer assumes that images / samples from the source and target sets are collected in the same batch, with the source data stored in the first `n` elements and the target data stored in the remaining `N - n` elements.
The splitting point `n` can be freely configured by the user.
Similarly to BatchNormLayer, DialLayer computes on-line estimates of the input's mean and standard deviation, but it does so separately for the source and target sets.
At test time, DialLayer assumes that the batches contain samples from a single set, and uses the same (configurable) statistics to normalize all inputs.

DialLayer accepts all of BatchNormLayer's parameters (`use_global_stats`, `moving_average_fraction`, `eps`), with the addition of:

* `slice_point`: the batch index `n` of the first target sample.
* `test_stats`, one of `SOURCE` or `TARGET`: determines which of the stored statistics are used to normalize the input at test time.
* `weight_filler`: a filler to initialize `alpha` (see the paper).

## Abstract and citation

Classifiers trained on given databases perform poorly when tested on data acquired in different settings. This is explained in domain adaptation through a shift among distributions of the source and target domains. Attempts to align them have traditionally resulted in works reducing the domain shift by introducing appropriate loss terms, measuring the discrepancies between source and target distributions, in the objective function. Here we take a different route, proposing to align the learned representations by embedding in any given network specific Domain Alignment Layers, designed to match the source and target feature distributions to a reference one. Opposite to previous works which define a priori in which layers adaptation should be performed, our method is able to automatically learn the degree of feature alignment required at different levels of the deep network. Thorough experiments on different public benchmarks, in the unsupervised setting, confirm the power of our approach.

    @inproceedings{carlucci2017autodial,
      title={AutoDIAL: Automatic DomaIn Alignment Layers},
      author={Carlucci, Fabio Maria and Porzi, Lorenzo and Caputo, Barbara and Ricci, Elisa and Rota Bul{\`o}, Samuel},
      booktitle={International Conference on Computer Vision},
      year={2017}
    }

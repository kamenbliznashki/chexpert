# Model implementations

This is a collection of model implementations applied on the CheXpert dataset. Each model was first trained on CIFAR10 or CIFAR100 with hyperparameters described in the original paper to check functionality; due to compute constraints, duration of training and hyperparameters vary across architectures as results below aim to show functionality rather than benchmarking.

## EfficientNet

Implementation of [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).

#### Results on CIFAR-10

Model was trained for 70 epochs with parameters published in the paper and referenced in the official tensorflow implementation with the following exceptions: learning rate was set to 2.5e-3, exponential moving average was not applied (cf author implementation). Weight decay was not fine-tuned so both models overfit quite a bit.

| Model | Train loss | Eval loss | Accuracy @ top1 | Accuracy @ top5 |
| --- | --- | --- | --- | --- |
| EfficientNet-B0 | 0.0426 | 1.0626 | 0.7703 | 0.9736 |
| EfficientNet-B7 | 0.1186 | 1.0641 | 0.7649 | 0.9730 |

#### References

* Official author implementation in tensorflow https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

## Attention Augmented Convolutional Networks

Implementation of [Attention Augmented Convolutional Networks](https://arxiv.org/pdf/1904.09925.pdf).

**Densenet** architecture is augmented at the Transition layer (2nd and further as feature map sized are downsampled to sufficiently to allow attention). Specifically, in the first attention configuration the Transition layer remained the same as in the original DenseNet only substituting the 1x1 convolution for an attention augmented convolution; in the second attention configuration the AvgPool was removed and Transition layer used a 3x3 attention augmented convolution with stride 2 for downsampling. **WideResnet** architecture is augmented at the first 3x3 convolution in each BasicBlock of layers 2 and 3 (as described in the paper).

#### Results on CIFAR-100

Models trained on CIFAR-100 in order to compare against results from the paper (this is probably not a good proxy task for Chexpert). Models were trained using SGD (learning rate 0.1, nestorov momentum 0.9); WideResnet further used cosine annealing (T=50) and weight decay per original papers.

| Model | Epochs | Train loss | Eval loss | Accuracy @ top1 | Accuracy @ top5 | Attention params | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|Densenet	(k12 L100) AAConv1x1 | 100 | 0.0183 | 2.006 | 0.6934 | 0.9068 | Nh 8, k 0.2, v 0.7 | Transition layer = BatchNorm -> Relu -> AAConv1x1 -> Avgpool |
|Densenet	(k12 L100) AAConv3x3 w/ InstanceNorm | 100 | 0.8863 | 1.240 | 0.6653 | 0.9067 | Nh 8, k 0.2, v 0.7 | Transition layer = InstanceNorm -> Relu -> AAConv3x3 <br> (performed better than BatchNorm or LayerNorm) |
|Densenet	(k12 L100) baseline -- no attention | 100 | 0.4426 | 1.382 | 0.6499 | 0.8981 | | |
| WideResNet-28-10 attention | 150 | 0.0003 | 1.7780 | 0.7422 | 0.9257 | Nh 8, k 0.2, v 0.1 |
| WideResNet-28-10 baseline -- nattention | 150 | 0.0002 | 1.5990 | 0.7593 | 0.9362 | |


#### Attention visualizations on CIFAR100 using Densenet (k12 L100) AAConv3x3 w/ InstanceNorm

Attention maps at the 4 pixels highlighted in yellow (corners of the centered square of size 1/3rd of the image). Rows denote different attention heads, column denote the visualized pixels.

| Transition layer 1 | Transition layer 2|
| --- | --- |
![attn_aug_densenet_transition1](https://raw.github.com/kamenbliznashki/chexpert/master/images/vis_attn_densenet_aaconv3x3_instancenorm_cifar100_image_0_layer_0.png) | ![attn_aug_densenet_transition2](https://raw.github.com/kamenbliznashki/chexpert/master/images/vis_attn_densenet_aaconv3x3_instancenorm_cifar100_image_0_layer_1.png) |

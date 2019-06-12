# Model implementations

This is a collection of model implementations applied on the CheXpert dataset.

## EfficientNet

Implementation of [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).

#### Results on CIFAR 10

Model was trained for 70 epochs with parameters published in the paper and referenced in the official tensorflow implementation with the following exceptions: learning rate was set to 2.5e-3, exponential moving average was not applied (cf author implementation). Weight decay was not fine-tuned so both models overfit quite a bit.

| Model | Train loss | Eval loss | Accuracy @ top1 | Accuracy @ top5 |
| --- | --- | --- | --- | --- |
| EfficientNet-B0 | 0.0426 | 1.0626 | 0.7703 | 0.9736 |
| EfficientNet-B7 | 0.1186 | 1.0641 | 0.7649 | 0.9730 |

#### References

* Official author implementation in tensorflow https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

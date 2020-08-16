# nnieqat-pytorch

This is a quantize aware training package for  Neural Network Inference Engine(NNIE) on pytorch, it uses hisilicon quantization library to quantize module's weight and input data as fake fp32 format. To train model which is more friendly to NNIE, just import nnieqat and replace torch.nn default modules with corresponding one.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Code Examples](#examples)
4. [Results](#results)
5. [Todo](#Todo)
6. [Reference](#reference)


<div id="installation"></div>  

## Installation

* Supported Platforms: Linux
* Accelerators and GPUs: NVIDIA GPUs via CUDA driver 10.
* Dependencies:
  * python >= 3.5, < 4
  * llvmlite >= 0.31.0
  * pytorch >= 1.0
  * numba >= 0.42.0
  * numpy >= 1.18.1
* Install nnieqat via pypi:
  $ pip install nnieqat

<div id="usage"></div>

## Usage

* Replace default module with NNIE quantization optimized one. include:
  * torch.nn.modules.conv -> nnieqat.modules.conv
  * torch.nn.modules.linear -> nnieqat.modules.linear
  * torch.nn.modules.pooling -> nnieqat.modules.pooling

  ```python
  from nnieqat.modules import convert_layers
  ...
  ...
    model = convert_layers(model)
    print(model)  # Quantized layers have "Quantized" prefix.
  ...
  ```

* Freeze bn after a few epochs of training

  ```python
  from nnieqat.gpu.quantize import freeze_bn
  ...
  ...
      if epoch > 2:
        net.apply(freeze_bn)
  ...
  ```

* Unquantize weight before update it

  ```python
  from nnieqat.gpu.quantize import unquant_weight
  ...
  ...
      net.apply(unquant_weight)
      optimizer.step()
  ...
  ```

* Dump weight quantized model

  ```python
  from nnieqat.gpu.quantize import quant_weight, unquant_weight
  ...
  ...
      net.apply(quant_weight)
      save_checkpoint(...)
      net.apply(unquant_weight)
  ...
  ```

<div id="examples"></div>

## Code Examples

* [Cifar10 quantization aware training example][cifar10_qat]  (add nnieqat into [pytorch_cifar10_tutorial][cifar10_example])

  ```python test/test_cifar10.py```

* [ImageNet quantization finetuning example][imagenet_qat]  (add nnieqat into [pytorh_imagenet_main.py][imagenet_example])

  ```python test/test_imagenet.py  --pretrained  path_to_imagenet_dataset```

<div id="results"></div>

## Results  

* ImageNet

  ```
  python test/test_imagenet.py /data/imgnet/ --arch squeezenet1_1  --lr 0.001 --pretrained --epoch 10   # nnie_lr_e-3_ft
  python pytorh_imagenet_main.py /data/imgnet/ --arch squeezenet1_1  --lr 0.0001 --pretrained --epoch 10  # lr_e-4_ft
  python test/test_imagenet.py /data/imgnet/ --arch squeezenet1_1  --lr 0.0001 --pretrained --epoch 10  # nnie_lr_e-4_ft
  ```

  finetune result：

    |     | trt_fp32 | trt_int8     | nnie     |
    | -------- |  -------- | -------- | -------- |
    | torchvision     | 0.56992  | 0.56424  | 0.56026 |
    | nnie_lr_e-3_ft | 0.56600   | 0.56328   | 0.56612 |
    | lr_e-4_ft  | 0.57884   | 0.57502   | 0.57542 |
    | nnie_lr_e-4_ft | 0.57834   | 0.57524   | 0.57730 |  


<div id="Todo"></div>

## Todo

* Multiple GPU training support.
* Other platforms and accelerators support.
* Generate quantized model directly.

<div id="reference"></div>  

## Reference

HiSVP 量化库使用指南

[Quantizing deep convolutional networks for efficient inference: A whitepaper][quant_whitepaper]

[8-bit Inference with TensorRT][trt_quant]

[Distilling the Knowledge in a Neural Network][distillingNN]

[cifar10_qat]: https://gitlab.deepglint.com/chenMQ/nnieqat-pytorch/-/blob/master/test/test_cifar10.py

[imagenet_qat]: https://gitlab.deepglint.com/chenMQ/nnieqat-pytorch/-/blob/master/test/test_imagenet.py

[imagenet_example]: https://github.com/pytorch/examples/blob/master/imagenet/main.py

[cifar10_example]: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

[quant_whitepaper]: https://arxiv.org/abs/1806.08342

[trt_quant]: https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

[distillingNN]: https://arxiv.org/abs/1503.02531

[apprentice]: https://arxiv.org/abs/1711.05852

#!/usr/bin/env python
"""Quantize function.
"""

import ctypes
import datetime
import logging
from os.path import abspath, dirname
import torch
import numpy as np
from numba import cuda
from quant_impl import fake_quantize

_USE_GFPQ_QUANT_LIB = (torch.cuda.device_count() <= 1)


class GFPQParamSt(ctypes.Structure):
    r"""GFPQ param, corresponds with struct GFPQ_PARAM_ST in gfpq.hpp"""
    _fields_ = [("mode", ctypes.c_int), ("param", ctypes.c_byte * 16)]


class _types:
    r"""Some alias types."""
    handle = ctypes.c_void_p
    stream = ctypes.c_void_p


class QuantAndDeQuantGPU():
    r"""quantize and dequantize data with GFPG library.
    """
    def __init__(self,
                 libquant_path=dirname(abspath(__file__)) +
                 "/gpu/lib/libgfpq_gpu.so",
                 libcublas_path="libcublas.so",
                 bit_width=8,
                 param_mode=0):
        global _USE_GFPQ_QUANT_LIB
        self._bit_width = bit_width
        if _USE_GFPQ_QUANT_LIB:
            self._libquant = ctypes.cdll.LoadLibrary(libquant_path)
            self._libcublas = ctypes.cdll.LoadLibrary(libcublas_path)
            self._libcublas.cublasCreate_v2.restype = int
            self._libcublas.cublasCreate_v2.argtypes = [ctypes.c_void_p]
            self._cublas_handle = _types.handle()
            self._libcublas.cublasCreate_v2(ctypes.byref(self._cublas_handle))
            self._param = GFPQParamSt()
            self._stream = cuda.stream()
            self._param.mode = param_mode

    def __call__(self, tensor, mode=0):
        r""" Converts float weights to quantized weights.

        Args:
            - tensor: input data
            - mode: GFPQ mode for param
                GFPQ_MODE_INIT(0): There is no valid parameter in param[].
                    Generate the parameter and filled in param[].
                GFPQ_MODE_UPDATE(1): There is parameter in param[]. Generate
                    new parameter, update param[] when the new parameter is
                    better.
                GFPQ_MODE_APPLY_ONLY(2): There is parameter in param[]. Don't
                    generate parameter. Just use the param[].
        """

        global _USE_GFPQ_QUANT_LIB
        if _USE_GFPQ_QUANT_LIB:
            try:
                if isinstance(tensor, tuple):
                    for tensor_item in tensor:
                        data_cuda_array = cuda.as_cuda_array(
                            tensor_item.data.detach())
                        data_p = data_cuda_array.device_ctypes_pointer
                        self._param.mode = mode
                        ret = self._libquant.HI_GFPQ_QuantAndDeQuant_GPU_PY(
                            data_p, data_cuda_array.size, self._bit_width,
                            ctypes.byref(self._param), self._stream.handle,
                            self._cublas_handle)
                else:
                    data_cuda_array = cuda.as_cuda_array(tensor.data.detach())
                    data_p = data_cuda_array.device_ctypes_pointer
                    self._param.mode = mode
                    ret = self._libquant.HI_GFPQ_QuantAndDeQuant_GPU_PY(
                        data_p, data_cuda_array.size, self._bit_width,
                        ctypes.byref(self._param), self._stream.handle,
                        self._cublas_handle)
            except:
                pass
            finally:
                if ret != 0:
                    _USE_GFPQ_QUANT_LIB = False
                    logger = logging.getLogger(__name__)
                    logger.setLevel(logging.WARNING)
                    logger.warning(
                        """Failed to quantize data with default HiSVP GFPQ library,
                        Use implemented quantization algorithm instead.""")
                    if isinstance(tensor, tuple):
                        for tensor_item in tensor:
                            tensor_item.data = fake_quantize(
                                tensor_item.data.detach().clone(), self._bit_width)
                    else:
                        tensor.data = fake_quantize(tensor.data.detach().clone(),
                                                    self._bit_width)
        else:
            if isinstance(tensor, tuple):
                for tensor_item in tensor:
                    tensor_item.data = fake_quantize(tensor_item.data.detach().clone(),
                                                     self._bit_width)
            else:
                tensor.data = fake_quantize(tensor.data.detach().clone(),
                                            self._bit_width)
        return tensor


_QUANT_HANDLE = QuantAndDeQuantGPU()


def _fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """ fuse convolution and batch norm's weight.

    Args:
        conv_w (torch.nn.Parameter): convolution weight.
        conv_b (torch.nn.Parameter): convolution bias.
        bn_rm (torch.nn.Parameter): batch norm running mean.
        bn_rv (torch.nn.Parameter): batch norm running variance.
        bn_eps (torch.nn.Parameter): batch norm epsilon.
        bn_w (torch.nn.Parameter): batch norm weight.
        bn_b (torch.nn.Parameter): batch norm weight.

    Returns:
        conv_w(torch.nn.Parameter): fused convolution weight.
        conv_b(torch.nn.Parameter): fused convllution bias.
    """

    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * \
        (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def _fuse_conv_bn(conv, bn):
    conv.weight, conv.bias = \
        _fuse_conv_bn_weights(conv.weight, conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return conv


def _fuse_modules(model):
    r"""Fuses a list of modules into a single module

    Fuses only the following sequence of modules:
    conv, bn
    All other sequences are left unchanged.
    For these sequences, fuse modules on weight level, keep model structure unchanged.

    Arguments:
        model: Model containing the modules to be fused

    Returns:
        model with fused modules.

    """
    children = list(model.named_children())
    conv_module = None
    conv_name = None

    for name, child in children:
        if isinstance(child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                              torch.nn.BatchNorm3d)):
            if isinstance(conv_module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                conv_module = _fuse_conv_bn(conv_module, child)
                model._modules[conv_name] = conv_module
                child.eval()
                child.running_mean = child.running_mean.new_full(
                    child.running_mean.shape, 0)
                child.running_var = child.running_var.new_full(
                    child.running_var.shape, 1)
                if child.weight is not None:
                    child.weight.data = child.weight.data.new_full(
                        child.weight.shape, 1)
                if child.bias is not None:
                    child.bias.data = child.bias.data.new_full(
                        child.bias.shape, 0)
                child.track_running_stats = False
                child.momentum = 0
                child.eps = 0
            conv_module = None
        elif isinstance(child, (torch.nn.Conv2d, torch.nn.Conv3d)):
            conv_module = child
            conv_name = name
        else:
            _fuse_modules(child)
    return model


def freeze_bn(m, freeze_bn_affine=True):
    """Freeze batch normalization.
        reference: https://arxiv.org/abs/1806.08342


    Args:
        - m (nn.module): torch module
        - freeze_bn_affine (bool, optional): Freeze affine scale and
        translation factor or not. Defaults: True.
    """

    if isinstance(
            m,
        (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):

        m.eval()
        if freeze_bn_affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def merge_freeze_bn(model):
    """merge batch norm's weight into convolution, then freeze it.

    Args:
        model (nn.module): model.

    Returns:
        [nn.module]: model.
    """
    model = _fuse_modules(model)
    model.apply(freeze_bn)
    return model


def unquant_weight(m):
    """ unquantize weight before update weight, avoid training turbulence.

    Args:
        - m (nn.module): torch module.
    """
    try:
        if hasattr(m, "weight_origin") and m.weight is not None:
            m.weight.data.copy_(m.weight_origin.data)
    except AttributeError:
        pass
    except TypeError:
        pass


def quant_dequant_weight(m):
    """ quant weight manually.

    Args:
        - m (nn.module): torch module.
    """
    global _QUANT_HANDLE
    global _USE_GFPQ_QUANT_LIB
    quant_handle = _QUANT_HANDLE
    if not _USE_GFPQ_QUANT_LIB:
        quant_handle = QuantAndDeQuantGPU()
    try:
        if hasattr(m, "weight_origin") and m.weight is not None:
            m.weight_origin.data.copy_(m.weight.data)
            m.weight.data = quant_handle(m.weight.data.detach().clone())
    except AttributeError:
        pass
    except TypeError:
        pass


def _quantizing_activation(module, input, output):
    if isinstance(
            module,
        (torch.nn.ReLU, torch.nn.ELU, torch.nn.LeakyReLU, torch.nn.PReLU)):
        global _QUANT_HANDLE
        global _USE_GFPQ_QUANT_LIB
        quant_handle = _QUANT_HANDLE
        if not _USE_GFPQ_QUANT_LIB:
            quant_handle = QuantAndDeQuantGPU()
        # print("quantizing activation.")
        # print(output[0][0][0])
        output_type = output.dtype
        module.activation_max_value = torch.max(torch.max(torch.abs(output.detach())), module.activation_max_value.to(output_type))
        # print(module.activation_max_value)
        tensor_t = torch.cat((output, torch.ones(output[0].shape).cuda().unsqueeze(0) * module.activation_max_value))
        output.data = quant_handle(tensor_t.float())[:-1]
        output = output.to(output_type)
        # print(output[0][0][0])


def _quantizing_data(module, input):
    global _QUANT_HANDLE
    global _USE_GFPQ_QUANT_LIB
    quant_handle = _QUANT_HANDLE
    if not _USE_GFPQ_QUANT_LIB:
        quant_handle = QuantAndDeQuantGPU()
    # print("quantizing data.")
    # print(input[0][0][0])
    # print("quantizing data.")
    # print(input[0][0][0])
    # input_type = input.dtype
    if isinstance(input, tuple):
        for item in input:
            item_type = item.dtype
            item = quant_handle(item.float())
            item.to(item_type)
    else:
        input = quant_handle(input.float())
    # input = input.to(input_type)
    # print(input[0][0][0])


def _quantizing_weight(module, input):
    global _QUANT_HANDLE
    global _USE_GFPQ_QUANT_LIB
    quant_handle = _QUANT_HANDLE
    if not _USE_GFPQ_QUANT_LIB:
        quant_handle = QuantAndDeQuantGPU()
    # print("quantizing weight.")
    # print(module.weight[0][0][0])
    module.weight_origin.data.copy_(module.weight.data)
    module.weight.data = quant_handle(module.weight.data.detach().clone())
    # print(module.weight[0][0][0])


def register_quantization_hook(model,
                               quant_weight=True,
                               quant_activation=True,
                               quant_data=False):
    """register quantization hook for model.

    Args:
        model (:class:`Module`): Module.

    Returns:
        Module: self
    """

    #  weight quantizing.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for _, module in model._modules.items():
        if len(list(module.children())) > 0:
            register_quantization_hook(module, quant_weight, quant_activation)
        else:
            if quant_weight and hasattr(
                    module,
                    "weight") and module.weight is not None and not isinstance(
                        module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                 torch.nn.BatchNorm3d)):
                module.register_buffer('weight_origin', module.weight.detach().clone())
                if quant_data:
                    module.register_forward_pre_hook(_quantizing_data)
                    logger.info("Quantizing input data of %s", str(module))
                module.register_forward_pre_hook(_quantizing_weight)
                logger.info("Quantizing weight of %s", str(module))

            if quant_activation and isinstance(
                    module, (torch.nn.ReLU, torch.nn.ELU, torch.nn.LeakyReLU, torch.nn.PReLU)):
                module.register_buffer("activation_max_value", torch.tensor(0, dtype=torch.float).cuda())
                module.register_forward_hook(_quantizing_activation)
                logger.info("Quantizing activation of %s", str(module))

    return model


def test():
    r""" Test GFPG library QuantAndDeQuantGPU.
    """
    quant_handle = QuantAndDeQuantGPU()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    tensor = torch.Tensor(np.array([-9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).cuda()
    logging.info("Origin Data: ")
    logging.info(tensor)

    start_time = datetime.datetime.now()
    quant_tensor = quant_handle(tensor)
    end_time = datetime.datetime.now()

    logging.info("Quant Data: ")
    logging.info(quant_tensor)

    data_expected = np.array([
        -8.7240619659, 0.0000000000, 1.0000000000, 2.0000000000, 2.9536523819,
        4.0000000000, 4.9674310684, 5.9073047638, 7.0250086784, 8.0000000000,
        8.7240619659
    ])

    logging.info("Data expected:  ")
    logging.info(" ".join([str(v) for v in data_expected]))

    data_diff = quant_tensor.data.detach().cpu().numpy() - data_expected
    flag = "success."
    for num in data_diff:
        if abs(num) > 0.000000001:
            flag = "failed."

    run_time = end_time - start_time
    logging.info("QuantAndDeQuantGPU time: %s", str(run_time))
    logging.info("QuantAndDeQuantGPU %s", flag)

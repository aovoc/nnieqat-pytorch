#!/usr/bin/env python
"""Quantize function.
"""

import ctypes
import datetime
import logging
from os.path import abspath, dirname
import numpy as np
from numba import cuda


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
                 "/lib/libgfpq_gpu.so",
                 libcublas_path="libcublas.so",
                 bit_width=8):
        self._libquant = ctypes.cdll.LoadLibrary(libquant_path)
        self._libcublas = ctypes.cdll.LoadLibrary(libcublas_path)
        self._libcublas.cublasCreate_v2.restype = int
        self._libcublas.cublasCreate_v2.argtypes = [ctypes.c_void_p]
        self._cublas_handle = _types.handle()
        self._libcublas.cublasCreate_v2(ctypes.byref(self._cublas_handle))
        self._bit_width = bit_width
        self._param = GFPQParamSt()
        self._stream = cuda.stream()
        self._param.mode = 0

    def __call__(self, tensor, mode=0):
        r""" Converts float weights to quantized weights.
        Args:
            tensor: input data
            mode: GFPQ mode for param.
                GFPQ_MODE_INIT(0): There is no valid parameter in param[].
                    Generate the parameter and filled in param[].
                GFPQ_MODE_UPDATE(1): There is parameter in param[]. Generate
                    new parameter, update param[] when the new parameter is
                    better.
                GFPQ_MODE_APPLY_ONLY(2): There is parameter in param[]. Don't
                    generate parameter. Just use the param[].
        """

        data_cuda_array = cuda.as_cuda_array(tensor.data.detach())
        data_p = data_cuda_array.device_ctypes_pointer
        self._param.mode = mode
        ret = self._libquant.HI_GFPQ_QuantAndDeQuant_GPU_PY(
            data_p, data_cuda_array.size, self._bit_width,
            ctypes.byref(self._param), self._stream.handle,
            self._cublas_handle)
        assert ret == 0, "HI_GFPQ_QuantAndDeQuant failed(%d)\n" % (ret)
        self._stream.synchronize()
        return tensor


_QUANT_HANDLE = QuantAndDeQuantGPU()


def freeze_bn(m, freeze_bn_affine=True):
    """Freeze batch normalization.
        reference: https://arxiv.org/abs/1806.08342
    Args:
        m (nn.module): torch module.
        freeze_bn_affine (bool, optional): Freeze affine scale and
        translation factor or not. Defaults: True.
    """
    import torch
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()
        if freeze_bn_affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def unquant_weight(m):
    """ unquantize weight before update weight, avoid training turbulence.
    """
    global _QUANT_HANDLE
    try:
        m.weight.data = m.weight_origin
    except AttributeError:
        pass


def quant_weight(m):
    """ quant weight manually.
    """
    global _QUANT_HANDLE
    try:
        m.weight = _QUANT_HANDLE(m.weight)

    except AttributeError:
        pass


def test():
    r""" Test GFPG library QuantAndDeQuantGPU.
    """
    quant_handle = QuantAndDeQuantGPU()
    import torch
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    tensor = torch.Tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).cuda()
    logging.info("Origin Data: ")
    logging.info(tensor)

    start_time = datetime.datetime.now()
    quant_tensor = quant_handle(tensor)
    end_time = datetime.datetime.now()

    logging.info("Quant Data: ")
    logging.info(quant_tensor)

    data_expected = np.array([
        0.0000000000, 1.0000000000, 2.0000000000, 2.9536523819, 4.0000000000,
        4.9674310684, 5.9073047638, 7.0250086784, 8.0000000000, 8.7240619659
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

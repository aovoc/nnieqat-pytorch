# -*- coding:utf-8 -*-
import unittest
from ddt import ddt, data
import math
import ctypes
import datetime
from ctypes import *
import numpy as np
from numba import cuda
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@ddt
class TestQuantImpl(unittest.TestCase):
    max_thres = 512
    data0 = np.array([0])
    data1 = np.array([v / 25600 + 1.04
                      for v in range(25600)] + [100, max_thres])
    data2 = np.array([v / 25600 + 1.04
                      for v in range(25600)] + [100, max_thres])
    data2 = np.array([-v / 25600 - 1.04
                      for v in range(25600)] + [-100, -max_thres])
    data3 = np.array(
        [0, 1, 2, 2.03992188, 2.03996094, 3, 4, 5, 10, 100, max_thres])
    max_thres = 513
    data4 = np.array([v / 25600 + 1.04
                      for v in range(25600)] + [100, max_thres])
    data5 = np.array([v / 25600 + 1.04
                      for v in range(25600)] + [100, max_thres])
    data6 = np.array([-v / 25600 - 1.04
                      for v in range(25600)] + [-100, -max_thres])
    data7 = np.array(
        [0, 1, 2, 2.03992188, 2.03996094, 3, 4, 5, 10, 100, max_thres])
    data8 = np.array([
        0, -1, -2, -2.03992188, -2.03996094, -3, -4, -5, -10, -100, -max_thres
    ])
    data9 = np.array(range(1234))
    data10 = np.array([-v for v in range(1234)])

    @data(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9,
          data10)
    def test(self, data):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # load library
        dl = ctypes.cdll.LoadLibrary
        quant_lib = dl("nnieqat/gpu/lib/libgfpq_gpu.so")
        _libcublas = ctypes.cdll.LoadLibrary("libcublas.so")

        # struct GFPQ_PARAM_ST in gfpq.hpp
        class GFPQ_PARAM_ST(ctypes.Structure):
            _fields_ = [("mode", ctypes.c_int), ("buf", ctypes.c_byte * 16)]

        class _types:
            """Some alias types."""
            handle = ctypes.c_void_p
            stream = ctypes.c_void_p

        data_origin = data.copy()

        print(
            "----------------------------------------------------------------------"
        )
        print("\n\nOriginal data:")
        print(data)

        data = data.astype(np.float32)
        stream = cuda.stream()

        _libcublas.cublasCreate_v2.restype = int
        _libcublas.cublasCreate_v2.argtypes = [ctypes.c_void_p]
        cublas_handle = _types.handle()
        _libcublas.cublasCreate_v2(ctypes.byref(cublas_handle))

        data_gpu = cuda.to_device(data, stream=stream)
        data_p = data_gpu.device_ctypes_pointer
        bit_width = 8

        param = GFPQ_PARAM_ST()
        # init or update param first
        param.mode = 0
        ret = quant_lib.HI_GFPQ_QuantAndDeQuant_GPU_PY(data_p, data.size,
                                                       bit_width,
                                                       ctypes.byref(param),
                                                       stream.handle,
                                                       cublas_handle)
        if ret != 0:
            print("HI_GFPQ_QuantAndDeQuant failed(%d)\n" % (ret)),

        # use apply param
        param.mode = 2
        ret = quant_lib.HI_GFPQ_QuantAndDeQuant_GPU_PY(data_p, data.size,
                                                       bit_width,
                                                       ctypes.byref(param),
                                                       stream.handle,
                                                       cublas_handle)
        if ret != 0:
            print("HI_GFPQ_QuantAndDeQuant failed(%d)" % (ret)),

        data_gpu.copy_to_host(data, stream=stream)
        # data may not be available
        stream.synchronize()
        _libcublas.cublasDestroy_v2(cublas_handle)

        import nnieqat
        from quant_impl import fake_quantize
        import torch
        tensor = torch.Tensor(data_origin).cuda()
        tensor.data = fake_quantize(tensor.data.detach(), 8)

        diff = abs(tensor.cpu().numpy() - data)
        # diff_thres = np.max(abs(data)) * 0.001
        # print("\nDIFF > 0.1%: ")
        # print("idx: ", np.where(diff > diff_thres))
        # print("Original data:", data_origin[np.where(diff > diff_thres)])
        # print("GFPQ result:", data[np.where(diff > diff_thres)])
        # print("Impl result:", tensor.cpu().numpy()[np.where(diff > diff_thres)])
        diff_max = np.max(diff)
        print("\nDIFF MAX: " + str(diff_max))
        print("\nDIFF RATIO: " +
              str(diff_max / max(np.max(abs(data)), pow(10, -18))))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestQuantImpl("test"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

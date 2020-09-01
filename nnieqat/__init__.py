""" quantize aware training package for  Neural Network Inference Engine(NNIE) on pytorch.
"""
import sys
try:
    from .quantize import quant_dequant_weight, unquant_weight, freeze_bn, merge_freeze_bn, register_quantization_hook, test
except:
    raise
__all__ = [
    "QuantAndDeQuantGPU", "test", "quant_dequant_weight", "unquant_weight", "freeze_bn", "merge_freeze_bn", "register_quantization_hook"
]
test()

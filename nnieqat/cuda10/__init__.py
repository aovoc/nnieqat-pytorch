"""Quantize function.
"""
import sys
try:
    from .quantize import QuantAndDeQuantGPU, test, quant_dequant_weight, \
        unquant_weight, freeze_bn, merge_freeze_bn
except:
    raise
__all__ = [
    "QuantAndDeQuantGPU", "test", "quant_dequant_weight", "unquant_weight", "freeze_bn", "merge_freeze_bn"
]
test()

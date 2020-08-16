"""Quantize function.
"""
import sys
try:
    from .quantize import QuantAndDeQuantGPU, test, quant_weight, \
        unquant_weight, freeze_bn
except:
    sys.stderr.write("Error: Please import nniepat before torch modules.\n")
    raise
__all__ = [
    "QuantAndDeQuantGPU", "test", "quant_weight", "unquant_weight", "freeze_bn"
]
test()

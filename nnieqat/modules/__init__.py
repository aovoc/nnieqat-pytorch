"""Inherit some of default torch.nn modules, add quantizating operation.
"""

from .linear import Linear, Bilinear
from .conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, \
    ConvTranspose3d
from .pooling import AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, \
    MaxPool3d, MaxUnpool1d, MaxUnpool2d, MaxUnpool3d, FractionalMaxPool2d,  \
    LPPool1d, LPPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, \
    AdaptiveMaxPool3d, AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d

__all__ = [
    'Linear', 'Bilinear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d',
    'ConvTranspose2d', 'ConvTranspose3d', 'AvgPool1d', 'AvgPool2d',
    'AvgPool3d', 'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'MaxUnpool1d',
    'MaxUnpool2d', 'MaxUnpool3d', 'FractionalMaxPool2d', 'LPPool1d',
    'LPPool2d', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d'
]


def convert_layers(model):
    """Convert model to a quantized one.

    Args:
        model (:class:`Module`): Module.

    Returns:
        Module: self
    """

    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = convert_layers(model=module)
        try:
            module_str = str(module)
            module_new = eval(module_str)
            try:
                module_new.weight = module.weight
                module_new.bias = module.bias
            except:
                pass
            model._modules[name] = module_new
            logger.info("Quantizing " + str(name) + " " + str(module))
        except:
            pass
    return model

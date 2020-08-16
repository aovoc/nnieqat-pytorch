from typing import Optional, List
import nnieqat.gpu.quantize as Q
import torch
from torch import Tensor
from torch.nn.modules.utils import _single, _triple


class Conv1d(torch.nn.Conv1d):
    r"""This is the quantized version of :class:`~torch.nn.Conv1d`.
        Args: Same as torch.nn.Conv1d
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True):
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, False,
                                     _single(0), groups, bias)
        self.quant_handle = Q.QuantAndDeQuantGPU()
        self.weight_origin = None
        self._bit_width = 8

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        self.weight_origin = self.weight.clone()
        self.weight = self.quant_handle(self.weight)
        ret = super(Conv1d, self).forward(input)
        return ret

    def _get_name(self):
        return 'QuantizedConv1d'



class Conv2d(torch.nn.Conv2d):
    r"""This is the quantized version of :class:`~torch.nn.Conv2d`.
        Args: Same as torch.nn.Conv2d
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)
        self.quant_handle = Q.QuantAndDeQuantGPU()
        self.weight_origin = None
        self._bit_width = 8

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        self.weight_origin = self.weight.clone()
        self.weight = self.quant_handle(self.weight)
        ret = super(Conv2d, self).forward(input)
        return ret

    def _get_name(self):
        return 'QuantizedConv2d'


class Conv3d(torch.nn.Conv3d):
    r"""This is the quantized version of :class:`~torch.nn.Conv3d`.
            Args: Same as torch.nn.Conv3d
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, False,
                                     _triple(0), groups, bias)
        self.quant_handle = Q.QuantAndDeQuantGPU()
        self.weight_origin = None
        self._bit_width = 8

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        self.weight_origin = self.weight.clone()
        self.weight = self.quant_handle(self.weight)
        ret = super(Conv3d, self).forward(input)
        return ret

    def _get_name(self):
        return 'QuantizedConv3d'


class ConvTranspose1d(torch.nn.ConvTranspose1d):
    r"""This is the quantized version of :class:`~torch.nn.ConvTranspose1d`.
        Args: Same as torch.nn.ConvTranspose1d
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation=1):
        super(ConvTranspose1d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, True, output_padding, groups,
                             bias)
        self.quant_handle = Q.QuantAndDeQuantGPU()
        self.weight_origin = None
        self._bit_width = 8

    def forward(self,
                input: Tensor,
                output_size: Optional[List[int]] = None) -> Tensor:
        input = self.quant_handle(input)
        self.weight_origin = self.weight.clone()
        self.weight = self.quant_handle(self.weight)
        ret = super(ConvTranspose1d, self).forward(input)
        return ret

    def _get_name(self):
        return 'QuantizedConvTranspose1d'


class ConvTranspose2d(torch.nn.ConvTranspose1d):
    r"""This is the quantized version of :class:`~torch.nn.ConvTranspose2d`.
        Args: Same as torch.nn.ConvTranspose2d
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1):
        super(ConvTranspose2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, True, output_padding, groups,
                             bias)
        self.quant_handle = Q.QuantAndDeQuantGPU()
        self.weight_origin = None
        self._bit_width = 8

    def forward(self,
                input: Tensor,
                output_size: Optional[List[int]] = None) -> Tensor:
        input = self.quant_handle(input)
        self.weight_origin = self.weight.clone()
        self.weight = self.quant_handle(self.weight)
        ret = super(ConvTranspose2d, self).forward(input)
        return ret

    def _get_name(self):
        return 'QuantizedConvTranspose2d'


class ConvTranspose3d(torch.nn.ConvTranspose1d):
    r"""This is the quantized version of :class:`~torch.nn.ConvTranspose3d`.
        Args: Same as torch.nn.ConvTranspose3d
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation=1):
        super(ConvTranspose3d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, True, output_padding, groups,
                             bias)
        self.quant_handle = Q.QuantAndDeQuantGPU()
        self.weight_origin = None
        self._bit_width = 8

    def forward(self,
                input: Tensor,
                output_size: Optional[List[int]] = None) -> Tensor:
        input = self.quant_handle(input)
        self.weight_origin = self.weight.clone()
        self.weight = self.quant_handle(self.weight)
        ret = super(ConvTranspose3d, self).forward(input)
        return ret

    def _get_name(self):
        return 'QuantizedConvTranspose3d'

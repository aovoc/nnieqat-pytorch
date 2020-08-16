import nnieqat.gpu.quantize as Q
import torch
from torch import Tensor
import torch.nn.functional as F


class MaxPool1d(torch.nn.MaxPool1d):
    r"""This is the quantized version of :class:`~torch.nn.MaxPool1d`.
        Args: Same as torch.nn.MaxPool1d
    """
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices: bool = False,
                 ceil_mode: bool = False) -> None:
        super(MaxPool1d, self).__init__(kernel_size, stride, padding, dilation,
                                        return_indices, ceil_mode)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.max_pool1d(input, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

    def _get_name(self):
        return 'QuantizedMaxPool1d'


class MaxPool2d(torch.nn.MaxPool2d):
    r"""This is the quantized version of :class:`~torch.nn.MaxPool2d`.
        Args: Same as torch.nn.MaxPool2d
    """
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices: bool = False,
                 ceil_mode: bool = False) -> None:
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                        return_indices, ceil_mode)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

    def _get_name(self):
        return 'QuantizedMaxPool2d'

class MaxPool3d(torch.nn.MaxPool3d):
    r"""This is the quantized version of :class:`~torch.nn.MaxPool3d`.
        Args: Same as torch.nn.MaxPool3d
    """
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices: bool = False,
                 ceil_mode: bool = False) -> None:
        super(MaxPool3d, self).__init__(kernel_size, stride, padding, dilation,
                                        return_indices, ceil_mode)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.max_pool3d(input, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

    def _get_name(self):
        return 'QuantizedMaxPool3d'


class MaxUnpool1d(torch.nn.MaxUnpool1d):
    r"""This is the quantized version of :class:`~torch.nn.MaxPool3d`.
        Args: Same as torch.nn.MaxUnpool1d
    """
    def __init__(self, kernel_size, stride=None, padding=0) -> None:
        super(MaxUnpool1d, self).__init__(kernel_size, stride, padding)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input, indices, output_size=None) -> Tensor:
        input = self.quant_handle(input)
        return F.max_unpool1d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def _get_name(self):
        return 'QuantizedMaxUnpool1d'


class MaxUnpool2d(torch.nn.MaxUnpool2d):
    r"""This is the quantized version of :class:`~torch.nn.MaxUnpool2d`.
        Args: Same as torch.nn.MaxUnpool2d
    """
    def __init__(self, kernel_size, stride=None, padding=0) -> None:
        super(MaxUnpool2d, self).__init__(kernel_size, stride, padding)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input, indices, output_size=None) -> Tensor:
        input = self.quant_handle(input)
        return F.max_unpool2d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def _get_name(self):
        return 'QuantizedMaxUnpool2d'


class MaxUnpool3d(torch.nn.MaxUnpool3d):
    r"""This is the quantized version of :class:`~torch.nn.MaxUnpool3d`.
        Args: Same as torch.nn.MaxUnpool3d
    """
    def __init__(self, kernel_size, stride=None, padding=0) -> None:
        super(MaxUnpool3d, self).__init__(kernel_size, stride, padding)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input, indices, output_size=None) -> Tensor:
        input = self.quant_handle(input)
        return F.max_unpool3d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def _get_name(self):
        return 'QuantizedMaxUnpool3d'


class AvgPool1d(torch.nn.AvgPool1d):
    r"""This is the quantized version of :class:`~torch.nn.AvgPool1d`.
        Args: Same as torch.nn.AvgPool1d
    """
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True) -> None:
        super(AvgPool1d, self).__init__(kernel_size, stride, padding,
                                        ceil_mode, count_include_pad)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.avg_pool1d(input, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)

    def _get_name(self):
        return 'QuantizedAvgPool1d'


class AvgPool2d(torch.nn.AvgPool2d):
    r"""This is the quantized version of :class:`~torch.nn.AvgPool1d`.
        Args: Same as torch.nn.AvgPool2d
    """
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True,
                 divisor_override: bool = None) -> None:
        super(AvgPool2d,
              self).__init__(kernel_size, stride, padding, ceil_mode,
                             count_include_pad, divisor_override)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad,
                            self.divisor_override)

    def _get_name(self):
        return 'QuantizedAvgPool2d'


class AvgPool3d(torch.nn.AvgPool3d):
    r"""This is the quantized version of :class:`~torch.nn.AvgPool3d`.
        Args: Same as torch.nn.AvgPool3d
    """
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True,
                 divisor_override=None) -> None:
        super(AvgPool3d,
              self).__init__(kernel_size, stride, padding, ceil_mode,
                             count_include_pad, divisor_override)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.avg_pool3d(input, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad,
                            self.divisor_override)

    def _get_name(self):
        return 'QuantizedAvgPool3d'


class FractionalMaxPool2d(torch.nn.FractionalMaxPool2d):
    r"""This is the quantized version of :class:`~torch.nn.FractionalMaxPool2d`
        Args: Same as torch.nn.FractionalMaxPool2d
    """
    def __init__(self,
                 kernel_size,
                 output_size=None,
                 output_ratio=None,
                 return_indices: bool = False,
                 _random_samples=None) -> None:
        super(FractionalMaxPool2d,
              self).__init__(kernel_size, output_ratio, return_indices,
                             _random_samples)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.fractional_max_pool2d(input,
                                       self.kernel_size,
                                       self.output_size,
                                       self.output_ratio,
                                       self.return_indices,
                                       _random_samples=self._random_samples)

    def _get_name(self):
        return 'QuantizedFractionalMaxPool2d'


class LPPool1d(torch.nn.LPPool1d):
    r"""This is the quantized version of :class:`~torch.nn.LPPool1d`
        Args: Same as torch.nn.LPPool1d
    """
    def __init__(self,
                 norm_type,
                 kernel_size,
                 stride=None,
                 ceil_mode: bool = False) -> None:
        super(LPPool1d, self).__init__(norm_type=norm_type,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       ceil_mode=ceil_mode)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.lp_pool1d(input, float(self.norm_type), self.kernel_size,
                           self.stride, self.ceil_mode)

    def _get_name(self):
        return 'QuantizedLPPool1d'


class LPPool2d(torch.nn.LPPool2d):
    r"""This is the quantized version of :class:`~torch.nn.LPPool2d`
        Args: Same as torch.nn.LPPool2d
    """
    def __init__(self,
                 norm_type,
                 kernel_size,
                 stride=None,
                 ceil_mode: bool = False) -> None:
        super(LPPool2d, self).__init__(norm_type=norm_type,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       ceil_mode=ceil_mode)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.lp_pool2d(input, float(self.norm_type), self.kernel_size,
                           self.stride, self.ceil_mode)

    def _get_name(self):
        return 'QuantizedLPPool2d'


class AdaptiveMaxPool1d(torch.nn.AdaptiveMaxPool1d):
    r"""This is the quantized version of :class:`~torch.nn.AdaptiveMaxPool1d`
        Args: Same as torch.nn.AdaptiveMaxPool1d
    """
    def __init__(self, output_size, return_indices: bool = False) -> None:
        super(AdaptiveMaxPool1d, self).__init__(output_size=output_size,
                                                return_indices=return_indices)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.adaptive_max_pool1d(input, self.output_size,
                                     self.return_indices)

    def _get_name(self):
        return 'QuantizedAdaptiveMaxPool1d'


class AdaptiveMaxPool2d(torch.nn.AdaptiveMaxPool2d):
    r"""This is the quantized version of :class:`~torch.nn.AdaptiveMaxPool2d`
        Args: Same as torch.nn.AdaptiveMaxPool2d
    """
    def __init__(self, output_size, return_indices: bool = False) -> None:
        super(AdaptiveMaxPool2d, self).__init__(output_size=output_size,
                                                return_indices=return_indices)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.adaptive_max_pool2d(input, self.output_size,
                                     self.return_indices)

    def _get_name(self):
        return 'QuantizedAdaptiveMaxPool2d'


class AdaptiveMaxPool3d(torch.nn.AdaptiveMaxPool3d):
    r"""This is the quantized version of :class:`~torch.nn.AdaptiveMaxPool3d`
        Args: Same as torch.nn.AdaptiveMaxPool3d
    """
    def __init__(self, output_size, return_indices: bool = False) -> None:
        super(AdaptiveMaxPool3d, self).__init__(output_size=output_size,
                                                return_indices=return_indices)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.adaptive_max_pool3d(input, self.output_size,
                                     self.return_indices)

    def _get_name(self):
        return 'QuantizedAdaptiveMaxPool3d'


class AdaptiveAvgPool1d(torch.nn.AdaptiveAvgPool1d):
    r"""This is the quantized version of :class:`~torch.nn.AdaptiveAvgPool1d`
        Args: Same as torch.nn.AdaptiveAvgPool1d
    """
    def __init__(self, output_size) -> None:
        super(AdaptiveAvgPool1d, self).__init__(output_size)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.adaptive_avg_pool1d(input, self.output_size)

    def _get_name(self):
        return 'QuantizedAdaptiveAvgPool1d'


class AdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    r"""This is the quantized version of :class:`~torch.nn.AdaptiveAvgPool2d`
        Args: Same as torch.nn.AdaptiveAvgPool2d
    """
    def __init__(self, output_size) -> None:
        super(AdaptiveAvgPool2d, self).__init__(output_size)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.adaptive_avg_pool2d(input, self.output_size)

    def _get_name(self):
        return 'QuantizedAdaptiveAvgPool2d'


class AdaptiveAvgPool3d(torch.nn.AdaptiveAvgPool3d):
    r"""This is the quantized version of :class:`~torch.nn.AdaptiveAvgPool3d`
        Args: Same as torch.nn.AdaptiveAvgPool3d
    """
    def __init__(self, output_size) -> None:
        super(AdaptiveAvgPool3d, self).__init__(output_size)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.adaptive_avg_pool3d(input, self.output_size)

    def _get_name(self):
        return 'QuantizedAdaptiveAvgPool3d'

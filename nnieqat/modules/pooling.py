import nnieqat.gpu.quantize as Q
import torch
from torch import Tensor
import torch.nn.functional as F


class MaxPool1d(torch.nn.MaxPool1d):
    r"""A the quantized version of :class:`~torch.nn.MaxPool1d`.

    Args and shape is the same as original version.

    Applies a 1D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        - kernel_size: the size of the window to take a max over
        - stride: the stride of the window. Default value is :attr:`kernel_size`
        - padding: implicit zero padding to be added on both sides
        - dilation: a parameter that controls the stride of elements in the window
        - return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        - ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Examples::

        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
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
    r"""A quantized version of :class:`~torch.nn.MaxPool2d`.

    Args and shape is the same as original version.

    Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        - kernel_size: the size of the window to take a max over
        - stride: the stride of the window. Default value is :attr:`kernel_size`
        - padding: implicit zero padding to be added on both sides
        - dilation: a parameter that controls the stride of elements in the window
        - return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        - ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
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
    r"""A quantized version of :class:`~torch.nn.MaxPool3d`.

    Args and shape is the same as original version.

    Applies a 3D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        - kernel_size: the size of the window to take a max over
        - stride: the stride of the window. Default value is :attr:`kernel_size`
        - padding: implicit zero padding to be added on all three sides
        - dilation: a parameter that controls the stride of elements in the window
        - return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        - ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50,44, 31)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
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
    r"""A quantized version of :class:`~torch.nn.MaxPool3d`.

    Args and shape is the same as original version.

    Computes a partial inverse of :class:`MaxPool1d`.

    :class:`MaxPool1d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool1d` takes in as input the output of :class:`MaxPool1d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: :class:`MaxPool1d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        - kernel_size (int or tuple): Size of the max pooling window.
        - stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        - padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool1d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in})`
        - Output: :math:`(N, C, H_{out})`, where

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{kernel\_size}[0]

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> pool = nn.MaxPool1d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool1d(2, stride=2)
        >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])

        >>> # Example showcasing the use of output_size
        >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices, output_size=input.size())
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.,  0.]]])

        >>> unpool(output, indices)
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])
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
    r"""A quantized version of :class:`~torch.nn.MaxUnpool2d`.

    Args and shape is the same as original version.

    Computes a partial inverse of :class:`MaxPool2d`.

    :class:`MaxPool2d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool2d` takes in as input the output of :class:`MaxPool2d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: :class:`MaxPool2d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        - kernel_size (int or tuple): Size of the max pooling window.
        - stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        - padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool2d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
            H_{out} = (H_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
            W_{out} = (W_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool2d(2, stride=2)
        >>> input = torch.tensor([[[[ 1.,  2,  3,  4],
                                    [ 5,  6,  7,  8],
                                    [ 9, 10, 11, 12],
                                    [13, 14, 15, 16]]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[[  0.,   0.,   0.,   0.],
                  [  0.,   6.,   0.,   8.],
                  [  0.,   0.,   0.,   0.],
                  [  0.,  14.,   0.,  16.]]]])

        >>> # specify a different output size than input size
        >>> unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
        tensor([[[[  0.,   0.,   0.,   0.,   0.],
                  [  6.,   0.,   8.,   0.,   0.],
                  [  0.,   0.,   0.,  14.,   0.],
                  [ 16.,   0.,   0.,   0.,   0.],
                  [  0.,   0.,   0.,   0.,   0.]]]])
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
    r"""A quantized version of :class:`~torch.nn.MaxUnpool3d`.

    Args and shape is the same as original version.

    Computes a partial inverse of :class:`MaxPool3d`.

    :class:`MaxPool3d` is not fully invertible, since the non-maximal values are lost.
    :class:`MaxUnpool3d` takes in as input the output of :class:`MaxPool3d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: :class:`MaxPool3d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs section below.

    Args:
        - kernel_size (int or tuple): Size of the max pooling window.
        - stride (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        - padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool3d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = (D_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          .. math::
              W_{out} = (W_{in} - 1) \times \text{stride[2]} - 2 \times \text{padding[2]} + \text{kernel\_size[2]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> # pool of square window of size=3, stride=2
        >>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool3d(3, stride=2)
        >>> output, indices = pool(torch.randn(20, 16, 51, 33, 15))
        >>> unpooled_output = unpool(output, indices)
        >>> unpooled_output.size()
        torch.Size([20, 16, 51, 33, 15])
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
    r"""A quantized version of :class:`~torch.nn.AvgPool1d`.

    Args and shape is the same as original version.

    Applies a 1D average pooling over an input signal composed of several
    input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \text{out}(N_i, C_j, l) = \frac{1}{k} \sum_{m=0}^{k-1}
                               \text{input}(N_i, C_j, \text{stride} \times l + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be
    an ``int`` or a one-element tuple.

    Args:
        - kernel_size: the size of the window
        - stride: the stride of the window. Default value is :attr:`kernel_size`
        - padding: implicit zero padding to be added on both sides
        - ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        - count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} +
              2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

    Examples::

        >>> # pool with window of size=3, stride=2
        >>> m = nn.AvgPool1d(3, stride=2)
        >>> m(torch.tensor([[[1.,2,3,4,5,6,7]]]))
        tensor([[[ 2.,  4.,  6.]]])
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
    r"""A quantized version of :class:`~torch.nn.AvgPool1d`.

    Args and shape is the same as original version.

    Applies a 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        - kernel_size: the size of the window
        - stride: the stride of the window. Default value is :attr:`kernel_size`
        - padding: implicit zero padding to be added on both sides
        - ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        - count_include_pad: when True, will include the zero-padding in the averaging calculation
        - divisor_override: if specified, it will be used as divisor, otherwise :attr:`kernel_size` will be used

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)
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
    r"""A quantized version of :class:`~torch.nn.AvgPool3d`.

    Args and shape is the same as original version.

    Applies a 3D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
                                              & \frac{\text{input}(N_i, C_j, \text{stride}[0] \times d + k,
                                                      \text{stride}[1] \times h + m, \text{stride}[2] \times w + n)}
                                                     {kD \times kH \times kW}
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides
    for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        - kernel_size: the size of the window
        - stride: the stride of the window. Default value is :attr:`kernel_size`
        - padding: implicit zero padding to be added on all three sides
        - ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        - count_include_pad: when True, will include the zero-padding in the averaging calculation
        - divisor_override: if specified, it will be used as divisor, otherwise :attr:`kernel_size` will be used

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] -
                    \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -
                    \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -
                    \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50,44, 31)
        >>> output = m(input)
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
    r"""A quantized version of :class:`~torch.nn.FractionalMaxPool2d`.

    Args and shape is the same as original version.

    Applies a 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        - kernel_size: the size of the window to take a max over.
                     Can be a single number k (for a square kernel of k x k) or a tuple `(kh, kw)`
        - output_size: the target output size of the image of the form `oH x oW`.
                     Can be a tuple `(oH, oW)` or a single number oH for a square image `oH x oH`
        - output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        - return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :meth:`nn.MaxUnpool2d`. Default: ``False``

    Examples:
        >>> # pool of square window of size=3, and target output size 13x12
        >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _Fractional MaxPooling:
        https://arxiv.org/abs/1412.6071
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
    r"""A quantized version of :class:`~torch.nn.LPPool1d`

    Args and shape is the same as original version.

    Applies a 1D power-average pooling over an input signal composed of several input
    planes.

    On each window, the function computed is:

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    - At p = :math:`\infty`, one gets Max Pooling
    - At p = 1, one gets Sum Pooling (which is proportional to Average Pooling)

    .. note:: If the sum to the power of `p` is zero, the gradient of this function is
              not defined. This implementation will set the gradient to zero in this case.

    Args:
        - kernel_size: a single int, the size of the window
        - stride: a single int, the stride of the window. Default value is :attr:`kernel_size`
        - ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

    Examples::
        >>> # power-2 pool of window of length 3, with stride 2.
        >>> m = nn.LPPool1d(2, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)
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
    r"""A quantized version of :class:`~torch.nn.LPPool2d`.

    Args and shape is the same as original version.

    Applies a 2D power-average pooling over an input signal composed of several input
    planes.

    On each window, the function computed is:

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    - At p = :math:`\infty`, one gets Max Pooling
    - At p = 1, one gets Sum Pooling (which is proportional to average pooling)

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note:: If the sum to the power of `p` is zero, the gradient of this function is
              not defined. This implementation will set the gradient to zero in this case.

    Args:
        - kernel_size: the size of the window
        - stride: the stride of the window. Default value is :attr:`kernel_size`
        - ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} - \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} - \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> # power-2 pool of square window of size=3, stride=2
        >>> m = nn.LPPool2d(2, 3, stride=2)
        >>> # pool of non-square window of power 1.2
        >>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)
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
    r"""A quantized version of :class:`~torch.nn.AdaptiveMaxPool1d`.

    Args and shape is the same as original version.

    Applies a 1D adaptive max pooling over an input signal composed of several input planes.

    The output size is H, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        - output_size: the target output size H
        - return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool1d. Default: ``False``

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveMaxPool1d(5)
        >>> input = torch.randn(1, 64, 8)
        >>> output = m(input)
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
    r"""A quantized version of :class:`~torch.nn.AdaptiveMaxPool2d`.

    Args and shape is the same as original version.

    Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        - output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
        - return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool2d. Default: ``False``

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveMaxPool2d((5,7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveMaxPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveMaxPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
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
    r"""A quantized version of :class:`~torch.nn.AdaptiveMaxPool3d`.

    Args and shape is the same as original version.

    Applies a 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        - output_size: the target output size of the image of the form D x H x W.
                     Can be a tuple (D, H, W) or a single D for a cube D x D x D.
                     D, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

        - return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool3d. Default: ``False``

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveMaxPool3d((5,7,9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveMaxPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveMaxPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)

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
    r"""A quantized version of :class:`~torch.nn.AdaptiveAvgPool1d`.

    Args and shape is the same as original version.

    Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    The output size is H, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        - output_size: the target output size H

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = torch.randn(1, 64, 8)
        >>> output = m(input)


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
    r"""A quantized version of :class:`~torch.nn.AdaptiveAvgPool2d`.

    Args and shape is the same as original version.

    Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        - output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

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
    r"""A quantized version of :class:`~torch.nn.AdaptiveAvgPool3d`.

    Args and shape is the same as original version.

    Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        - output_size: the target output size of the form D x H x W.
                     Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                     D, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveAvgPool3d((5,7,9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """
    def __init__(self, output_size) -> None:
        super(AdaptiveAvgPool3d, self).__init__(output_size)
        self.quant_handle = Q.QuantAndDeQuantGPU()

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        return F.adaptive_avg_pool3d(input, self.output_size)

    def _get_name(self):
        return 'QuantizedAdaptiveAvgPool3d'

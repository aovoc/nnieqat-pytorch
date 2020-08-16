import nnieqat.gpu.quantize as Q
import torch
from torch import Tensor
import torch.nn.functional as F


class Linear(torch.nn.Linear):
    r"""This is the quantized version of :class:`~torch.nn.Linear`.
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args: Same as torch.nn.Linear
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
            additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension are
            the same shape as the input and
            :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias: the learnable bias of the module of shape :math:`(\text{out\_features})`.
            If :attr:`bias` is ``True``, the values are initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.quant_handle = Q.QuantAndDeQuantGPU()
        self.weight_origin = None

    def forward(self, input: Tensor) -> Tensor:
        input = self.quant_handle(input)
        self.weight_origin = self.weight.clone()
        self.weight = self.quant_handle(self.weight)
        return F.linear(input, self.weight, self.bias)

    def _get_name(self):
        return 'QuantizedLinear'

class Bilinear(torch.nn.Bilinear):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1^T A x_2 + b`

    Args: Same as torch.nn.Bilinear
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super(Bilinear, self).__init__(in_features, out_features, bias)
        self.quant_handle = Q.QuantAndDeQuantGPU()
        self.weight_origin = None

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        input1 = self.quant_handle(input1)
        input2 = self.quant_handle(input2)
        self.weight_origin = self.weight.clone()
        self.weight = self.quant_handle(self.weight)
        return F.bilinear(input1, input2, self.weight, self.bias)

    def _get_name(self):
        return 'QuantizedBilinear'

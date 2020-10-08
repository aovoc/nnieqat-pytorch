# -*- coding:utf-8 -*-
import unittest
from ddt import ddt, data
import torch
from torch import nn
from nnieqat import merge_freeze_bn, freeze_bn


@ddt
class TestMergeFreezeBNImpl(unittest.TestCase):
    def conv_bn(inp,
                oup,
                stride,
                conv_layer=nn.Conv2d,
                norm_layer=nn.BatchNorm2d):
        return nn.Sequential(conv_layer(inp, oup, 3, stride, 1, bias=False),
                             norm_layer(oup))

    def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d):
        return nn.Sequential(conv_layer(inp, oup, 1, 1, 0, bias=False),
                             norm_layer(oup))

    data1 = conv_bn(3, 3, 2)
    data2 = conv_1x1_bn(3, 3)

    @data(data1, data2)
    def test(self, m):
        input = torch.randn(1, 3, 10, 10)
        m.eval()
        output_0 = m(input)
        print("module parameter before merge_freeze_bn: ")
        print(list(m.named_parameters()))

        m = merge_freeze_bn(m)
        m.eval()
        output_1 = m(input)
        print("module parameter after merge_freeze_bn: ")
        print(list(m.named_parameters()))

        print("output result before merge_freeze_bn: ")
        print(output_0)
        print("output result after merge_freeze_bn: ")
        print(output_1)
        print("output result diff: ")
        print(output_0 - output_1)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestMergeFreezeBNImpl("test"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

# coding = utf-8
# Arch   = manyArch
#
# @File name:       loss.py
# @brief:           复数损失函数定义
#                   网络输出如果本身是复数的, 那么无法直接和普通优化器相互作用. 一般情况下, 对于复数输出, 我们默认让实部虚部一起逼近标签 (例如下面的 loss_CVpart_stack ).
#                   因此设计了这么两个类, 使用它包裹普通的loss方法即可让任意loss兼容复数网络.
#                   例子:
#                       loss = cpl2.loss_CVpart_stack(torch.nn.BCEWithLogitsLoss())
# @attention:       None
# @Author:          wyb
# @History:         2025-02-13		Create

import torch
from typing import Union, List, Tuple, Dict, Optional


class loss_CVpart_stack(torch.nn.modules.loss._Loss):

    def __init__(self, loss: torch.nn.modules.loss._Loss) -> None:
        """
        复数损失函数, 首先将实部和虚部看作两个通道，之后，按照实数loss计算
        BCE, CE, LLN, MSE等都可以使用这个wrapper一下
        """
        super().__init__()
        self.loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        i = torch.stack([input.real, input.imag], dim=-1)
        t = torch.stack([target, target], dim=-1) # 标签是实数，为了让模型的实部虚部同时逼近，需要如此定义
        l = self.loss(i, t)
        return l


class loss_CVpart_add(torch.nn.modules.loss._Loss):

    def __init__(self, loss: torch.nn.modules.loss._Loss) -> None:
        """
        复数损失函数, 首先将实部和虚部看作两个通道，之后，按照实数loss计算
        不推荐这个loss,因为可能出现严重的精度损失导致模型发散
        """
        super().__init__()
        self.loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        i = input.real + input.imag
        l = self.loss(i, target)
        return l

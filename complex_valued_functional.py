# coding = utf-8
# Arch   = manyArch
#
# @File name:       complex_valued_functional.py
# @brief:           复数基本操作定义, 涉及一些基础模型结构和函数
#
# @attention:       函数名必须小写, 以此区分cpl2本库内的函数与类在外部的调用
# @cite:            None
# @Author:          wyb
# @History:         2024-10-15		Create
#                   2025-02-13      修改为cpl2

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union, List, Tuple, Dict, Optional

######################### 复数基本函数  ##################################################################


def cplx_unity(x: torch.Tensor, dim: List[int] | None = None, epsilon: float = 1e-8) -> torch.Tensor:
    """将复数矩阵x归一化，消除均值并将方差归一化到1"""
    return (x - x.mean(dim=dim, keepdim=True)) / (x.std(dim=dim, keepdim=True) + epsilon)


def relu(x: torch.Tensor) -> torch.Tensor:
    '''max(0, x.real) + 1j * max(0, x.imag)'''
    return F.relu(x.real) + 1j * F.relu(x.imag)


def leaky_relu(x: torch.Tensor, negative_slope=1e-2) -> torch.Tensor:
    '''max(k * x, x.real) + 1j * max(k * x, x.imag) 其中 k = negative_slope 默认1e-2'''
    return F.leaky_relu(x.real, negative_slope) + 1j * F.leaky_relu(x.imag, negative_slope)


def tanh(x: torch.Tensor) -> torch.Tensor:
    '''tanh(x.real) + 1j * tanh(x.imag)'''
    return F.tanh(x.real) + 1j * F.tanh(x.imag)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''sigmoid(x.real) + 1j * sigmoid(x.imag)'''
    return F.sigmoid(x.real) + 1j * F.sigmoid(x.imag)


def sigmoid_C2R(x: torch.Tensor) -> torch.Tensor:
    '''sigmoid(x.abs())'''
    return F.sigmoid(x.abs())


def dropout(x: torch.Tensor, p: int = 0.1, training: bool = True) -> torch.Tensor:
    '''正则化 : 随机丢弃'''
    mask = torch.ones(x.size(), dtype=torch.float32, device=x.device)
    mask = F.dropout(mask, p, training) * 1 / (1 - p)
    return x * mask


def cv_rv_softmax(x: torch.Tensor, temperature: float = 1.0, softmax_dim: int = -1) -> Tensor:
    """按绝对值 : 复数softmax得到一个实数结果 (不存在复数分量) , 可以作为权重或预测概率输出."""
    ret = F.softmax(x.abs() / temperature, dim=softmax_dim)
    return ret


def cv_rv_softmax2(x: torch.Tensor, temperature: float = 1.0, softmax_dim: int = -1) -> Tensor:
    """实部虚部分开然后相加 : 复数softmax得到一个实数结果 (不存在复数分量) , 可以作为权重或预测概率输出."""
    ret = F.softmax((x.real + x.imag) / temperature, dim=softmax_dim)
    return ret


def cv_cv_softmax_arg(x: torch.Tensor, temperature: float = 1.0, softmax_dim: int = -1) -> torch.Tensor:
    """复数softmax得到一个复数结果 , 可以作为权重或预测概率输出. 其中复数的模为 CV_Softmax, 辐角不变. 保留辐角虽然不是概率的含义了, 但是可能有助于模型用于相干能量积累"""
    ret = F.softmax(x.abs() / temperature, dim=softmax_dim)
    ret = ret * cplx_unity(x)
    return ret


def cv_sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''复数sigmoid'''
    output = F.sigmoid(x.abs())
    return output


def absmax_pool1d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, ceil_mode: bool = False) -> torch.Tensor:
    """1D - 按绝对值池化"""
    r = x.real
    i = x.imag
    stride = kernel_size if stride == None else stride
    x_abs2 = x.real.pow(2) + x.imag.pow(2)
    x_max2, x_max2_index = F.max_pool1d(x_abs2, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)
    r0 = r.gather(dim=-1, index=x_max2_index)
    i0 = i.gather(dim=-1, index=x_max2_index)
    x0 = r0 + 1j * i0
    return x0


def absmax_pool2d(x: torch.Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False) -> torch.Tensor:
    """2D - 按绝对值池化"""
    r = x.real
    i = x.imag
    stride = kernel_size if stride == None else stride
    x_abs2 = x.real.pow(2) + x.imag.pow(2)
    x_max2, x_max2_index2 = F.max_pool2d(x_abs2, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)
    x_max2_index = x_max2_index2.flatten(start_dim=-2, end_dim=-1)
    r0, i0 = r.flatten(start_dim=-2, end_dim=-1), i.flatten(start_dim=-2, end_dim=-1)
    r1 = torch.gather(r0, -1, x_max2_index)
    i1 = torch.gather(i0, -1, x_max2_index)
    r1 = r1.view(r.size(0), r.size(1), r.size(2) // stride, r.size(3) // stride)
    i1 = i1.view(i.size(0), i.size(1), i.size(2) // stride, i.size(3) // stride)
    x0 = r1 + 1j * i1
    return x0


def max_pool1d(x: torch.Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> torch.Tensor:
    '''一维最大池化'''
    x_r = F.max_pool1d(x.real, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    x_i = F.max_pool1d(x.imag, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    output = x_r + 1j * x_i
    return output


def max_pool2d(x: torch.Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> torch.Tensor:
    '''二维最大池化'''
    x_r = F.max_pool2d(x.real, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    x_i = F.max_pool2d(x.imag, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    output = x_r + 1j * x_i
    return output


def avg_pool1d(x: torch.Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> torch.Tensor:
    '''一维平均池化'''
    x_r = F.avg_pool1d(x.real, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    x_i = F.avg_pool1d(x.imag, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    output = x_r + 1j * x_i
    return output


def avg_pool2d(x: torch.Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> torch.Tensor:
    '''二维平均池化'''
    x_r = F.avg_pool2d(x.real, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    x_i = F.avg_pool2d(x.imag, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    output = x_r + 1j * x_i
    return output


def upsample(x: torch.Tensor, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None) -> torch.Tensor:
    '''线性上采样 torch.functional.interpolate 的复数版本'''
    x_r = F.interpolate(x.real, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    x_i = F.interpolate(x.imag, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    o = x_r + 1j * x_i
    return o


def get_RoPE_table(n_position: int, d_hid: int, theta=10000.0) -> torch.Tensor:
    """旋转位置编码, 复数形式应该是共轭相乘的时候, 相对距离定义位置编码距离, 而非普通共轭! 

    参数列表 :
        n_position (int): 序列最长的可能长度
        d_hid (int): 隐空间维度
        theta(float): 这个东西建议默认值就好. 最好比 n_pos * d_hid 还大一些. 

    返回值 :
        Tensor: [L, d_hid] 单位复数
    """
    row = theta**(-torch.arange(0, d_hid, dtype=torch.float) / d_hid).view(1, d_hid)
    col = torch.arange(0, n_position, dtype=torch.float).view(n_position, 1)
    p = row * col
    pos_table = torch.cos(p) + 1j * torch.sin(p)
    return pos_table


def interpolate(input: Tensor,
                size: Optional[int] = None,
                scale_factor: Optional[List[float]] = None,
                mode: str = 'nearest',
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None,
                antialias: bool = False) -> torch.Tensor:
    '''上采样'''
    r, i = input.real, input.imag
    r = F.interpolate(r, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
    i = F.interpolate(i, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
    return r + 1j * i


def ri_part_mul(r1, i1, r2, i2):
    '''将实部虚部按复数乘法计算'''
    ro = r1 * r2 - i1 * i2
    io = i1 * r2 + r1 * i2
    return ro, io


def expand_to_batch_dim(x: torch.Tensor, dim: int, at_start: bool = True) -> torch.Tensor:
    """将张量 x 扩展到指定的 batch_dim，其余维度保持为 1。

    参数列表 :
        x (torch.Tensor):           输入的张量
        dim (int):                  对齐到几个维度
        at_start (bool, 可选):      将原本张量维度对齐到前面还是后面. 缺省为 True.

    返回值 :
        torch.Tensor: 扩展后的张量

    例子:
        ```python
        a.size()    # [4]
        expand_to_batch_dim(a,3,False).size()   # [1,1,4]
        ```
    """
    if at_start == True:
        return x.view(x.size(0), *[1] * (dim - 1))
    else:
        return x.view(*[1] * (dim - 1), x.size(0))


######################### 复数基本函数  ##################################################################

######################### 复数到实数变换  ##################################################################


def get_Pr_binary_classify(x: torch.Tensor) -> torch.Tensor:
    '''复数 -> 实数二分类【概率】, 单图分类或二分割/雷达目标检测'''
    return 0.5 * (torch.sigmoid(x.real) + torch.sigmoid(x.imag))


def get_Pr_multi_classify(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    '''复数 -> 实数多分类【概率】, 这里【b,c,h,w】的c维度为类别logits维度, 单图分类或多分割'''
    return 0.5 * (torch.softmax(x.real / temperature, dim=1) + torch.softmax(x.imag / temperature, dim=1))


def get_classIdx_binary_classify(x: torch.Tensor, eta: float = 0.5) -> torch.Tensor:
    '''复数 -> 实数二分类【类别号】, 单图分类或二分割/雷达目标检测, 虚警可控'''
    # ((0.5 * (torch.sigmoid(x.real) + torch.sigmoid(x.imag))) >= 0.5).int()
    # 更高数值稳定性
    log_sigmoid_real = torch.nn.functional.logsigmoid(x.real)
    log_sigmoid_imag = torch.nn.functional.logsigmoid(x.imag)
    log_sum = torch.logsumexp(torch.stack([log_sigmoid_real, log_sigmoid_imag], dim=-1), dim=-1)
    return (log_sum >= torch.log(2 * eta)).int()


def get_classIdx_multi_classify(x: torch.Tensor, eta: List[float] | None = None) -> torch.Tensor:
    '''复数 -> 实数多分类【类别号】, 这里【b,c,h,w】的c维度为类别logits维度, 单图分类或多分割, 可带有虚警控制因子\\eta为一个列表，会将其加到每个类别logits上'''
    if eta is not None:
        xdim = x.dim() - 1
        e = torch.tensor(1 + 1j) * torch.tensor(eta, device=x.device, dtype=x.dtype)
        x = x + e.view(e.shape + (1, ) * (max(0, xdim - e.dim())))
    return torch.argmax((torch.softmax(x.real, dim=1) + torch.softmax(x.imag, dim=1)), dim=1)


def get_Pr_binary_classify_RV(x: torch.Tensor) -> torch.Tensor:
    '''实数二分类【概率】, 单图分类或二分割/雷达目标检测'''
    return torch.sigmoid(x)


def get_Pr_multi_classify_RV(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    '''实数多分类【概率】, 这里【b,c,h,w】的c维度为类别logits维度, 单图分类或多分割'''
    return torch.softmax(x / temperature, dim=1)


def get_classIdx_binary_classify_RV(x: torch.Tensor, eta: float = 0.5) -> torch.Tensor:
    '''实数二分类【类别号】, 单图分类或二分割/雷达目标检测, 带有虚警控制'''
    return (x >= torch.log(eta / (1 - eta))).int()


def get_classIdx_multi_classify_RV(x: torch.Tensor, eta: List[float] | None = None) -> torch.Tensor:
    '''实数多分类【类别号】, 这里【b,c,h,w】的c维度为类别logits维度, 单图分类或多分割, 可带有虚警控制因子\\eta为一个列表，会将其加到每个类别logits上'''
    if eta is not None:
        xdim = x.dim() - 1
        e = torch.tensor(eta, device=x.device, dtype=x.dtype)
        x = x + e.view(e.shape + (1, ) * (max(0, xdim - e.dim())))
    return torch.argmax(x, dim=1)


######################### 复数到实数变换  ##################################################################

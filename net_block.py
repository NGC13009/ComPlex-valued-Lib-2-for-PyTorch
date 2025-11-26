# coding = utf-8
# Arch   = manyArch
#
# @File name:       net_block.py
# @brief:           基本cpl复数网络层.
# @attention:       None
# @cite:            None
# @Author:          wyb
# @History:         2024-10-16		Create

import torch
from typing import List, Tuple, Dict, Union
from . import complex_valued_functional as CVF


class ReLU(torch.nn.Module):
    '''实部虚部同时过非线性函数'''

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.relu(input)


class Sigmoid(torch.nn.Module):
    '''实部虚部同时过非线性函数'''

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.sigmoid(input)


class SigmoidC2R(torch.nn.Module):
    '''返回实数形式 : sigmoid(x.abs())  复数 -> 实数的 sigmoid'''

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.sigmoid_C2R(input)


class ModReLU(torch.nn.Module):
    '''这个激活函数存在一个可设置的偏置参数'''

    def __init__(self, relu_squared=False, bias=False):
        super().__init__()
        self.pow = 2 if relu_squared else 1
        if bias == True:
            self.bias = torch.nn.Parameter(torch.tensor(0.))
        else:
            self.bias = torch.nn.Buffer(torch.tensor(0.))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xabs = x.abs()
        real = torch.nn.functional.relu(xabs + self.bias)**self.pow
        imag = x / xabs
        o = real + imag
        return o


class Dropout(torch.nn.Module):
    '''复数随机丢弃'''

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return CVF.dropout(x, self.p)
        else:
            return x


class MaxPool1d(torch.nn.Module):
    '''一维最大池化'''

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.max_pool1d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices)


class MaxPool2d(torch.nn.Module):
    '''二维最大池化'''

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.max_pool2d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices)


class AbsMaxPool1d(torch.nn.Module):
    """1D - 按绝对值池化"""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.absmax_pool1d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode)


class AbsMaxPool2d(torch.nn.Module):
    """2D - 按绝对值池化"""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.absmax_pool2d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode)


class AvgPool1d(torch.nn.Module):
    '''一维平均池化'''

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.avg_pool1d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices)


class AvgPool2d(torch.nn.Module):
    '''二维平均池化'''

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CVF.avg_pool2d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices)


class AdaptiveAvgPool2d(torch.nn.Module):
    '''自适应平均池化 2d'''

    def __init__(self, output_size):
        super().__init__()
        self.p1 = torch.nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # in:[b,channel,R,D]
        r = x.real
        i = x.imag
        r = self.p1(r)
        i = self.p1(i)
        x = r + 1j * i
        return x


class AdaptiveMaxPool2d(torch.nn.Module):
    '''自适应最大池化 2d'''

    def __init__(self, output_size):
        super().__init__()
        self.p1 = torch.nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # in:[b,channel,R,D]
        r = x.real
        i = x.imag
        r = self.p1(r)
        i = self.p1(i)
        x = r + 1j * i
        return x


class AdaptiveAvgPool1d(torch.nn.Module):
    '''自适应平均池化 1d'''

    def __init__(self, output_size):
        super().__init__()
        self.p1 = torch.nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # in:[b,channel,R,D]
        r = x.real
        i = x.imag
        r = self.p1(r)
        i = self.p1(i)
        x = r + 1j * i
        return x


class AdaptiveMaxPool1d(torch.nn.Module):
    '''自适应最大池化 1d'''

    def __init__(self, output_size):
        super().__init__()
        self.p1 = torch.nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # in:[b,channel,R,D]
        r = x.real
        i = x.imag
        r = self.p1(r)
        i = self.p1(i)
        x = r + 1j * i
        return x


class ChannelMaxPool(torch.nn.Module):
    '''通道最大池化注意力 : CBAM块 https://arxiv.org/abs/1807.06521'''

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, i = x.real, x.imag
        r, _ = torch.max(r, dim=1, keepdim=True)
        i, _ = torch.max(i, dim=1, keepdim=True)
        x = r + 1j * i
        return x


class ChannelAvgPool(torch.nn.Module):
    '''通道平均池化注意力 : CBAM块 https://arxiv.org/abs/1807.06521'''

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=1, keepdim=True)
        return x


class ConvTranspose1d(torch.nn.Module):
    '''转置卷积 1d'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.conv_tran_r = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)

        self.conv_tran_i = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        real = self.conv_tran_r(input.real) - self.conv_tran_i(input.imag)
        imaginary = self.conv_tran_r(input.imag) + self.conv_tran_i(input.real)
        output = real + 1j * imaginary
        return output


class ConvTranspose2d(torch.nn.Module):
    '''转置卷积 2d'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.conv_tran_r = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)

        self.conv_tran_i = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        real = self.conv_tran_r(input.real) - self.conv_tran_i(input.imag)
        imaginary = self.conv_tran_r(input.imag) + self.conv_tran_i(input.real)
        output = real + 1j * imaginary
        return output


class Conv1d(torch.nn.Module):
    '''一维卷积'''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_r = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_mode)
        self.conv_i = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        real = self.conv_r(input.real) - self.conv_i(input.imag)
        imaginary = self.conv_r(input.imag) + self.conv_i(input.real)
        output = real + 1j * imaginary
        return output


class Conv2d(torch.nn.Module):
    '''二维卷积'''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_r = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_mode)
        self.conv_i = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        real = self.conv_r(input.real) - self.conv_i(input.imag)
        imaginary = self.conv_r(input.imag) + self.conv_i(input.real)
        output = real + 1j * imaginary
        return output


class Linear(torch.nn.Module):
    '''全连接层 FC'''

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc_r = torch.nn.Linear(in_features, out_features, bias=bias)
        self.fc_i = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        real = self.fc_r(input.real) - self.fc_i(input.imag)
        imaginary = self.fc_r(input.imag) + self.fc_i(input.real)
        output = real + 1j * imaginary
        return output


class _BatchNorm(torch.nn.Module):
    '''复数BN基类, 内部调用'''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features, 2))

        else:
            self.weight = torch.nn.Parameter(None)
            self.bias = torch.nn.Parameter(None)

        if self.track_running_stats:
            self.running_mean = torch.nn.Buffer(torch.zeros(num_features, 2))
            self.running_covar = torch.nn.Buffer(torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked = torch.nn.Buffer(torch.tensor(0, dtype=torch.long))
        else:
            self.running_mean = torch.nn.Parameter(None)
            self.running_covar = torch.nn.Parameter(None)
            self.num_batches_tracked = torch.nn.Parameter(None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            torch.nn.init.zeros_(self.weight[:, 2])
            torch.nn.init.zeros_(self.bias)


class BatchNorm2d(_BatchNorm):
    '''二维复数BN'''

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_r = input.real
        input_i = input.imag

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:                     # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])
            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r - mean_r[None, :, None, None]
            input_i = input_i - mean_i[None, :, None, None]
            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 0]
                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 1]
                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2] # +self.eps

            input_r = input_r - mean[None, :, 0, None, None]
            input_i = input_i - mean[None, :, 1, None, None]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * \
            input_i, Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r
        if self.affine:
            input_r, input_i = self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None, None] * input_i \
                + self.bias[None, :, 0, None, None], self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None, None] * input_i + self.bias[None, :, 1, None, None]
        output = input_r + 1j * input_i
        return output


class BatchNorm1d(_BatchNorm):
    '''一维复数BN'''

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_r = input.real
        input_i = input.imag
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:                     # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2])
            mean_i = input_i.mean([0, 2])
            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
            # zero mean values
            input_r = input_r - mean_r[None, :, None]
            input_i = input_i - mean_i[None, :, None]
            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2])
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 0]
                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 1]
                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]
            # zero mean values
            input_r = input_r - mean[None, :, 0, None]
            input_i = input_i - mean[None, :, 1, None]
        # calculate the inverse square root the covariance matrix

        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :, None] * input_r + Rri[None, :, None] * input_i, Rii[None, :, None] * input_i + Rri[None, :, None] * input_r
        if self.affine:
            input_r, input_i = self.weight[None, :, 0, None] * input_r + self.weight[None, :, 2, None] * input_i \
                + self.bias[None, :, 0, None], self.weight[None, :, 2, None] * input_r + self.weight[None, :, 1, None] * input_i + self.bias[None, :, 1, None]
        #del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        output = input_r + 1j * input_i
        return output


class LayerNorm(torch.nn.Module):
    '''复数层正则化 LN'''

    def __init__(self, normalized_shape, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if (isinstance(normalized_shape, list)) != True:
            normalized_shape = [normalized_shape] if (isinstance(normalized_shape, int) == True) else list(normalized_shape)

        if self.affine:
            self.weight_r = torch.nn.Parameter(torch.randn(normalized_shape))
            self.weight_i = torch.nn.Parameter(torch.randn(normalized_shape))
            self.bias_r = torch.nn.Parameter(torch.randn(normalized_shape))
            self.bias_i = torch.nn.Parameter(torch.randn(normalized_shape))

        else:
            self.weight_r = torch.nn.Parameter(None)
            self.weight_i = torch.nn.Parameter(None)
            self.bias_r = torch.nn.Parameter(None)
            self.bias_i = torch.nn.Parameter(None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.constant_(self.weight_r, 1.4142135623730951 / 2)
            torch.nn.init.constant_(self.weight_i, 1.4142135623730951 / 2)
            torch.nn.init.zeros_(self.bias_r)
            torch.nn.init.zeros_(self.bias_i)

    def forward(self, input: torch.Tensor) -> torch.Tensor: # 传入需要有通道维度. 建议: [b,channel,L,dk]
        input_r = input.real                                # [b,c,L,dk]
        input_i = input.imag

        # LN是在c,h,w或c,L,dk上进行求取平均什么的
        mean_r = input_r.mean(dim=list(range(1, input_r.dim())))
        mean_i = input_i.mean(dim=list(range(1, input_r.dim())))

        # zero mean values
        input_r = input_r - CVF.expand_to_batch_dim(mean_r, input_r.dim())
        input_i = input_i - CVF.expand_to_batch_dim(mean_i, input_i.dim())

        # 协方差
        n = input_r.numel() / input_r.size(0)
        Crr = 1. / n * input_r.pow(2).sum(dim=list(range(1, input_r.dim()))) + self.eps
        Cii = 1. / n * input_i.pow(2).sum(dim=list(range(1, input_r.dim()))) + self.eps
        Cri = (input_r.mul(input_i)).mean(dim=list(range(1, input_r.dim())))

        # calculate the inverse square root the covariance matrix https://arxiv.org/pdf/1705.09792
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = CVF.expand_to_batch_dim(Rrr, input_r.dim()) * input_r + CVF.expand_to_batch_dim(Rri, input_r.dim()) * \
            input_i, CVF.expand_to_batch_dim(Rii, input_i.dim()) * input_i + CVF.expand_to_batch_dim(Rri, input_i.dim()) * input_r

        if self.affine:
            input_r, input_i = CVF.ri_part_mul(input_r, input_i, self.weight_r, self.weight_i)
            input_r, input_i = input_r + self.bias_r, input_i + self.bias_i
        #del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        output = input_r + 1j * input_i
        return output


class PositionalEncoding_Sinusoid_rv(torch.nn.Module):

    def __init__(self, n_position, d_hid):
        """Transformer 的那个 sin cos 位置嵌入，实数

        Args:
            n_position (int): 嵌入序列长度，必须大于可能遇见的最大序列.
            d_hid (int): 隐藏空间维度
        Forward:
            x + m.pos_table
            x需要为 [..., L, d_hid] 格式，其中x的序列长度不能大于嵌入序列长度
        Return:
                    [..., L, d_hid]
        """
        super().__init__()
        self.pos_table = torch.nn.Buffer(self._get_sinusoid_encoding_table(n_position, d_hid)) # Not a parameter

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        sinusoid = (torch.arange(0, n_position, 1).unsqueeze(0).mT) / (10000.0**torch.tensor([2 * (hid_j // 2) / d_hid for hid_j in range(d_hid)]))
        sinusoid_table = torch.zeros(n_position, d_hid)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid[:, 0::2]) # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid[:, 1::2]) # dim 2i+1
        return torch.FloatTensor(sinusoid_table)

    def forward(self, x) -> torch.Tensor:
        return x + self.pos_table[:x.size(-2), :].clone().detach()


class PositionalEncoding_CRoPE(torch.nn.Module):

    def __init__(self, n_position, d_hid, theta=10000.0):
        """ RoPE 旋转位置编码  复数版本
        注意: 对于复数RoPE，他是相乘的，而非相加到嵌入上
        注意：位置变量在后面应该共轭矩阵乘，不能直接乘。因为相对位置应该是和夹角有关。

        Args:
            n_position (int): 嵌入序列长度，必须大于可能遇见的最大序列.
            d_hid (int): 隐藏空间维度
        Forward:
            x + m.pos_table
            x需要为 [..., L, d_hid] 格式，其中x的序列长度不能大于嵌入序列长度
        Return:
                    [..., L, d_hid]
        """
        super().__init__()
        pos_table = CVF.get_RoPE_table(n_position, d_hid, theta)
        self.pos_table = torch.nn.Buffer(pos_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.pos_table[:x.size(-2), :]


class PositionalEncoding_trainable_mul(torch.nn.Module):
    '''可训练的位置嵌入, 乘法机制'''

    def __init__(self, n_position, d_hid):
        super().__init__()
        theta = 10000.0
        pos_table = CVF.get_RoPE_table(n_position, d_hid, theta)
        self.pos_table = torch.nn.Parameter(pos_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.pos_table[:x.size(-2), :]


class PositionalEncoding_trainable_add(torch.nn.Module):
    '''可训练的位置嵌入, 加法机制'''

    def __init__(self, n_position, d_hid):
        super().__init__()
        theta = 10000.0
        pos_table = CVF.get_RoPE_table(n_position, d_hid, theta).permute(1, 0, 2) # [L,dim]
        self.pos_table = torch.nn.Parameter(pos_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_table[:x.size(-2), :]


class ScaledDotProductAttention(torch.nn.Module):
    '''缩放注意力 MHA中的那个softmax一坨'''

    def __init__(self, temperature=1., attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature # 温度参数  hhttps://arxiv.org/abs/1706.03762
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]: # b x n x lq x dv

        attn_cp = (q / self.temperature) @ (k.mH)
        attn_rv = attn_cp.abs()
        if mask is not None:
            attn_rv = attn_rv.masked_fill(mask == 0, -1e9) # 对于复数的绝对值来说，直接给其一个“负的绝对值”反正在数学上可以令softmax后输出为0，达到目的就行了

        attn_rv = self.dropout(torch.nn.functional.softmax(attn_rv, dim=-1))
        output = attn_rv @ v

        return output, attn_rv # b x n x lq x dv,         b x n x lq x dv


class RV_ScaledDotProductAttention(torch.nn.Module):
    '''flash attention 的放缩点积 : 实数'''

    def __init__(self, temperature=1., attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature # 温度参数  hhttps://arxiv.org/abs/1706.03762
        self.dropout = attn_dropout

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = torch.nn.functional.scaled_dot_product_attention(q / self.temperature, k, v, attn_mask=mask, dropout_p=self.dropout)
        return attn_output


class CMultiHeadAttention(torch.nn.Module):
    '''MHA复数版'''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """一维序列复数多头注意力MHA
        推理输入: q,k,v,mask,三个矩阵。
            q:      [batch , L , dim_q]
            k:      [batch , L , dim_k]
            v:      [batch , L , dim_v]
            mask:   [batch , L ,    L ]
        推理输出：
            q:      就是最终的输出, 复数
            attn:   softmax(qk^T/\\sqrt{d_k})  实数

        Args:
            n_head (int): MHA有几个注意力头
            d_model (int): 输入嵌入维度数目
            d_k (int): k维度
            d_v (int): v维度
            dropout (float, optional): 正则化选项. 默认 0.1.
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = Linear(d_model, n_head * d_v, bias=False)
        self.fc = Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model, eps=1e-6)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1) # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class CFeedForwardNet(torch.nn.Module):
    ''' CFFN: 复数值域的 fead forward 层'''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = Linear(d_in, d_hid) # position-wise
        self.w_2 = Linear(d_hid, d_in) # position-wise
        self.layer_norm = LayerNorm(d_in, eps=1e-6)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.w_2(CVF.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class CEncoderLayer(torch.nn.Module):
    ''' MHSA + FFN '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = CMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = CFeedForwardNet(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class CDecoderLayer(torch.nn.Module):
    ''' MHSA + MHCA + FFN '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = CMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = CMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = CFeedForwardNet(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class CTransformerEncoder(torch.nn.Module):

    def __init__(self, emb_in_dim, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, n_position=30, scale_emb=False):
        """
        A encoder model with self attention mechanism. 复数
        初始化Transformer模型的编码器部分。

        参数:
            chirp_num (int):                输入的chirp数目。作为嵌入用的维度
            d_word_vec (int):               词嵌入的维度。这里是针对需要嵌入的维度做了个仿射变换
            n_layers (int):                 编码器中的层数（通常为6或12）。
            n_head (int):                   多头自注意力中的注意力头的数量。
            d_k (int):                      多头注意力机制中key向量的维度。
            d_v (int):                      多头注意力机制中value向量的维度。
            d_model (int):                  模型维度，表示编码器层的输入和输出维度。
            d_inner (int):                  前馈神经网络中的隐藏层维度。
            pad_idx (int):                  用于mask填充的索引，以忽略填充内容。
            dropout (float, 可选):          Dropout概率，用于防止过拟合，默认为0.1。
            n_position (int, 可选):         位置编码的最大长度，即序列的最大长度. 设置这个需要检查一下我实现的 CPE 支持的最大长度是多少? 不过应该肯定是够用了
                                            对于IPIX我感觉应该是30
            scale_emb (bool, 可选):         是否对嵌入向量进行缩放（√d_model），默认为False。
        """
        super().__init__()

        self.src_word_emb = Linear(emb_in_dim, d_word_vec)
        self.position_enc = PositionalEncoding_CRoPE(n_position=n_position, d_hid=d_word_vec)
        self.dropout = Dropout(p=dropout)
        self.layer_stack = torch.nn.ModuleList([CEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq: torch.Tensor, src_mask=None, return_attns=False) -> torch.Tensor: # [b,chirp,1] -> [b,L,d_word_vec]

        enc_slf_attn_list = []

        enc_output = self.src_word_emb(src_seq) # [b,L,d_word_vec]
        if self.scale_emb:
            enc_output *= self.d_model**0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output, # [b,L,d_word_vec]


class CTransformerDecoder(torch.nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, emb_in_dim, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, n_position=30, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = Linear(emb_in_dim, d_word_vec)
        self.position_enc = PositionalEncoding_CRoPE(d_word_vec, n_position=n_position)
        self.dropout = Dropout(p=dropout)
        self.layer_stack = torch.nn.ModuleList([CDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False) -> torch.Tensor:

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq) # [b,L,d_word_vec]
        if self.scale_emb:
            dec_output *= self.d_model**0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, # [b,L,d_word_vec]


class DepthwiseSeparableConv1d(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, padding_mode: str = 'zeros'):
        """深度可分离卷积（Depthwise Separable Convolution）的实现。"""

        super().__init__()
        self.depthwise = Conv1d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False,
                                groups=in_channels,
                                padding_mode=padding_mode,
                                dilation=dilation)
        self.pointwise = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv2d(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, padding_mode: str = 'zeros'):
        """深度可分离卷积（Depthwise Separable Convolution）的实现。"""

        super().__init__()
        self.depthwise = Conv2d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False,
                                groups=in_channels,
                                padding_mode=padding_mode,
                                dilation=dilation)
        self.pointwise = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv1d_RV(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, padding_mode: str = 'zeros'):
        """深度可分离卷积（Depthwise Separable Convolution）的实现。"""

        super().__init__()
        self.depthwise = torch.nn.Conv1d(in_channels=in_channels,
                                         out_channels=in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False,
                                         groups=in_channels,
                                         padding_mode=padding_mode,
                                         dilation=dilation)
        self.pointwise = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv2d_RV(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, padding_mode: str = 'zeros'):
        """深度可分离卷积（Depthwise Separable Convolution）的实现。"""

        super().__init__()
        self.depthwise = torch.nn.Conv2d(in_channels=in_channels,
                                         out_channels=in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False,
                                         groups=in_channels,
                                         padding_mode=padding_mode,
                                         dilation=dilation)
        self.pointwise = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class log_with_clip_rv(torch.nn.Module):
    '''按矩阵等价实数去处理。实部虚部分别直接做对数。相位不会变。我感觉这玩意不靠谱，至少没法直接用'''

    def __init__(self, max=None, min=-10) -> None:
        super().__init__()
        self.max = max
        self.min = min

    def forward(self, x: Union[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = torch.log(x) # 底数是e
        x = torch.clip(x, self.min, self.max)
        return x


class Logarithmic_connection_rv(torch.nn.Module):
    '''实数对数非线性衔接层  https://ieeexplore.ieee.org/document/11243251'''

    def __init__(self, w) -> None:
        """实数对数非线性衔接层
        设置一个衔接点，小于他的原封不动，大于他的用一个衔接点连续的对数替代。
        如此一来，可以处理特别大的范围的情况。一般来说比较大的地方本身是很突出的
        神经网络嘛，受限于机理，动态范围也就那样了。所以考虑用这么个结构去处理高动态范围
        <p><pre><code>
        ^                                                                  
        |                                                                  
        |                                                                  
        |                              [对数部分]                   .      
        |                           *                                      
        |             *                                                    
        |        X      <--- 衔接点, 由x轴坐标 w 定义                      
        |     .                                                            
        |  .       <--- 线性部分                                           
        . -------X-------------------------------------------------------> 
        </code></pre></p>
        Args:
            w (float): 衔接点位置. Defaults to 7.
        """
        super().__init__()
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor: # [b,c,h,w] -> [b,c,h,w]
        mask = x > self.w
        xge = x * mask + ((~mask) * self.w + 1)
        xls = x * (~mask)
        xlog = self.w + torch.log(1 - self.w + xge)
        xlog = xlog * mask
        x = xlog + xls
        return x


class Logarithmic_connection(torch.nn.Module):
    '''复数对数非线性层 https://ieeexplore.ieee.org/document/11243251'''

    def __init__(self, w: float = 15) -> None:
        super().__init__()
        self.w = w
        self.LC = Logarithmic_connection_rv(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # [b,2,c,h,w] -> [b,2,c,h,w]
        xabs = x.abs()
        xabslog = self.LC(xabs)
        xfactor = xabs / xabslog                        # 矩阵点除, 得到变化系数
        x = x / xfactor
        return x

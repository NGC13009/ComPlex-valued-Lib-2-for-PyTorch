# ComPlex-valued-Lib-2-for-PyTorch 复数网络构件库

## 简介

complex_lib (cpl2) 封装了原生PyTorch复数，并提供了一些基础与流行的算子，模块，网络，损失函数。

该库的开发过程基本涵盖了应有的单元测试与系统测试, 并在一些项目, 工程或论文中起到了作用同时得到了验证. 然而, 难以避免仍旧有一些构件可能是存在缺陷或bug.

该库建议使用数据类型为`torch.complex64`的精度 (本质相当于2个float32), 对于`torch.complex32`, 一些算子可能不支持或存在限制.

当前版本: v2.1.0

> 之所以是cpl2而不是cpl, 是因为在遥远的古代, PyTorch还没原生支持复数, 那个时候我把这个库叫cpl (实部虚部是放在一个额外的维度的), 后来支持复数就整体升级了. 最早的版本没保留.

作者在一篇论文中使用了这部分代码, 如果代码对您的工作有帮助，请考虑引用它, 谢谢: [![DOI](https://img.shields.io/badge/DOI-10.1109%2FIGARSS55030.2025.11243251-blue)](https://doi.org/10.1109/IGARSS55030.2025.11243251)

```latex
Y. Wang et al., "LCB-CV-UNet: Enhanced Detector for High Dynamic Range Radar Signals," IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium, Brisbane, Australia, 2025, pp. 6050-6054, doi: 10.1109/IGARSS55030.2025.11243251. keywords: {Radar remote sensing;Computational modeling;Simulation;Radar detection;Radar;Object detection;Coherence;Robustness;High dynamic range;Signal to noise ratio;High dynamic range radar signals;radar target detection;phase coherence preservation;lightweight models;semi-synthetic dataset},
```

### 安装

```powershell
PS > pip install .               # 安装到当前 conda 环境下
PS > pip uninstaller cpl2        # 卸载. 如果升级有问题....
```

将源码部署到本地

```bash
git clone https://github.com/NGC13009/ComPlex-valued-Lib-2-for-PyTorch.git
```

将源码部署到项目中的库文件内（按照git子库方式管理）

```bash
# 根据需要替换成你的项目目录, 使用文件时, 一般推荐将这个库的名字命名为 `cpl2`
git submodule add https://github.com/NGC13009/ComPlex-valued-Lib-2-for-PyTorch.git ./lib/cpl2
git submodule init
git submodule update --remote
```

### 使用

如果已经 pip 安装:

```python
import torch
import cpl2
```

如果想复制粘贴到项目本地调用:

```python
# 如果需要从子目录访问上级目录的包, 类似于 c/c++ 的 #include "../src/abc.h"
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# 如果不需要引入上级目录, 那么没必要这一段代码

# 引入某个文件夹下的包 (每层必须有一个 __init__.py  文件, 如果没有那就创建一个空文件, 叫这个名字)
from path.to import cpl2

# 然后正常使用
```

目前提供了类似于PyTorch中, torch.nn的接口, 即你可以用下面的方式调用复数部件:

```python
import cpl2             # 类似于 import torch.nn
c1 = cpl2.Conv1d(...)   # 类似于 c1 = torch.nn.Conv1d(...)

# 类似于 import torch.nn.functional as F, 因为是复数值域的(complex valued) 因此是CVF
import cpl2.complex_valued_functional as CVF
a = torch.randn(1,2,3,4, dtype=torch.complex64)
b = CVF.sigmoid(a)      # 类似于 b = F.sigmoid(a)
```

所有函数应该都是见字明意的(并不复杂), 或者带有注释说明 (目前都是中文的). 源码目录如下:

```text
cpl2/
├── readme.md                       # 项目的说明文档，介绍库的功能、用法和安装方式
├── __init__.py                     # 包初始化文件，定义模块导出和包元数据
├── license                         # GPLv3 许可
├── loss.py                         # 用于模型训练损失函数的类
├── complex_valued_functional.py    # 用于复数值张量操作的函数
├── net_block.py                    # 定义神经网络中的模块或块（如正则化、卷积块等）
├── requirements.txt                # 依赖的第三方 Python 包及其版本
└── setup.py                        # 使用 setuptools 配置库的打包和安装方式
```

--------------------------------

## 复数网络的核心原理?

### 优化方法

复数网络的训练基于魏廷格微分定义的形式导数梯度，使用普通优化器进行优化，请参考[此处](https://pytorch.org/docs/stable/notes/autograd.html)。

此处不介绍复数神经网络的数学细节, 仅介绍本框架内, 复数网络被实现的核心原理.

### 实现方法

#### PyTorch已经实现了什么

PyTorch在比较新的版本中添加了`torch.complex` 类型，包括多种精度，例如，`torch.complex64` 实际上就是实部虚部为 `torch.float32` 构成的复数张量。

PyTorch实现了复数张量的一些 $ \mathbb{C} $ 上的运算，例如共轭转置，计算绝对值，方差，等。甚至高级的，例如 `toech.fft.fft` 之类的函数也被实现了，并且获得了来自 `cuDNN`, `cuBLAS`等库的加速支持。

#### PyTorch没有实现什么

PyTorch没有在标准模块内支持复数，也没有在标准模块内支持实现全自动复数求导，所以网络层以及loss需要人为定义。按照[此处](https://pytorch.org/docs/stable/notes/autograd.html)的方法，以及[这个论文](https://ieeexplore.ieee.org/document/8495012)，我们在每一个基本的module（例如CNN，BN，LN，激活函数，pooling等）内部内，将复数拆分成实部和虚部 `x.real` , `x.imag` 去进行运算，按照复数的运算法则运算后，再拼成复数。如此，我们重新封装了所有复数操作，使用这些模块，即可当作普通的模块用于任何复数输入输出的模型中。

## 复数网络的提升有多大

我们做了一些实验，以及根据相关的论文研究，基本上是：

1. 由于复数运算实际上和矩阵运算同构，所以将复数的实部虚部拆分后送入神经网络，进行训练（此时模型的输入大小变为原尺寸的两倍，因为实部虚部看作两个不同的数），得到的效果，理论上是和复数网络相同的。
2. 试验下来，发现确实能逼近和复数网络相似的性能，但是略差于复数网络的结果。
3. 同构实数网络训练时间更长，收敛更困难，这应该是因为模型需要额外学习到复数运算的性质导致的。而复数网络已经被人工实现了这些功能，所以复数网络在理论上收敛性是更优的。也就是更易被训练得到更好效果。
4. 复数网络由于复数运算，会带来两倍的参数量以及大概最多四倍的计算量（主要是在矩阵乘法上）。而使用实数网络，输入参数与模型参数量都应该是两倍（与复数网络相同），计算量按照卷积，pooling，ReLU，BN等去估计的话，大概只有两倍。所以，实数网络为了获得逼近复数网络性能，必须增加额外的计算量与模型层。因为更深的层才能拟合复数运算那些更复杂的函数关系，例如乘法表示相位相加等。信号领域常使用这些性质，所以复数网络在信号处理上具有优势。
5.

总结一下表格：

| 方面    | 实数网络 (RNN)   | 复数网络 (CNN)  |
|-------|----------|------------------|
| **输入大小**     | 原始尺寸的两倍（实部和虚部分开作为单独输入）     | 原始尺寸             |
| **性能**           | 可以逼近复数网络的性能但略差        | 更好的性能                      |
| **训练时间/收敛性**    | 较长的训练时间，更难收敛         | 较短的训练时间，更容易收敛   |
| **参数数量**        | 原始的两倍（与复数网络相同）       | 原始                           |
| **计算负载**      | 大约是原始的两倍（由于需要更深的层来处理复杂操作）  | 在矩阵乘法上大约是原始的四倍，但在其他操作上只有两倍  |
| **信号处理中的优势** | 需要额外的层和计算资源来逼近复数网络的性能     | 自然地处理相位关系和其他复杂操作，在信号处理中具有优势 |

## Acknowledgments

这个库中的一些代码可能修改自互联网上他人的公开代码 (例如GitHub, 知乎等平台), 然而由于这些代码是从一个持续时间很久的大工程项目中拆除来的, 导致这些来源已经不可考证了, 我尝试寻找然而未果, 在此感谢这些愿意分享代码的作者.

2025年11月27日

# ComPlex-valued-Lib-2-for-PyTorch : A basic complex-valued network library written for PyTorch

点击[此处](readme.md)查看汉语说明.

## Brief

complex_lib (cpl2) wraps native PyTorch complex numbers and provides some basic and popular operators, modules, networks, and loss functions.

The development of this library has basically covered the necessary unit tests and system tests, and it has been used in some projects, engineering, or papers and verified. However, it is inevitable that some components may still have defects or bugs.

The library recommends using the data type `torch.complex64` (essentially equivalent to 2 float32s). For `torch.complex32`, some operators may not be supported or have limitations.

Current version: v2.1.0

> The reason it is called cpl2 instead of cpl is that in ancient times, PyTorch did not natively support complex numbers, and I called this library cpl (the real and imaginary parts were placed in an additional dimension). Later, when PyTorch supported complex numbers, the library was upgraded overall. The earliest version is not preserved.

The author used part of this code in a paper. If the code is helpful for your work, please consider citing it. Thank you.: [![DOI](https://img.shields.io/badge/DOI-10.1109%2FIGARSS55030.2025.11243251-blue)](https://doi.org/10.1109/IGARSS55030.2025.11243251)

```latex
Y. Wang et al., "LCB-CV-UNet: Enhanced Detector for High Dynamic Range Radar Signals," IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium, Brisbane, Australia, 2025, pp. 6050-6054, doi: 10.1109/IGARSS55030.2025.11243251. keywords: {Radar remote sensing;Computational modeling;Simulation;Radar detection;Radar;Object detection;Coherence;Robustness;High dynamic range;Signal to noise ratio;High dynamic range radar signals;radar target detection;phase coherence preservation;lightweight models;semi-synthetic dataset},
```

### Setup

```powershell
PS > pip install .               # setup to current conda env.
PS > pip uninstaller cpl2        # remove if needed
```

Deploy the source code locally

```bash
git clone https://github.com/NGC13009/ComPlex-valued-Lib-2-for-PyTorch.git
```

Deploy the source code into the project's library files (managed as a Git submodule)

```bash
# Replace it with your project directory as needed. When using the file, it is generally recommended to name this library `cpl2`
git submodule add https://github.com/NGC13009/ComPlex-valued-Lib-2-for-PyTorch.git ./lib/cpl2
git submodule init
git submodule update --remote
```

### Usage

if install with pip:

```python
import torch
import cpl2
```

If you want to copy and paste for local project usage:

```python
# If you need to access packages in the parent directory from a subdirectory, similar to `#include "../src/abc.h"` in C/C++
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# If you don't need to import packages from the parent directory, this section of code is not necessary.

# Import packages from a specific folder (each level must have a `__init__.py` file. If it doesn't exist, create an empty file with this name)
from path.to import cpl2

# then use it
```

Currently, it provides an interface similar to `torch.nn` in PyTorch, meaning you can call complex components in the following way:

```python
import cpl2             # Similar to import torch.nn
c1 = cpl2.Conv1d(...)   # Similar to c1 = torch.nn.Conv1d(...)

# Similar to import torch.nn.functional as F, since it is in the complex valued domain, it is named CVF
import cpl2.complex_valued_functional as CVF
a = torch.randn(1,2,3,4, dtype=torch.complex64)
b = CVF.sigmoid(a)      # Similar to b = F.sigmoid(a)
```

All functions should be self-explanatory (not complex), or accompanied by comments (currently in Chinese). The source code directory structure is as follows:

```text
cpl2/
├── readme.md                       # Project documentation, introducing the library's functionality, usage, and installation method
├── __init__.py                     # Package initialization file, defining module exports and package metadata
├── license                         # GPLv3 license
├── loss.py                         # Classes for loss functions used in model training
├── complex_valued_functional.py    # Functions for operations on complex-valued tensors
├── net_block.py                    # Defines modules or blocks in neural networks (e.g., regularization, convolution blocks, etc.)
├── requirements.txt                # Third-party Python packages and their versions that the library depends on
└── setup.py                        # Configuration file for library packaging and installation using setuptools
```

--------------------------------

## What is the core principle of the complex network?

### Optimization Methods

The training of complex networks is based on the gradient of the formal derivative defined by Wirtinger's calculus, and standard optimizers are used for optimization. Please refer to [this link](https://pytorch.org/docs/stable/notes/autograd.html).

This section does not introduce the mathematical details of complex neural networks, but only explains the core principles of how complex networks are implemented within this framework.

### Implementation Method

#### What PyTorch Has Implemented

In newer versions, PyTorch has added the `torch.complex` type, including various precisions. For example, `torch.complex64` is actually a complex tensor composed of `torch.float32` for the real and imaginary parts.

PyTorch has implemented some operations on $ \mathbb{C} $, such as conjugate transpose, absolute value calculation, variance, etc. Even advanced operations like `torch.fft.fft` have been implemented and are accelerated by libraries such as `cuDNN` and `cuBLAS`.

#### What PyTorch Has Not Implemented

PyTorch does not support complex numbers in standard modules, nor does it support automatic differentiation for complex numbers in standard modules. Therefore, network layers and loss functions need to be manually defined. Following the method described [here](https://pytorch.org/docs/stable/notes/autograd.html) and the paper [here](https://ieeexplore.ieee.org/document/8495012), we decompose complex numbers into real and imaginary parts (`x.real`, `x.imag`) within each basic module (e.g., CNN, BN, LN, activation functions, pooling, etc.), perform operations based on complex number rules, and then recombine them into complex numbers. In this way, we have rewrapped all complex operations, and these modules can be used as regular modules in any model with complex inputs and outputs.

## How Much Improvement Does the Complex Network Offer

We have conducted some experiments and reviewed related research papers, and the results are basically as follows:

1. Since complex operations are isomorphic to matrix operations, splitting the real and imaginary parts of complex numbers and feeding them into a neural network for training (at this point, the input size of the model doubles because the real and imaginary parts are treated as two different numbers) theoretically produces the same effect as a complex network.
2. Experiments have shown that it can indeed approximate the performance of a complex network, but the results are slightly worse than those of the complex network.
3. The isomorphic real network takes longer to train and is more difficult to converge, likely because the model needs to learn the properties of complex operations. In contrast, the complex network has these functions manually implemented, so the complex network theoretically has better convergence properties, making it easier to train for better results.
4. Due to complex operations, the complex network has twice the number of parameters and approximately up to four times the computational cost (mainly in matrix multiplication). In contrast, the real network has twice the number of input parameters and model parameters (the same as the complex network), and the computational cost is approximately twice as much when estimated based on convolution, pooling, ReLU, BN, etc. Therefore, to achieve performance comparable to the complex network, the real network must increase additional computational cost and model layers. Deeper layers are needed to fit the more complex functional relationships of complex operations, such as multiplication representing phase addition. These properties are commonly used in signal processing, giving the complex network an advantage in signal processing applications.

In summary, the table below shows the comparison:

| Aspect         | Real Network (RNN)                | Complex Network (CNN)              |
|----------------|----------------------------------|------------------------------------|
| **Input Size**       | Twice the original size (real and imaginary parts are treated as separate inputs) | Original size                      |
| **Performance**        | Can approximate the performance of the complex network but is slightly worse | Better performance                  |
| **Training Time/Convergence** | Longer training time, harder to converge | Shorter training time, easier to converge |
| **Number of Parameters** | Twice the original (same as the complex network) | Original                            |
| **Computational Load** | Approximately twice the original (due to deeper layers needed to handle complex operations) | Approximately four times the original in matrix multiplication, but only twice in other operations |
| **Advantage in Signal Processing** | Requires additional layers and computational resources to approximate the performance of the complex network | Naturally handles phase relationships and other complex operations, offering advantages in signal processing |

## Acknowledgments

Some of the code in this library may have been modified from publicly available code by others on the internet (e.g., GitHub, Zhihu, etc.). However, since this code was extracted from a long-running large project, the original sources are now untraceable. I have attempted to find them but was unsuccessful. Here, I extend my gratitude to the authors who are willing to share their code.

November 27, 2025

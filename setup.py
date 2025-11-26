# coding = utf-8
# Arch   = manyArch
#
# @File name:       setup.py
# @brief:           使用pip安装cpl包, 便于升级管理
# @attention:       None
# @cite:            None
# @Author:          wyb
# @History:         2024-10-16		Create
#                   2025-02-13      修改为cpl2
#                   2025-02-25      重构后更改安装方法

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='cpl2',
    version='2.1.0',
    packages=[''],
    package_dir={'': '.'},
    url='https://github.com/NGC13009/ComPlex-valued-Lib-2-for-PyTorch.git',
    license='GPLv3',
    author='NGC13009',
    author_email='ngc1300@126.com',
    description='ComPlex-valued-Lib-2-for-PyTorch : A basic complex-valued network library written for PyTorch.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)

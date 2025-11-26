# coding = utf-8
# Arch   = manyArch
#
# @File name:       __init__.py
# @brief:           cpl2 的主函数文件
#                   * 文档说明: ./readme.md
# @attention:       None
# @cite:            None
# @Author:          wyb
# @History:         2024-10-16		Create
#                   2025-02-13      修改为cpl2

from . import complex_valued_functional as CVF # 函数不应该被引入cpl2的全局变量空间
from . import complex_valued_functional
from .net_block import *
from .loss import *

__use_complex_lib_2 = True

if __name__ == "__main__":
    print("使用复数库。建议:")
    print('import cpl2')

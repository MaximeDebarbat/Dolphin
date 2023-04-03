"""
Dolphin: A Python package for CUDA-accelerated data processing,
fast deep learning inference and more.

debarbat.maxime@gmail.com
https://www.linkedin.com/in/mdebarbat/
"""


from .cutils.cuda_base import CudaBase
from .core.dtype import dtype
from .core.darray import (
    darray,
    zeros,
    zeros_like,
    empty,
    transpose,
    add,
    multiply,
    divide,
    reversed_divide,
    substract,
    reversed_substract,
    ones
)
from .core.dimage import dimage
from .cutils.bindings import CudaBinding
from .cutils import cufunc

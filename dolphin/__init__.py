"""
Dolphin: A Python package for CUDA-accelerated data processing,
fast deep learning inference and more.

debarbat.maxime@gmail.com
https://www.linkedin.com/in/mdebarbat/
"""


from .core.dtype import dtype
from .core.darray import darray
from .core.dimage import dimage
from .cutils.bindings import CudaBinding
from .cutils.cuda_base import CudaBase

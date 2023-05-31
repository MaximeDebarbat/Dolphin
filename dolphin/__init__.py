"""
Dolphin
=======

A python package for GPU-accelerated data processing for TensorRT inference.
Dolphin notably provides a set of functions to manipulate GPU arrays
(dolphin.darray) and images (dolphin.dimage) and TensorRT functions
(dolphin.Engine).

This package is strongly relying on the CUDA Python bindings PyCuda,
available at https://github.com/inducer/pycuda,
https://documen.tician.de/pycuda/.
And TensorRT, available at https://developer.nvidia.com/tensorrt.
"""

from .version import __version__
import pycuda.autoinit
import pycuda.driver as cuda  # pylint: disable=import-error

from .cutils.cuda_base import CudaBase
from .cutils.cudarray import (
  CuFillCompiler,
  AXpBZCompiler,
  AXpBYZCompiler,
  EltwiseMultCompiler,
  EltwiseDivCompiler,
  ScalDivCompiler,
  InvScalDivCompiler,
  EltWiseCastCompiler,
  EltwiseAbsCompiler,
  DiscontiguousCopyCompiler
)
from .cutils.cudimage import (
  CuResizeCompiler,
  CuNormalizeCompiler,
  CuCvtColorCompiler
)

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
    ones,
    ones_like,
    empty_like,
    absolute,
    from_numpy
)
from .core.dimage import (dimage,
                          dimage_dim_format,
                          dimage_channel_format,
                          dimage_resize_type,
                          dimage_normalize_type,
                          resize,
                          resize_padding,
                          cvtColor,
                          normalize)
from .core.bufferizer import Bufferizer
from .core.trtbuffer import CudaTrtBuffers
from .TrtWrapper.Engine import Engine

globals().update(dimage_dim_format.__members__)
globals().update(dimage_channel_format.__members__)
globals().update(dimage_resize_type.__members__)
globals().update(dimage_normalize_type.__members__)
globals().update(dtype.__members__)


def Stream(flags: cuda.event_flags = 0):
    """Wraps PyCUDA's Stream class in order not to expose the PyCUDA module,
    for the sake of clarity.
    Please refer to the official PyCUDA documentation for more information.
    https://documen.tician.de/pycuda/driver.html#pycuda.driver.Stream

    :param flags: Flag, defaults to 0
    :type flags: cuda.event_flags, optional
    :return: pycuda.driver.Stream object
    :rtype: pycuda.driver.Stream
    """

    return cuda.Stream(flags)

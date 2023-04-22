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

dolphin.darray
--------------

darrays has been implemented to clone the `numpy.ndarray` interface.
It is important to note that darrays are not `numpy.ndarray` objects
although there exist some bindings and functions to convert them to
`numpy.ndarray` objects and vice-versa.

Sample usage::

  import dolphin as dp
  cuda_array = dp.zeros((10, 10), dtype=dp.float32)
  cuda_array += 255
  print(cuda_array)

Also, dolphin provides a set of functions to manipulate `dolphin.darray`::

  import dolphin as dp
  np_array = np.random.rand(10, 10)
  stream = dp.Stream()
  cuda_array = dp.darray(numpy, dtype=dp.float32)
  cuda_array = dp.add(cuda_array, 255)

- Optimization note

A really important part of the optimization of the code regarding
`dolphin.darray` is the use of streams and mostly the use of
already allocated memory and dolphin darray functions::

  import dolphin as dp
  cuda_array = dp.zeros((10, 10), dtype=dp.float32)
  cuda_array = (cuda_array.transpose(1, 0)*50)*cuda_array

Is not going to be as efficient (at least in terms of latency)
as the following code::

  import dolphin as dp
  cuda_array = dp.zeros((10, 10), dtype=dp.float32)
  cuda_array_result = dp.zeros_like(cuda_array)
  dp.transpose((1, 0), cuda_array, cuda_array_result)
  dp.multiply(50, cuda_array_result, cuda_array_result)
  dp.multiply(cuda_array, cuda_array_result, cuda_array_result)

It is more memory consuming but it is way more efficient in terms of
latency.

For questions, suggestions, bug reports you can contribute directly on Dolphin's
github repository:
https://github.com/MaximeDebarbat/Dolphin

Or reach me at:

debarbat.maxime@gmail.com
https://www.linkedin.com/in/mdebarbat/
"""

import pycuda.autoinit
import pycuda.driver as cuda  # pylint: disable=import-error

from .cutils.cuda_base import CudaBase
from .cutils.cuda_base_new import CudaBaseNew
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
    abs
)
from .core.dimage import (dimage,
                          dimage_dim_format,
                          dimage_channel_format,
                          dimage_resize_type,
                          dimage_normalize_type,
                          resize,
                          cvtColor,
                          normalize)
from .cutils import cudarray, cudimage
from .core.bufferizer import Bufferizer
from .core.trtbuffer import CudaTrtBuffers
from .cutils.bindings import CudaBinding
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
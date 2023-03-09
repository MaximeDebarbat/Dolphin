
import sys
from functools import reduce

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import numpy as np

from bindings import CudaBinding, HostDeviceMem


class Bufferizer:

    def __init__(self, shape: tuple,
                 buffer_size: int,
                 dtype: np.dtype):

        self._shape = shape
        self._dtype = dtype
        self._size = buffer_size
        self._itemsize = trt.volume(self._shape)
        self._buffer = None

        self.allocate()

    def allocate(self) -> None:
        """_summary_
        """

        host = cuda.pagelocked_zeros(
            trt.volume(self._shape), self._dtype)
        device = cuda.mem_alloc(host.nbytes)
        self._buffer = HostDeviceMem(host, device)
        self._n_elements = 0

    def append_one(self, binding: CudaBinding,
                   stream: cuda.Stream = None) -> None:
        """_summary_

        :param binding: _description_
        :type binding: CudaBinding
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        if self._n_elements == self._size:
            raise BufferError(f"Bufferizer is full ({self._size} elements).")

        if stream is None:
            cuda.memcpy_dtod(self._buffer.device + self._n_elements
                             * self._itemsize,
                             binding.device,
                             self._itemsize*np.dtype(self._dtype).itemsize)
        else:
            cuda.memcpy_dtod_async(self._buffer.device + self._n_elements
                                   * self._itemsize,
                                   binding.device,
                                   self._itemsize *
                                   np.dtype(self._dtype).itemsize,
                                   stream)

        self._n_elements += 1

    def append_multiple(self, binding: CudaBinding,
                        stream: cuda.Stream = None) -> None:
        """_summary_

        :param binding: _description_
        :type binding: CudaBinding
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """
        batch_size = binding.shape[0]

        if self._n_elements == self._size:
            raise BufferError(f"Bufferizer is full (max {self._size} elements).")

        if self._n_elements + batch_size > self._size:
            raise BufferError(f"Bufferizer is full ({self._size} elements).\
                              Tried to push {batch_size} elements.")

        if stream is None:
            cuda.memcpy_dtod(self._buffer.device + self._n_elements
                             * self._itemsize,
                             binding.device,
                             batch_size*self._itemsize *
                             np.dtype(self._dtype).itemsize)
        else:
            cuda.memcpy_dtod_async(self._buffer.device + self._n_elements
                                   * self._itemsize,
                                   binding.device,
                                   batch_size*self._itemsize *
                                   np.dtype(self._dtype).itemsize,
                                   stream)

        self._n_elements += batch_size

    def flush(self, value: int = 0,
              stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        if stream is None:
            cuda.memset_d8_async(self._buffer.device,
                                 value,
                                 self._itemsize * self._size)
        else:
            cuda.memset_d8_async(self._buffer.device,
                                 value,
                                 self._itemsize * self._size, stream)

    def d2h(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        pass

    def h2d(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        pass

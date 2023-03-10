"""_summary_
"""

import sys
from functools import reduce

import pycuda.autoinit  # pylint: disable=import-error
import pycuda.driver as cuda  # pylint: disable=import-error
import tensorrt as trt  # pylint: disable=import-error

import numpy as np

from bindings import CudaBinding, HostDeviceMem  # pylint: disable=import-error


class Bufferizer:
    """_summary_
    """

    def __init__(self, shape: tuple,
                 buffer_size: int,
                 dtype: np.dtype,
                 flush_hook: callable = None,
                 allocate_hook: callable = None,
                 append_one_hook: callable = None,
                 append_multiple_hook: callable = None
                 ):

        self._shape = shape
        self._dtype = dtype
        self._size = buffer_size
        self._itemsize = trt.volume(self._shape)
        self._buffer = None
        self._nbytes = 0
        self._n_elements = 0

        self._flush_hook = None
        self._allocate_hook = None
        self._append_one_hook = None
        self._append_multiple_hook = None

        self.allocate()

        self._flush_hook = flush_hook
        self._allocate_hook = allocate_hook
        self._append_one_hook = append_one_hook
        self._append_multiple_hook = append_multiple_hook

    def allocate(self) -> None:
        """_summary_
        """

        host = np.empty(
            trt.volume((self._size,)+self._shape), self._dtype)
        device = cuda.mem_alloc(host.nbytes)  # pylint: disable=no-member
        self._nbytes = host.nbytes
        self._buffer = HostDeviceMem(host, device)
        self._n_elements = 0

        if self._allocate_hook is not None:
            self._allocate_hook()

        self.flush(0)

    def append_one(self, binding: CudaBinding,
                   stream: cuda.Stream = None) -> None:
        # pylint: disable=no-member
        """_summary_

        :param binding: _description_
        :type binding: CudaBinding
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        if self._n_elements == self._size:
            raise BufferError(f"Bufferizer is full ({self._size} elements).")

        if stream is None:
            cuda.memcpy_dtod(int(self._buffer.device) + self._n_elements
                             * self._itemsize,
                             binding.device,
                             self._itemsize * np.dtype(self._dtype).itemsize)
            # pylint: disable=no-member
        else:
            cuda.memcpy_dtod_async(self._buffer.device + self._n_elements
                                   * self._itemsize,
                                   binding.device,
                                   self._itemsize *
                                   np.dtype(self._dtype).itemsize,
                                   stream)  # pylint: disable=no-member

        self._n_elements += 1

        if self._append_one_hook is not None:
            self._append_one_hook()

    def append_multiple(self, binding: CudaBinding,
                        stream: cuda.Stream = None) -> None:
        # pylint: disable=no-member
        """_summary_

        :param binding: _description_
        :type binding: CudaBinding
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """
        batch_size = binding.shape[0]

        if self._n_elements == self._size:
            raise BufferError(f"Bufferizer is full (max \
                                {self._size} elements).")

        if self._n_elements + batch_size > self._size:
            raise BufferError(f"Bufferizer is full ({self._size} elements).\
                              Tried to push {batch_size} elements.")

        if stream is None:
            cuda.memcpy_dtod(self._buffer.device + self._n_elements
                             * self._itemsize,
                             binding.device,
                             batch_size * self._itemsize *
                             np.dtype(self._dtype).itemsize)
            # pylint: disable=no-member
        else:
            cuda.memcpy_dtod_async(self._buffer.device + self._n_elements
                                   * self._itemsize,
                                   binding.device,
                                   batch_size * self._itemsize *
                                   np.dtype(self._dtype).itemsize,
                                   stream)  # pylint: disable=no-member

        self._n_elements += batch_size

        if self._append_multiple_hook is not None:
            self._append_multiple_hook()

    def flush(self, value: int = 0,
              stream: cuda.Stream = None) -> None:
        # pylint: disable=no-member
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        size = self._nbytes

        print(f"Flushing buffer {self._buffer.device} with {size} \
            bytes of value {value}")

        if stream is None:
            cuda.memset_d8(self._buffer.device,
                           value,
                           size)
            # pylint: disable=no-member
        else:
            cuda.memset_d8_async(self._buffer.device,
                                 value,
                                 size, stream)
            # pylint: disable=no-member

        self._n_elements = 0

        if self._flush_hook is not None:
            self._flush_hook()

    def d2h(self, stream: cuda.Stream = None) -> None:
        # pylint: disable=no-member
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        if stream is None:
            cuda.memcpy_dtoh(self._buffer.host,
                             self._buffer.device)  # pylint: disable=no-member
        else:
            cuda.memcpy_dtoh_async(self._buffer.host,
                                   self._buffer.device,
                                   stream=stream)  # pylint: disable=no-member

    def h2d(self, stream: cuda.Stream = None) -> None:
        # pylint: disable=no-member
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        if stream is None:
            cuda.memcpy_htod(self._buffer.device,
                             self._buffer.host)  # pylint: disable=no-member
        else:
            cuda.memcpy_htod_async(self._buffer.device,
                                   self._buffer.host,
                                   stream=stream)  # pylint: disable=no-member

    def flush_hook(self, hook: callable):
        """_summary_

        :param hook: _description_
        :type hook: callable
        """
        self._flush_hook = hook

    def allocate_hook(self, hook: callable):
        """_summary_

        :param hook: _description_
        :type hook: callable
        """
        self._allocate_hook = hook

    def append_one_hook(self, hook: callable):
        """_summary_

        :param hook: _description_
        :type hook: callable
        """
        self._append_one_hook = hook

    def append_multiple_hook(self, hook: callable):
        """_summary_

        :param hook: _description_
        :type hook: callable
        """
        self._append_multiple_hook = hook

    @property
    def value(self) -> np.ndarray:
        """_summary_

        :return: _description_
        :rtype: np.ndarray
        """
        return self._buffer.host.reshape((self._size,) + self._shape)

    @property
    def element_nbytes(self) -> int:
        """_summary_

        :return: _description_
        :rtype: int
        """
        return self._itemsize

    @property
    def nbytes(self) -> int:
        """_summary_

        :return: _description_
        :rtype: int
        """
        return self._nbytes

    def __len__(self):
        return self._n_elements


if __name__ == "__main__":

    buffer = Bufferizer((2,), buffer_size=5, dtype=np.uint8)
    buffer.flush()

    data = CudaBinding()
    data.allocate((2,), np.uint8)
    data.write(np.array([1, 2], dtype=np.uint8))
    data.h2d()

    buffer.append_one(data)
    buffer.append_one(data)
    buffer.append_one(data)

    buffer.d2h()

    print(buffer.value)
    print(len(buffer))

    buffer.flush()
    buffer.d2h()

    print(buffer.value)
    print(len(buffer))

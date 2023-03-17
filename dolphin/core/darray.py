
import pycuda.driver as cuda  # pylint: disable=import-error

import numpy
from dolphin import dtype
from dolphin.cutils import CudaBinding


class darray:
    """_summary_
    """

    def __init__(self,
                 array: numpy.ndarray = None,
                 shape: tuple = None,
                 dtype: dtype = None,
                 stream: cuda.Stream = None
                 ) -> None:

        self._shape = shape
        self._dtype = dtype
        self._stream = stream

        self._binding = CudaBinding()
        self._binding.allocate(shape=self._shape,
                               dtype=self._dtype)

        if array is not None:
            self._binding.write(array)
            self._binding.h2d()

        self._nbytes = self._binding.nbytes

    def d2h(self) -> None:
        """Performs a copy from the device to the host optionally
        using a cuda Stream.

        :param stream: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html , defaults to None
        :type stream: cuda.Stream, optional
        """
        self._binding.d2h(stream=self._stream)

    def h2d(self) -> None:
        """Performs a copy from the host to the device optionally
        using a cuda Stream.

        :param stream: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html, defaults to None
        :type stream: cuda.Stream, optional
        """
        self._binding.h2d(stream=self._stream)

    def astype(self,
               dtype: dtype) -> None:

        self._binding.hdm.host.astype(dtype.numpy_dtype)
        self._binding.hdm.device = cuda.mem_alloc(
            self._binding.hdm.host.nbytes)



    @property
    def host(self) -> numpy.ndarray:
        """Returns the host memory.

        :return: Host memory
        :rtype: numpy.ndarray
        """
        return self._binding.host

    @property
    def device(self) -> cuda.DeviceAllocation:
        """Returns the device memory allocation

        :return: Device memory
        :rtype: cuda.DeviceAllocation
        """
        return self._binding.device

    @property
    def shape(self) -> tuple:
        """Returns the shape of the buffer.

        :return: Shape of the buffer
        :rtype: tuple
        """
        return self._shape

    @property
    def dtype(self) -> dtype:
        """Returns the data type of the object.

        :return: Data type of the buffer
        :rtype: dtype
        """
        return self._dtype

    @property
    def value(self) -> numpy.ndarray:
        """Returns the reshaped data on the host.

        :return: Reshaped data on the host
        :rtype: numpy.ndarray
        """
        return self._binding.value
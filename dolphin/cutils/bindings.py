

import numpy as np

import pycuda.driver as cuda
import tensorrt as trt

import dolphin


class HostDeviceMem:
    """
    This class is used to allocate memory on the host and the device.

    :param host_mem: Allocated memory on the host
    :type host_mem: np.ndarray
    :param device_mem: Allocated memory on the device
    :type device_mem: cuda.DeviceAllocation
    """

    def __init__(self, host_mem: np.ndarray,
                 device_mem: cuda.DeviceAllocation):
        self.host = host_mem
        self.device = device_mem

    def __repr__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)


class CudaBinding:
    """
    This class is used to allocate memory on the host and the device.
    It is also used to copy data from the host to the device and vice versa.

    It contains these functions :
      >>> allocate(shape: tuple, dtype: dolphin.dtype)   -> None
      >>> write(data: object)                       -> None
      >>> h2d(stream: cuda.Stream = None)           -> None
      >>> d2h(stream: cuda.Stream = None)           -> None

    Also, CudaBinding contains these properties :
        >>> CudaBinding.size: int                       # Returns the number of elements in the buffer  # noqa
        >>> CudaBinding.nbytes: int                     # Returns the number of bytes in the buffer  # noqa
        >>> CudaBinding.host: np.ndarray                # Returns the host memory
        >>> CudaBinding.device: cuda.DeviceAllocation   # Returns the device memory
        >>> CudaBinding.shape: tuple                    # Returns the shape of the buffer  # noqa
        >>> CudaBinding.dtype: dolphin.dtype            # Returns the data type of the buffer  # noqa
        >>> CudaBinding.value: np.ndarray               # Returns the reshaped data on the host  # noqa

    """

    def __init__(self):
        """
        Constructor of the class CudaBinding.
        It does not take any single argument. It initializes the following
        properties :
          >>> _hdm
          >>> _shape
          >>> _dtype
          >>> _nbytes
          >>> _size

        Also, CudaBinding contains these properties :
          >>> CudaBinding.size: int which returns the number of elements
                                in the buffer
          >>> CudaBinding.nbytes: int which returns the number of bytes
                                in the buffer
          >>> CudaBinding.host: np.ndarray which returns the host memory
          >>> CudaBinding.device: cuda.DeviceAllocation which returns
                                  the device memory
          >>> CudaBinding.shape: tuple which returns the shape of the buffer
          >>> CudaBinding.dtype: dolphin.dtype which returns the data type
                                 of the buffer
          >>> CudaBinding.value: np.ndarray which returns the reshaped data
                                 on the host
        """

        self._hdm = None
        self._shape = None
        self._dtype = None
        self._nbytes = 0
        self._size = 0

    def allocate(self,
                 shape: tuple,
                 dtype: dolphin.dtype) -> None:
        """
        This function allocates the memory on the host and the device.

        :param shape: Shape of the memory to be allocated
        :type shape: tuple
        :param dtype: Data type of the memory to be allocated
        :type dtype: np.dtype
        """

        self._shape = shape
        self._dtype = dtype

        _hm = np.zeros(trt.volume(self._shape), self._dtype.numpy_dtype)
        _dm = cuda.mem_alloc(_hm.nbytes)

        self._nbytes = _hm.nbytes
        self._hdm = HostDeviceMem(host_mem=_hm,
                                  device_mem=_dm)

        self._size = trt.volume(self._shape)

    def write(self, data: object) -> None:
        """
        This function copies the data from the host to the device.

        :param data: Data to be copied
        :type data: object
        """

        self._hdm.host = np.array(data,
                                  dtype=self._dtype.numpy_dtype,
                                  order="C")

    def h2d(self, stream: cuda.Stream = None) -> None:
        """
        This function copies the data from the host to the device.
        If a stream is provided, the copy will be done asynchronously.

        :param stream: Cuda Stream in order to perform asynchronous copy,
        defaults to None
        :type stream: cuda.Stream, optional
        """

        if stream is None:
            cuda.memcpy_htod(self._hdm.device, self._hdm.host)
        else:
            cuda.memcpy_htod_async(self._hdm.device,
                                   self._hdm.host,
                                   stream=stream)

    def d2h(self, stream: cuda.Stream = None) -> None:
        """
        This function copies the data from the device to the host.
        If a stream is provided, the copy will be done asynchronously.

        :param stream: Cuda Stream in order to perform asynchronous copy,
        defaults to None
        :type stream: cuda.Stream, optional
        """
        if stream is None:
            cuda.memcpy_dtoh(self._hdm.host, self._hdm.device)
        else:
            cuda.memcpy_dtoh_async(self._hdm.host,
                                   self._hdm.device,
                                   stream=stream)

    def __del__(self):
        """
        This function is called when the object is destroyed.
        It is used to free the device memory.
        """

        try:
            self._hdm.device.free()
            del self._hdm.device
            del self._hdm.host
        except AttributeError:
            pass
        except Exception as exception:
            print(f"Encountered Exception while destroying object \
                  {self.__class__} : {exception}")

    @property
    def hdm(self) -> HostDeviceMem:
        """
        This is a property that returns the host and device memory of
        the buffer as a HostDeviceMem object.

        :return: HostDeviceMem object of the CudaBinding object
        :rtype: HostDeviceMem
        """
        return self._hdm

    @property
    def size(self) -> int:
        """
        Returns the number of elements in the buffer

        :return: number of elements in the buffer
        :rtype: int
        """
        return self._size

    @property
    def nbytes(self) -> int:
        """
        Returns the number of bytes in the buffer

        :return: number of bytes in the buffer
        :rtype: int
        """
        return self._size

    @property
    def host(self) -> np.ndarray:
        """
        This is a property that returns the host memory of
        the buffer as a np.ndarray object.
        The host memory is the value of the buffer on the host memory.

        :return: Pointer to host memory
        :rtype: np.ndarray
        """
        return self._hdm.host

    @property
    def device(self) -> cuda.DeviceAllocation:
        """
        This is a property that returns the device memory of the buffer
        as a cuda.DeviceAllocation object.
        The device memory is the value of the buffer on the device memory.

        Returns:
            cuda.DeviceAllocation: Pointer to device memory
        """
        return self._hdm.device

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the buffer

        Returns:
            tuple: The shape of the buffer
        """
        return self._shape

    @property
    def dtype(self) -> dolphin.dtype:
        """
        Returns the dtype of the buffer

        Returns:
            dtype: The dolphin.dtype of the buffer
        """
        return self._dtype

    @property
    def value(self) -> np.ndarray:
        """
        This is a property that returns the value of the buffer
        as a numpy array.
        The value is the value of the buffer on the host memory.

        Returns:
            np.ndarray: The value of the buffer on the host memory
        """
        return self._hdm.host.reshape(self._shape, order="C").astype(
            self._dtype.numpy_dtype)

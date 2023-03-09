
from functools import reduce

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import numpy as np


class HostDeviceMem(object):
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
    """

    def __init__(self):
        """
        Constructor of the class
        """

        self._hdm = None
        self._shape = None
        self._dtype = None
        self._nbytes = 0
        self._size = 0

    def allocate(self, shape: tuple, dtype: np.dtype) -> None:
        """
        This function allocates the memory on the host and the device.

        :param shape: Shape of the memory to be allocated
        :type shape: tuple
        :param dtype: Data type of the memory to be allocated
        :type dtype: np.dtype
        """

        self._shape = shape
        self._dtype = dtype

        _hm = cuda.pagelocked_zeros(trt.volume(self._shape), self._dtype)
        _dm = cuda.mem_alloc(_hm.nbytes)

        self._nbytes = _hm.nbytes
        self._hdm = HostDeviceMem(host_mem=_hm,
                                  device_mem=_dm)

        if self._shape == ():
            self._size = 1
        else:
            self._size = reduce((lambda x, y: x * y), self._shape)

    def write(self, data: object) -> None:
        """
        This function copies the data from the host to the device.

        :param data: Data to be copied
        :type data: object
        """

        self._hdm.host = np.array(data, dtype=self._dtype, order="C")

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
        except Exception as exception:
            # pylint: disable=broad-exception-caught
            print(f"Encountered Exception while destroying object \
                  {self.__class__} : {exception}")

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
    def dtype(self) -> np.dtype:
        """
        Returns the dtype of the buffer

        Returns:
            np.dtype: The dtype of the buffer
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
            self._dtype)


class CUDA_Buffers(object):

    def __init__(self):

        self._inputs = {}
        self._outputs = {}

        self._input_order = []
        self._output_order = []

    def allocate_input(self, name: str,
                       shape: tuple,
                       dtype: object):
        """_summary_

        :param name: _description_
        :type name: str
        :param shape: _description_
        :type shape: tuple
        :param dtype: _description_
        :type dtype: object
        """

        self._inputs[name] = CudaBinding()
        self._inputs[name].allocate(shape, dtype)

        self._input_order.append(name)

    def allocate_output(self, name: str,
                        shape: tuple,
                        dtype: object):
        """_summary_

        :param name: _description_
        :type name: str
        :param shape: _description_
        :type shape: tuple
        :param dtype: _description_
        :type dtype: object
        """

        self._outputs[name] = CudaBinding()
        self._outputs[name].allocate(shape, dtype)

        self._output_order.append(name)

    def input_h2d_async(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._inputs.values():
            inp.h2d(stream)

    def input_d2h_async(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._inputs.values():
            inp.d2h(stream)

    def output_h2d_async(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._outputs.values():
            inp.h2d(stream)

    def output_d2h_async(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._outputs.values():
            inp.d2h(stream)

    def write_input_host(self,
                         name: str,
                         data: object):
        """_summary_

        :param name: _description_
        :type name: str
        :param data: _description_
        :type data: object
        :return: _description_
        :rtype: _type_
        """

        self._inputs[name].write(data)
        return self._inputs[name].shape

    @property
    def output(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """

        return [out.value for out in self._outputs]

    @property
    def input_bindings(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """

        return [self._inputs[name].device for name in self._input_order]

    @property
    def output_bindings(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """

        return [self._outputs[name].device for name in self._output_order]

    @property
    def bindings(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return self.input_bindings + self.output_bindings

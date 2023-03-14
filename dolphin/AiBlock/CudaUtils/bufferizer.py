"""_summary_
"""

import pycuda.autoinit  # pylint: disable=import-error
import pycuda.driver as cuda  # pylint: disable=import-error
import tensorrt as trt  # pylint: disable=import-error

import numpy as np

from .bindings import CudaBinding, HostDeviceMem  # pylint: disable=import-error


class Bufferizer(CudaBinding):
    """_summary_
    """

    def __init__(self, shape: tuple,
                 buffer_size: int,
                 dtype: np.dtype,
                 flush_hook: callable = None,
                 allocate_hook: callable = None,
                 append_one_hook: callable = None,
                 append_multiple_hook: callable = None,
                 buffer_full_hook: callable = None
                 ):

        super().__init__()

        self._shape = shape
        self._dtype = dtype
        self._size = buffer_size
        self._itemsize = trt.volume(self._shape)
        self._hdm = None
        self._nbytes = 0
        self._n_elements = 0

        self._flush_hook = None
        self._allocate_hook = None
        self._append_one_hook = None
        self._append_multiple_hook = None
        self._buffer_full_hook = None

        self.allocate()

        self._flush_hook = flush_hook
        self._allocate_hook = allocate_hook
        self._append_one_hook = append_one_hook
        self._append_multiple_hook = append_multiple_hook
        self._buffer_full_hook = buffer_full_hook

    def allocate(self) -> None:
        """_summary_
        """

        host = np.zeros(
            trt.volume((self._size,) + self._shape), self._dtype)
        device = cuda.mem_alloc(host.nbytes)  # pylint: disable=no-member
        self._nbytes = host.nbytes
        self._hdm = HostDeviceMem(host, device)
        self._n_elements = 0

        if self._allocate_hook is not None:
            self._allocate_hook(self)

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
            cuda.memcpy_dtod(int(self._hdm.device) + self._n_elements
                             * self._itemsize,
                             binding.device,
                             self._itemsize * np.dtype(self._dtype).itemsize)
            # pylint: disable=no-member
        else:
            cuda.memcpy_dtod_async(int(self._hdm.device) + self._n_elements
                                   * self._itemsize,
                                   binding.device,
                                   self._itemsize *
                                   np.dtype(self._dtype).itemsize,
                                   stream)  # pylint: disable=no-member

        self._n_elements += 1

        if self._append_one_hook is not None:
            self._append_one_hook(self)

        if self.full:
            if self._buffer_full_hook is not None:
                self._buffer_full_hook(self)

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
            cuda.memcpy_dtod(self._hdm.device + self._n_elements
                             * self._itemsize,
                             binding.device,
                             batch_size * self._itemsize *
                             np.dtype(self._dtype).itemsize)
            # pylint: disable=no-member
        else:
            cuda.memcpy_dtod_async(self._hdm.device + self._n_elements
                                   * self._itemsize,
                                   binding.device,
                                   batch_size * self._itemsize *
                                   np.dtype(self._dtype).itemsize,
                                   stream)  # pylint: disable=no-member

        self._n_elements += batch_size

        if self._append_multiple_hook is not None:
            self._append_multiple_hook(self)

        if self.full:
            if self._buffer_full_hook is not None:
                self._buffer_full_hook(self)

    def flush(self, value: int = 0,
              stream: cuda.Stream = None) -> None:
        # pylint: disable=no-member
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        size = self._nbytes

        if stream is None:
            cuda.memset_d8(self._hdm.device,
                           value,
                           size)
            # pylint: disable=no-member
        else:
            cuda.memset_d8_async(self._hdm.device,
                                 value,
                                 size,
                                 stream)
            # pylint: disable=no-member

        self._n_elements = 0

        if self._flush_hook is not None:
            self._flush_hook(self)

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

    def buffer_full_hook(self, hook: callable):
        """_summary_

        :param hook: _description_
        :type hook: callable
        """
        self._buffer_full_hook = hook

    @property
    def value(self) -> np.ndarray:
        """_summary_

        :return: _description_
        :rtype: np.ndarray
        """
        return self._hdm.host.reshape((self._size,) + self._shape)

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

    @property
    def full(self) -> bool:
        """_summary_

        :return: _description_
        :rtype: bool
        """
        return self._n_elements == self._size

    @property
    def shape(self) -> tuple:
        """_summary_

        :return: _description_
        :rtype: tuple
        """
        return super().shape

    def __len__(self):
        return self._n_elements


class CudaTrtBuffers(object):
    """_summary_

    :param object: _description_
    :type object: _type_
    """

    def __init__(self, ):

        self._inputs = {}
        self._outputs = {}

        self._input_order = []
        self._output_order = []

    def allocate_input(self, name: str,
                       shape: tuple,
                       buffer_size: int,
                       dtype: object,
                       buffer_full_hook: callable = None,
                       flush_hook: callable = None,
                       allocate_hook: callable = None,
                       append_one_hook: callable = None,
                       append_multiple_hook: callable = None):
        """_summary_

        :param name: _description_
        :type name: str
        :param shape: _description_
        :type shape: tuple
        :param dtype: _description_
        :type dtype: object
        """

        self._inputs[name] = Bufferizer(shape=shape,
                                        buffer_size=buffer_size,
                                        dtype=dtype,
                                        buffer_full_hook=buffer_full_hook,
                                        flush_hook=flush_hook,
                                        allocate_hook=allocate_hook,
                                        append_one_hook=append_one_hook,
                                        append_multiple_hook=append_multiple_hook)
        self._inputs[name].allocate()

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

    def input_h2d(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._inputs.values():
            inp.h2d(stream)

    def input_d2h(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._inputs.values():
            inp.d2h(stream)

    def output_h2d(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._outputs.values():
            inp.h2d(stream)

    def output_d2h(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._outputs.values():
            inp.d2h(stream)

    def flush(self, value: int = 0,
              stream: cuda.Stream = None) -> None:
        """_summary_

        :param value: _description_, defaults to 0
        :type value: int, optional
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        for inp in self._inputs.values():
            inp.flush(value, stream)

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

    def append_one_input(self, name: str,
                         data: CudaBinding,
                         stream: cuda.Stream = None):
        """_summary_

        :param name: _description_
        :type name: str
        :param data: _description_
        :type data: CudaBinding
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        self._inputs[name].append_one(data, stream)

    def append_multiple_input(self, name: str,
                              data: CudaBinding,
                              stream: cuda.Stream = None):
        """_summary_

        :param name: _description_
        :type name: str
        :param data: _description_
        :type data: CudaBinding
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        self._inputs[name].append_multiple_input(data, stream)

    def __del__(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        try:
            for element in self._inputs.values():
                del element
            for element in self._outputs.values():
                del element
        except Exception as exception:
            # pylint: disable=broad-exception-caught
            print(f"Encountered Exception while destroying object \
                  {self.__class__} : {exception}")

    @property
    def shape(self) -> dict:
        """_summary_

        :return: _description_
        :rtype: dict
        """
        return {key: inp.shape for key, inp in self._inputs.items()}

    @property
    def full(self) -> bool:
        """_summary_

        :return: _description_
        :rtype: bool
        """

        for binding_name in self._inputs.keys():
            if self._inputs[binding_name].full:
                return True

        return False

    @property
    def output(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """

        return {key: out.value for key, out in self._outputs.items()}

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


if __name__ == "__main__":

    def append_one_handler(binding: CudaBinding):
        # pylint: disable=C0116
        print(f"append_one_handler : {len(binding)}")

    def buffer_full_handler(binding: CudaBinding):
        # pylint: disable=C0116
        print(f"buffer_full_handler : {len(binding)}")

    buffer = Bufferizer((2,),
                        buffer_size=5,
                        dtype=np.uint8,
                        buffer_full_hook=buffer_full_handler,
                        append_one_hook=append_one_handler)

    data = CudaBinding()
    data.allocate((2,), np.uint8)
    data.write(np.array([1, 2], dtype=np.uint8))
    data.h2d()

    buffer.append_one(data)
    buffer.append_one(data)
    buffer.append_one(data)
    buffer.append_one(data)
    buffer.append_one(data)

    buffer.d2h()

    print(buffer.value)
    print(len(buffer))

    buffer.flush(value=8)
    buffer.d2h()

    print(buffer.value)
    print(len(buffer))

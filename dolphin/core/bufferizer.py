"""_summary_
"""

from typing import Any, Union, List

import pycuda.driver as cuda  # pylint: disable=import-error
import tensorrt as trt  # pylint: disable=import-error

import dolphin.core.dtype


class Bufferizer:
    """
    Bufferizer is a class that allows to easily bufferize data on the GPU.
    The purpose is to handle seamlessly batched data and to avoid unnecessary
    memory allocation but rather reuse the same memory buffer and favour
    copy operations.

    Bufferizer can, through its `append` methods, append data to the buffer.
    It can be either one `darray` at a time, a list of `darray`s or a single
    batched `darray`.

    In addition to bufferizing data, the class also allows to trigger hooks
    at different moments of its lifecycle.

    - flush_hook : callable triggered when buffer is flushed
    - allocate_hook : callable triggered when buffer is allocated
    - append_one_hook : callable triggered when
      buffer has a new element appended
    - append_multiple_hook : callable triggered when buffer
      has new elements appended
    - buffer_full_hook : callable triggered when the buffer
      is full after calling any append

    :param shape: shape of element to bufferize
    :type shape: tuple
    :param buffer_size: size of the buffer
    :type buffer_size: int
    :param dtype: dtype of the element to bufferize
    :type dtype: dolphin.dtype
    :param stream: stream to use for the buffer, defaults to None
    :type stream: cuda.Stream, optional
    :param flush_hook: callable triggered when buffer is flushed,
                       defaults to None
    :type flush_hook: callable, optional
    :param allocate_hook: callable triggered when buffer is allocated,
      not triggered by the first allocation, defaults to None
    :type allocate_hook: callable, optional
    :param append_one_hook: callable triggered when buffer has
                            a new element appended, defaults to None
    :type append_one_hook: callable, optional
    :param append_multiple_hook: callable triggered when buffer has
                                 new elements appended, defaults to None
    :type append_multiple_hook: callable, optional
    :param buffer_full_hook: callable triggered when the buffer is
                             full after calling any append, defaults to None
    :type buffer_full_hook: callable, optional
    """

    def __init__(self,
                 shape: tuple,
                 buffer_size: int,
                 dtype: dolphin.dtype,
                 stream: cuda.Stream = None,
                 flush_hook: callable = None,
                 allocate_hook: callable = None,
                 append_one_hook: callable = None,
                 append_multiple_hook: callable = None,
                 buffer_full_hook: callable = None
                 ):
        """
        Constructor of the `Bufferizer` class.
        """

        self._shape: tuple = shape
        self._dtype: dolphin.dtype = dtype
        self._buffer_len: int = buffer_size
        self._itemsize: int = trt.volume(self._shape)
        self._nbytes: int = (self._buffer_len * self._itemsize *
                             self._dtype.itemsize)
        self._n_elements: int = 0
        self._allocation: cuda.DeviceAllocation = None
        self._stream: cuda.Stream = stream

        self._flush_hook: callable = None
        self._allocate_hook: callable = None
        self._append_one_hook: callable = None
        self._append_multiple_hook: callable = None
        self._buffer_full_hook: callable = None

        self.allocate()

        self._flush_hook = flush_hook
        self._allocate_hook = allocate_hook
        self._append_one_hook = append_one_hook
        self._append_multiple_hook = append_multiple_hook
        self._buffer_full_hook = buffer_full_hook

    def allocate(self) -> None:
        """
        Method to allocate the buffer on the GPU.
        This method is called automatically when the class is instanciated.

        Once the buffer is allocated, it is not possible to change the size
        and it the allocation is initialized to 0.

        Also, this methods triggers the allocate_hook if it is not None.
        """

        self._allocation = cuda.mem_alloc(self._buffer_len * self._itemsize
                                          * self._dtype.itemsize)
        self._n_elements = 0

        if self._allocate_hook is not None:
            self._allocate_hook(self)

        self.flush(0)

    def append(self, element: Union[dolphin.darray,
                                    List[dolphin.darray]]) -> None:
        """
        General purpose append method.
        You can provide either a single `darray`, a batched `darray` or
        a list of `darray` and the method will handle it.

        For more details about the handling of each case, see the
        `append_one` and `append_multiple` methods.

        :param element: The element to append to the buffer.
        :type element: Union[dolphin.darray, List[dolphin.darray]]
        """

        if isinstance(element, dolphin.darray):
            if element.shape == self._shape:
                self.append_one(element)
            elif element.shape[1:] == self._shape:
                self.append_multiple(element)
            else:
                raise ValueError(f"Element shape {element.shape} does not \
                                  match bufferizer shape {self._shape}.")
        elif isinstance(element, list):
            self.append_multiple(element)

        else:
            raise TypeError(f"Element type {type(element)} is not supported.")

    def append_one(self, element: dolphin.darray) -> None:
        """
        Method to append one element to the buffer.
        The element is copied and appended to the buffer.
        `element` must be a `darray` of the same shape and dtype as the
        `bufferizer`.
        The size of the buffer is increased by one.

        Appending one element triggers the `append_one_hook` if it is not None.
        Once the buffer is full, the `buffer_full_hook` is triggered.

        :param element: The element to append to the buffer.
        :type element: dolphin.darray
        """

        if self._n_elements == self._buffer_len:
            raise BufferError(
                f"Bufferizer is full ({self._buffer_len} elements).")

        if element.shape != self._shape:
            raise ValueError(f"Element shape {element.shape} does not match \
                              bufferizer shape {self._shape}.")

        if element.dtype != self._dtype:
            raise ValueError(f"Element dtype {element.dtype} does not match \
                              bufferizer dtype {self._dtype}.")

        tmp_darray = dolphin.darray(shape=(self._itemsize, ),
                                    dtype=self._dtype,
                                    allocation=(int(self._allocation) +
                                                self._n_elements
                                    * self._itemsize * self._dtype.itemsize))

        element.flatten(dst=tmp_darray)

        self._n_elements += 1

        if self._append_one_hook is not None:
            self._append_one_hook(self)

        if self.full:
            if self._buffer_full_hook is not None:
                self._buffer_full_hook(self)

    def append_multiple(self, element: Union[dolphin.darray,
                                             List[dolphin.darray]]) -> None:
        # pylint: disable=no-member
        """
        Function used in order to append multiple `darray` to the buffer
        at once. The `darray` must be of the same shape and dtype as the
        `bufferizer` with the exception of the first dimension which can be
        different in case of a batched `darray`. Otherwise, the `darray` must
        be a list of `darray` of the same shape and dtype as the `bufferizer`.

        The size of the buffer is increased by the number of elements in
        `element`.

        It is assumed that the number of elements in `element` is defined
        as the first dimension of its shape.
        For instance::

            element.shape = (batch_size, *self._shape)

        Appending multiple elements triggers the `append_multiple_hook` if it
        is not None. Once the buffer is full, the `buffer_full_hook` is
        triggered if it is not None.

        :param element: The element to append to the buffer.
        :type element: darray
        """
        if isinstance(element, list):
            for elem in element:
                self.append_one(elem)

        else:
            batch_size = element.shape[0]

            if element.shape[1:] != self._shape:
                raise ValueError(f"Element shape {element.shape} does \
                                 not match bufferizer shape {self._shape}.")

            if self._n_elements == self._buffer_len:
                raise BufferError(f"Bufferizer is full (max \
                                    {self._buffer_len} elements).")

            if self._n_elements + batch_size > self._buffer_len:
                raise BufferError(f"Bufferizer is going to be overflowed \
                    ({self._buffer_len} elements). Tried to push \
                    {batch_size} elements.")

            tmp_darray = dolphin.darray(shape=(self._itemsize * batch_size,),
                                        dtype=self._dtype,
                                        allocation=(int(self._allocation) +
                                        self._n_elements * self._itemsize *
                                        self._dtype.itemsize))

            element.flatten(dst=tmp_darray)

            self._n_elements += batch_size

            if self._append_multiple_hook is not None:
                self._append_multiple_hook(self)

            if self.full:
                if self._buffer_full_hook is not None:
                    self._buffer_full_hook(self)

    def flush(self, value: Any = 0) -> None:
        # pylint: disable=no-member
        """
        Set the buffer to a given value.
        Useful in order to get rid of any residual data in the buffer.

        Calling this method triggers the `flush_hook` if it is not None.

        :param stream: Cuda stream, defaults to None
        :type stream: cuda.Stream, optional
        """

        self.darray.fill(value)

        # pylint: disable=no-member

        self._n_elements = 0

        if self._flush_hook is not None:
            self._flush_hook(self)

    def flush_hook(self, hook: callable):
        """
        Method to set the flush hook.
        This hook is called when the `flush` method is called.

        :param hook: Callable function called each time the `flush` method is
                        called.
        :type hook: callable
        """
        self._flush_hook = hook

    def allocate_hook(self, hook: callable):
        """
        Method to set the allocate hook.
        This hook is called when the `allocate` method is called.

        :param hook: Callable function called each time the `allocate` method
                        is called.
        :type hook: callable
        """
        self._allocate_hook = hook

    def append_one_hook(self, hook: callable):
        """
        Method to set the append one hook.

        :param hook: Callable function called each time the `append_one`
                        method is called.
        :type hook: callable
        """
        self._append_one_hook = hook

    def append_multiple_hook(self, hook: callable):
        """
        Method to set the append multiple hook.

        :param hook: Callable function called each time the
                        `append_multiple` method is called.
        :type hook: callable
        """
        self._append_multiple_hook = hook

    def buffer_full_hook(self, hook: callable):
        """
        Method to set the buffer full hook.

        :param hook: Callable function called each time the buffer is full.
        :type hook: callable
        """
        self._buffer_full_hook = hook

    @property
    def allocation(self) -> cuda.DeviceAllocation:
        """
        Property in order to get the allocation of the buffer.

        :return: Allocation of the buffer.
        :rtype: cuda.DeviceAllocation
        """
        return self._allocation

    @property
    def darray(self) -> dolphin.darray:
        """
        Property in order to convert a bufferizer to a `darray`.

        Important note : The `darray` returned by this property is not a copy
        of the bufferizer. It is a view of the bufferizer. Any change to the
        `darray` will be reflected in the bufferizer and vice-versa.

        :return: darray of bufferizer
        :rtype: dolphin.darray
        """

        res = dolphin.darray(shape=(self._buffer_len,) + self._shape,
                             dtype=self._dtype,
                             allocation=self._allocation,
                             stream=self._stream)
        return res

    @property
    def element_nbytes(self) -> int:
        """
        Property in order to get the number of bytes of a single element in
        the buffer.

        :return: Number of bytes of a single element in the buffer.
        :rtype: int
        """
        return self._itemsize

    @property
    def nbytes(self) -> int:
        """
        Property in order to get the number of bytes of the buffer.

        :return: Number of bytes of the buffer.
        :rtype: int
        """
        return self._nbytes

    @property
    def full(self) -> bool:
        """
        Property in order to know if the buffer is full.

        :return: True if the buffer is full, False otherwise.
        :rtype: bool
        """
        return self._n_elements == self._buffer_len

    @property
    def shape(self) -> tuple:
        """
        Property in order to get the shape of the the buffer.

        :return: Shape of the buffer.
        :rtype: tuple
        """
        return (self._buffer_len,) + self._shape

    @property
    def element_shape(self) -> tuple:
        """
        Property in order to get the shape of a single element in the buffer.

        :return: Shape of a single element in the buffer.
        :rtype: tuple
        """
        return self._shape

    @property
    def dtype(self) -> dolphin.dtype:
        """
        Property in order to get the dtype of the buffer.

        :return: Dtype of the buffer.
        :rtype: dolphin.dtype
        """
        return self._dtype

    def __len__(self) -> int:
        """
        Method in order to get the number of elements in the buffer.

        :return: Number of elements in the buffer.
        :rtype: int
        """

        return self._n_elements

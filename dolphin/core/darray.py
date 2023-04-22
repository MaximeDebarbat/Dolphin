"""
# darray

This module implements the `darray` class.
`darray` is a generic numpy style array that can be used
with the `dolphin` library. It implements common features
such as `astype`, `transpose`, `copy`...

Along `darray` class, this module also implements wrappers
to use `darray` operations directly from the `dolphin` module
in a numpy style.

## darray class

Main class for array operations. You can find more information about
this class in the `darray` class documentation directly. To use it,
you can simply import it with::

    import numpy as np
    import dolphin as dp

    a = np.random.rand(10, 10)
    d = dp.darray(a)

    d+=10
    e = d.astype(dp.dtype.float32)
    ...

## Wrappers

The purpose of the wrappers is to provide a simple and fast way to perform
Numpy style operations on `darray` objects.

These operations are as follow:
- transpose
- astype
- copy
- add
- subtract
- multiply
- divide
- reversed_divide
- reversed_subtract
- zeros
- zeros_like
- ones
- ones_like
- empty
- empty_like

It is important to note that if you want to perform operations in
an efficient way, you should allocate the destination before calling the
functions.
Creation functions and methods do not allow to specify the destination. Thus,
they are not efficient and thus have to be called with caution and if possible,
only once.
"""


import math
import time
from typing import Union

import pycuda.driver as cuda  # pylint: disable=import-error
import tensorrt as trt  # pylint: disable=import-error

import numpy
import dolphin


class darray(dolphin.CudaBaseNew):
    """
    ## darray

    This class implements a generic numpy style array that can be used
    with the `dolphin` library. It implements common features available
    with numpy arrays such as `astype`, `transpose`, `copy`...

    ### Constructor

    `darray` constructor can be created with the following parameters::
        array: numpy.ndarray = None
        shape: tuple = None
        dtype: dolphin.dtype = None
        stream: cuda.Stream = None
        allocation: cuda.DeviceAllocation = None

    ### Overview

    `darray` is made with the same philosophy as `numpy.ndarray`. The usability
    is really close to numpy arrays. However, `darray` is meant to be much more
    performant than `numpy.ndarray` since it is GPU accelerated.

    ### Proporties

    `darray` has the following proporties::

        strides: tuple
            Strides of the array
        shape: tuple
            Shape of the array
        dtype: dolphin.dtype
            Dtype of the array
        size: int
            Number of elements in the array
        nbytes: int
            Number of bytes in the array
        stream: cuda.Stream
            Stream used for the operations
        allocation: cuda.DeviceAllocation
            Allocation of the array on the device

    ### Methods

    The methods `darray` currently supports are::

        from_ndarray(array: numpy.ndarray) -> None
            Creates a darray from a numpy array.
        astype(dtype: dolphin.dtype, dst: 'darray' = None) -> 'darray'
            Casts the array to a new dtype.
        transpose(*axes: int, dst: 'darray' = None) -> 'darray'
            Transposes the array.
        copy() -> 'darray'
            Copies the array.
        add(other: 'darray', dst: 'darray' = None) -> 'darray'
            Adds the array to another array or a scalar.
        subtract(other: 'darray', dst: 'darray' = None) -> 'darray'
            Subtracts the array to another array or a scalar.
        multiply(other: 'darray', dst: 'darray' = None) -> 'darray'
            Multiplies the array to another array or a scalar.
        divide(other: 'darray', dst: 'darray' = None) -> 'darray'
            Divides the array to another array or a scalar.
        reversed_divide(other: 'darray', dst: 'darray' = None) -> 'darray'
            Divides another array or a scalar to the array.
        reversed_subtract(other: 'darray', dst: 'darray' = None) -> 'darray'
            Subtracts another array or a scalar to the array.
    """

    def __init__(self,
                 array: numpy.ndarray = None,
                 shape: tuple = None,
                 dtype: dolphin.dtype = None,
                 stream: cuda.Stream = None,
                 allocation: cuda.DeviceAllocation = None
                 ) -> None:

        super().__init__()

        if array is not None:
            dtype = dolphin.dtype.from_numpy_dtype(array.dtype)
            shape = array.shape

        self._stream: cuda.Stream = stream
        self._dtype: dolphin.dtype = dtype
        self._shape: tuple = shape
        self._size: int = trt.volume(self._shape)
        self._nbytes: int = int(self._size * self._dtype.itemsize)

        if allocation is not None:
            self._allocation: cuda.DeviceAllocation = allocation
        else:
            self._allocation: cuda.DeviceAllocation = cuda.mem_alloc(self._nbytes)

        if array is not None:
            cuda.memcpy_htod_async(self._allocation,
                                   array,
                                   self._stream)

        # self._block = (int(min(self.MAX_THREADS_PER_BLOCKS, self._size)), 1, 1) # LEADS TO 0.17ms of timing
        # self._grid = (int(math.ceil(self._size / self._block[0])), 1) # LEADS TO 0.17ms of timing

        self._block, self._grid = self.GET_BLOCK_GRID_1D(self._size)

        self._cu_axpbz = dolphin.cudarray.CU_AXPBZ
        self._cu_axpbyz = dolphin.cudarray.CU_AXPBYZ
        self._cu_eltwise_mult = dolphin.cudarray.CU_ELTWISE_MULT
        self._cu_eltwise_div = dolphin.cudarray.CU_ELTWISE_DIV
        self._cu_scal_div = dolphin.cudarray.CU_SCAL_DIV
        self._cu_invscal_div = dolphin.cudarray.CU_INVSCAL_DIV
        self._cu_eltwise_cast = dolphin.cudarray.CU_ELTWISE_CAST
        self._cu_eltwise_abs = dolphin.cudarray.CU_ELTWISE_ABS
        self._cu_transpose = dolphin.cudarray.CU_TRANSPOSE
        self._cu_fill = dolphin.cudarray.CU_FILL

    @staticmethod
    def compute_strides(shape: tuple) -> tuple:
        """Computes the strides of an array from the shape.
        The strides are the number of elements to skip to get to the next
        element. Also, the strides are in elements, not bytes.

        :param shape: shape of the ndarray
        :type shape: tuple
        :return: Strides
        :rtype: tuple
        """
        if shape:
            strides = [1]
            for s in shape[:0:-1]:
                strides.append(strides[-1] * max(1, s))
            return tuple(strides[::-1])

        return ()

    def from_ndarray(self, array: numpy.ndarray) -> None:
        """Writes allocation from a numpy array.
        If the array is not the same shape or dtype as the darray,
        an error is raised.

        :param array: Numpy array create the darray from
        :type array: numpy.ndarray
        """

        if array.shape != self._shape:
            raise ValueError(
                f"array shape doesn't match darray : {array.shape} \
                    != {self.shape}")

        if dolphin.dtype.from_numpy_dtype(array.dtype) != self._dtype:
            raise ValueError(
                f"array does not match the dtype : {array.dtype} \
                    != {self.dtype}")

        cuda.memcpy_htod_async(self._allocation,
                               array.flatten(order="C"),
                               self._stream)

    def astype(self, dtype: dolphin.dtype,
               dst: 'darray' = None) -> 'darray':
        """Converts the darray to a different dtype.
        Note that a copy from device to device is performed.

        :param dtype: Dtype to convert the darray to
        :type dtype: dolphin.dtype
        """

        if dst is not None and self._shape != dst.shape:
            raise ValueError(
                f"dst shape doesn't match darray : {self.shape} != {dst.shape}")

        if dst is not None and dtype != dst.dtype:
            raise ValueError(
                f"dst does not match the dtype : {self.dtype} != {dst.dtype}")

        if dtype == self._dtype:
            if dst is None:
                return self.copy()

            cuda.memcpy_dtod_async(dst.allocation,
                                   self._allocation,
                                   self._nbytes,
                                   self._stream)
            return dst

        if dst is None:
            dst = self.__class__(shape=self._shape,
                                 dtype=dtype,
                                 stream=self._stream)

        self._cu_eltwise_cast(self,
                              dst,
                              self._size,
                              block=self._block,
                              grid=self._grid,
                              stream=self._stream)

        return dst

    def transpose(self, *axes: int, dst: 'darray' = None) -> 'darray':
        """Transposes the darray according to the axes.

        :param axes: Axes to permute
        :type axes: Tuple[int]
        :return: Transposed darray
        :rtype: darray
        """

        if len(axes) != len(self._shape):
            raise ValueError("axes don't match array")

        if not all(isinstance(v, int) for v in axes):
            raise ValueError("axes must be integers")

        strides = self.strides
        new_shape = [self.shape[i] for i in axes]
        new_strides = [strides[i] for i in axes]

        if dst is not None:
            if dst.shape != tuple(new_shape):
                raise ValueError("dst shape doesn't match")
            if dst.dtype != self.dtype:
                raise ValueError("dst dtype doesn't match")

        new_shape = numpy.array(new_shape,
                                dtype=numpy.uint32)
        new_strides = numpy.array(new_strides,
                                  dtype=numpy.uint32)

        new_shape_allocation = cuda.mem_alloc(new_shape.nbytes)
        new_strides_allocation = cuda.mem_alloc(new_strides.nbytes)

        cuda.memcpy_htod_async(new_shape_allocation,
                               new_shape,
                               self._stream)
        cuda.memcpy_htod_async(new_strides_allocation,
                               new_strides,
                               self._stream)
        if dst is not None:
            res = dst
        else:
            res = self.__class__(shape=tuple(new_shape),
                                 dtype=self._dtype,
                                 stream=self._stream)

        self._cu_transpose(
            self,
            res,
            new_shape_allocation,
            new_strides_allocation,
            len(new_shape),
            self._size,
            block=self._block,
            grid=self._grid,
            stream=self._stream)

        return res

    def copy(self) -> 'darray':
        """Returns a copy of the current darray.
        Note that a copy from device to device is performed.

        :return: Copy of the array with another cuda allocation
        :rtype: darray
        """
        res = self.__class__(shape=self._shape,
                             dtype=self._dtype,
                             stream=self._stream)

        cuda.memcpy_dtod_async(res.allocation,
                               self._allocation,
                               self._nbytes,
                               stream=self._stream)

        return res

    @property
    def strides(self) -> tuple:
        """Property to access the strides of the array.

        :return: _description_
        :rtype: _type_
        """
        return self.compute_strides(self._shape)

    @property
    def allocation(self) -> cuda.DeviceAllocation:
        """Property to access the cuda allocation of the array

        :return: The cuda allocation of the array
        :rtype: cuda.DeviceAllocation
        """
        return self._allocation

    @property
    def size(self) -> int:
        """Property to access the size of the array.
        Size is defined as the number of elements in the array.

        :return: The size of the array, in terms of number of elements
        :rtype: int
        """
        return self._size

    @property
    def dtype(self) -> dolphin.dtype:
        """Property to access the dolphin.dtype of the array

        :return: dolphin.dtype of the array
        :rtype: dolphin.dtype
        """
        return self._dtype

    @property
    def shape(self) -> tuple:
        """Property to access the shape of the array

        :return: Shape of the array
        :rtype: tuple
        """
        return self._shape

    @property
    def ndarray(self) -> numpy.ndarray:
        """Property to convert the darray to a numpy.ndarray.
        Note that a copy from the device to the host is performed.

        :return: numpy.ndarray of the darray
        :rtype: numpy.ndarray
        """
        res = numpy.empty(self._shape, dtype=self._dtype.numpy_dtype)

        cuda.memcpy_dtoh_async(res,
                               self._allocation,
                               self._stream)
        return res

    @property
    def nbytes(self) -> int:
        """Property to access the number of bytes of the array

        :return: Number of bytes of the array
        :rtype: int
        """
        return self._nbytes

    @property
    def stream(self) -> cuda.Stream:
        """Property to access the cuda stream of the array

        :return: Stream used by the darray
        :rtype: cuda.Stream
        """
        return self._stream

    @property
    def T(self) -> 'darray':
        """Performs a transpose operation on the darray.
        This transpose reverse the order of the axes::

            >>> a = darray(shape=(2, 3, 4))
            >>> a.T.shape
            (4, 3, 2)

        Also, please note that this function is not efficient as
        it performs a copy of the `darray`.

        :return: Transposed darray
        :rtype: darray
        """

        return self.transpose(*self._shape[::-1])

    @stream.setter
    def stream(self, stream: cuda.Stream) -> None:
        """Sets the stream of the darray.

        :param stream: Stream to set
        :type stream: cuda.Stream
        """
        self._stream = stream

    @allocation.setter
    def allocation(self, allocation: cuda.DeviceAllocation) -> None:
        """Does not perform any copy, just sets the allocation.
        Make sure that the allocation is compatible with the darray.
        In terms of size and dtype.

        :param allocation: Cuda allocation to set
        :type allocation: cuda.DeviceAllocation
        """
        self._allocation = allocation

    @ndarray.setter
    def ndarray(self, array: numpy.ndarray) -> None:
        """Calls from_ndarray() to create a darray from a numpy array.

        :param array: _description_
        :type array: numpy.ndarray
        """
        self.from_ndarray(array)

    def __str__(self) -> str:
        """Returns the string representation of the numpy array.
        Note that a copy from the device to the host is performed.

        :return: String representation of the numpy array
        :rtype: str
        """
        return self.ndarray.__str__()  # pylint: disable=E1120

    def __repr__(self) -> str:
        """Returns the representation of the numpy array.
        Note that a copy from the device to the host is performed.

        :return: Representation of the numpy array
        :rtype: str
        """
        return self.ndarray.__repr__()  # pylint: disable=E1120

    def fill(self, value: Union[int, float, numpy.number]) -> 'darray':
        """
        Fills the darray with the value of value.

        :param value: Value to fill the array with
        :type value: object
        :return: Filled darray
        :rtype: darray
        """

        if not isinstance(value, (numpy.number, int, float)):

            raise ValueError("Only scalar values are supported")

        self._cu_fill(self,
                      value,
                      self._size,
                      self._block,
                      self._grid,
                      self._stream)

        return self

    def add(self, other: object,
            dst: 'darray' = None) -> 'darray':
        """Efficient addition of a darray with another object.
        Can be a darray or a scalar. If dst is None, normal __add__
        is called.
        This method is much more efficient than the __add__ method
        because __add__ implies a copy of the array.
        cuda.memalloc is really time consuming (up to 95% of the total
        latency is spent in cuda.memalloc only).

        :param dst: darray where to write the result
        :type dst: darray
        :param other: numpy.ndarray or scalar to add
        :type other: [scalar, numpy.ndarray]
        :raises ValueError: If the size, dtype or shape of dst is not matching
        :return: darray where the result is written. Can be dst or self
        :rtype: darray
        """
        if dst is None:
            dst = self.copy()

        if dst.nbytes != self._nbytes:
            raise ValueError(f"Size mismatch : \
{dst.nbytes} != {self._nbytes}")
        if dst.dtype != self._dtype:
            raise ValueError(f"dtype mismatch : \
{dst.dtype} != {self._dtype}")
        if dst.shape != self._shape:
            raise ValueError(f"Shape mismatch : \
{dst.shape} != {self._shape}")

        if isinstance(other, (numpy.number, int, float)):
            if other == 0:
                cuda.memcpy_dtod_async(dst.allocation,
                                       self._allocation,
                                       self._nbytes,
                                       self._stream)
            else:
                self._cu_axpbz(
                    x_array=self,
                    z_array=dst,
                    a_scalar=1,
                    b_scalar=other,
                    size=self._size,
                    block=self._block,
                    grid=self._grid,
                    stream=self._stream)

        elif isinstance(other, darray):
            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : \
{other.shape} != {self._shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : \
{other.dtype} != {self._dtype}")
            self._cu_axpbyz(
                x_array=self,
                y_array=other,
                z_array=dst,
                a_scalar=1,
                b_scalar=1,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' \
                    and '{type(other)}'")

        return dst

    def __add__(self, other: object) -> 'darray':
        """Non-Efficient addition of a darray with another object.
        It is not efficient because it implies a copy of the array
        where the result is written. cuda.memalloc is really time
        consuming (up to 95% of the total latency is spent in
        cuda.memalloc only)

        :param other: scalar or darray to add
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """

        return self.add(other)

    __radd__ = __add__  # object + array

    def __iadd__(self, other: object) -> 'darray':
        """Implements += operator. As __add__, this method is not
        efficient because it implies a copy of the array where the
        usage of cuda.memalloc which is really time consuming
        (up to 95% of the total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to add
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """

        return self.add(other, self)

    def substract(self, other: object,
                  dst: 'darray' = None) -> 'darray':
        """Efficient substraction of a darray with another object.

        :param other: scalar or darray to substract
        :type other: [scalar, darray]
        :param dst: darray where the result is written
        :type dst: darray
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """
        if dst is None:
            dst = self.copy()

        if dst.nbytes != self._nbytes:
            raise ValueError(f"Size mismatch : \
{dst.nbytes} != {self._nbytes}")
        if dst.dtype != self._dtype:
            raise ValueError(f"dtype mismatch : \
{dst.dtype} != {self._dtype}")
        if dst.shape != self._shape:
            raise ValueError(f"Shape mismatch : \
{dst.shape} != {self._shape}")

        if isinstance(other, (numpy.number, int, float)):
            if other == 0:
                cuda.memcpy_dtod_async(dst.allocation,
                                       self._allocation,
                                       self._nbytes,
                                       self._stream)
            else:
                self._cu_axpbz(
                    x_array=self,
                    z_array=dst,
                    a_scalar=1,
                    b_scalar=-other,
                    size=self._size,
                    block=self._block,
                    grid=self._grid,
                    stream=self._stream)
        elif isinstance(other, darray):
            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : \
{other.shape} != {self._shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : \
{other.dtype} != {self._dtype}")
            self._cu_axpbyz(
                x_array=self,
                y_array=other,
                z_array=dst,
                a_scalar=1,
                b_scalar=-1,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
        else:
            raise TypeError(
                f"unsupported operand type(s) for -: '{type(self)}' \
                    and '{type(other)}'")

        return dst

    sub = substract

    def __sub__(self, other: object) -> 'darray':  # array - object
        """Non-Efficient substraction of a darray with another object.
        It is not efficient because it implies a copy of the array
        where the result is written. cuda.memalloc is really time
        consuming (up to 95% of the total latency is spent in
        cuda.memalloc only)

        :param other: scalar or darray to substract
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """

        return self.substract(other)

    def reversed_substract(self, other: object,
                           dst: 'darray' = None) -> 'darray':
        """Efficient reverse substraction of an object with darray.
        It is efficient if dst is provided because it does not
        invoke cuda.memalloc.
        If dst is not provided, normal __rsub__ is called.

        :param other: scalar or darray to substract
        :type other: [scalar, darray]
        :param dst: darray where the result is written
        :type dst: darray
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """
        if dst is None:
            dst = self.copy()

        if isinstance(other, (numpy.number, int, float)):
            self._cu_axpbz(
                x_array=self,
                z_array=dst,
                a_scalar=-1,
                b_scalar=other,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
        elif isinstance(other, darray):
            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : \
{other.shape} != {self._shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : \
{other.dtype} != {self._dtype}")
            self._cu_axpbyz(
                x_array=self,
                y_array=other,
                z_array=dst,
                a_scalar=-1,
                b_scalar=1,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)

        else:
            raise TypeError(f"unsupported operand type(s) for -: \
'{type(self).__name__}' and '{type(other).__name__}'")
        return dst

    def __rsub__(self, other: object) -> 'darray':  # object - array
        """Non-Efficient substraction of another object with darray.
        It is not efficient because it implies a copy of the array
        where the result is written. cuda.memalloc is really time
        consuming (up to 95% of the total latency is spent in
        cuda.memalloc only)

        :param other: scalar or darray to substract
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """
        return self.reversed_substract(other)

    def __isub__(self, other: object) -> 'darray':  # array -= object
        """Non-Efficient -= operation.
        It is not efficient because it implies a copy of the array
        where the result is written. cuda.memalloc is really time
        consuming (up to 95% of the total latency is spent in
        cuda.memalloc only)

        :param other: scalar or darray to substract
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """
        return self.substract(other, dst=self)

    def multiply(self, other: object,
                 dst: 'darray' = None) -> 'darray':  # array.multiply(object)
        """Efficient multiplication of a darray with another object.
        Can be a darray or a scalar. If dst is None, normal __add__
        is called.
        This method is much more efficient than the __mul__ method
        because __mul__ implies a copy of the array.
        cuda.memalloc is really time consuming (up to 95% of the total
        latency is spent in cuda.memalloc only).

        :param dst: darray where to write the result
        :type dst: darray
        :param other: numpy.ndarray or scalar to multiply
        :type other: [scalar, numpy.ndarray]
        :raises ValueError: If the size, dtype or shape of dst is not matching
        :return: darray where the result is written. Can be dst or self
        :rtype: darray
        """
        if dst is None:
            return self * other

        if isinstance(other, (numpy.number, int, float)):
            if other == 1:
                cuda.memcpy_dtod_async(dst.allocation,
                                       self._allocation,
                                       self._nbytes,
                                       self._stream)
            self._cu_axpbz(
                x_array=self,
                z_array=dst,
                a_scalar=other,
                b_scalar=0,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)

        elif isinstance(other, type(self)):

            if other.shape != self._shape:
                raise ValueError(
                    f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(
                    f"Type mismatch : {other.dtype} != {self.dtype}")

            self._cu_eltwise_mult(
                x_array=self,
                y_array=other,
                z_array=dst,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
        else:
            raise TypeError(
                f"unsupported operand type(s) for *: '{type(self)}' \
                    and '{type(other)}'")
        return dst

    mul = multiply

    def __mul__(self, other: object) -> 'darray':  # array * object
        """Non-Efficient multiplication of a darray with another object.
        This multiplication is element-wise multiplication, not matrix
        multiplication. For matrix multiplication please refer to matmul.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to multiply
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """
        if isinstance(other, (numpy.number, int, float)):

            if other == 1:
                return self.copy()

            result = self.copy()
            self._cu_axpbz(
                x_array=self,
                z_array=result,
                a_scalar=other,
                b_scalar=0,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
            return result

        if isinstance(other, type(self)):

            if other.shape != self._shape:
                raise ValueError(
                    f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(
                    f"Type mismatch : {other.dtype} != {self.dtype}")

            result = self.copy()
            self._cu_eltwise_mult(
                x_array=self,
                y_array=other,
                z_array=result,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
            return result

        else:
            raise TypeError(
                f"unsupported operand type(s) for *: '{type(self)}' \
                    and '{type(other)}'")

    __rmul__ = __mul__  # object * array

    def __imul__(self, other: object) -> 'darray':
        """Non-Efficient multiplication of a darray with another object.
        This multiplication is element-wise multiplication, not matrix
        multiplication. For matrix multiplication please refer to matmul.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to multiply
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """
        return self.multiply(other, dst=self)

    def divide(self, other: object, dst: 'darray' = None) -> 'darray':
        """Efficient division of a darray with another object.
        Can be a darray or a scalar. If dst is None, normal __div__
        is called.
        This method is much more efficient than the __div__ method
        because __div__ implies a copy of the array.
        cuda.memalloc is really time consuming (up to 95% of the total
        latency is spent in cuda.memalloc only).

        :param other: scalar or darray to divide by
        :type other: [scalar, darray]
        :param dst: darray where to write the result
        :type dst: darray
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """

        if dst is None:
            dst = self.copy()

        if isinstance(other, (numpy.number, int, float)):

            if other == 1:
                cuda.memcpy_dtod_async(dst.allocation,
                                       self._allocation,
                                       self._nbytes,
                                       self._stream)
            elif other == 0:
                raise ZeroDivisionError("Division by zero")
            else:

                self._cu_scal_div(
                    x_array=self,
                    z_array=dst,
                    a_scalar=numpy.float32(other),
                    size=self._size,
                    block=self._block,
                    grid=self._grid,
                    stream=self._stream)

        elif isinstance(other, type(self)):

            if other.shape != self._shape:
                raise ValueError(
                    f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(
                    f"Type mismatch : {other.dtype} != {self.dtype}")

            self._cu_eltwise_div(
                x_array=self,
                y_array=other,
                z_array=dst,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
        else:
            raise TypeError(
                f"unsupported operand type(s) for /: '{type(self)}' \
                    and '{type(other)}'")

        return dst

    div = divide

    def __div__(self, other: object) -> 'darray':  # array / object
        """Non-Efficient division of a darray with another object.
        This division is element-wise.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to divide by
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """
        return self.divide(other)

    def reversed_divide(self, other: object, dst: 'darray') -> 'darray':
        """Efficient division of a darray with another object.
        Can be a darray or a scalar. If dst is None, normal __rdiv__
        is called.
        This method is much more efficient than the __rdiv__ method
        because __rdiv__ implies a copy of the array.
        cuda.memalloc is really time consuming (up to 95% of the total
        latency is spent in cuda.memalloc only).

        :param other: scalar or darray to divide by
        :type other: [scalar, darray]
        :param dst: darray where to write the result
        :type dst: darray
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """
        if dst is None:
            return other / self

        if isinstance(other, (numpy.number, int, float)):
            self._cu_invscal_div(
                x_array=self,
                z_array=dst,
                a_scalar=other,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
        elif isinstance(other, type(self)):

            if other.shape != self._shape:
                raise ValueError(
                    f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(
                    f"Type mismatch : {other.dtype} != {self.dtype}")

            self._cu_eltwise_div(
                x_array=self,
                y_array=other,
                z_array=dst,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
        else:
            raise TypeError(
                f"unsupported operand type(s) for /: '{type(self)}' \
                    and '{type(other)}'")
        return dst

    rdiv = reversed_divide

    def __rdiv__(self, other: object) -> 'darray':  # object / array
        """Non-Efficient reverse division of an object by darray.
        This division is element-wise.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to divide by
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """

        if isinstance(other, (numpy.number, int, float)):

            result = self.copy()
            self._cu_invscal_div(
                x_array=self,
                z_array=result,
                a_scalar=other,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
            return result

        if isinstance(other, type(self)):

            if other.shape != self._shape:
                raise ValueError(
                    f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(
                    f"Type mismatch : {other.dtype} != {self.dtype}")

            result = self.copy()
            self._cu_eltwise_div(
                x_array=self,
                y_array=other,
                z_array=result,
                size=self._size,
                block=self._block,
                grid=self._grid,
                stream=self._stream)
            return result

        else:
            raise TypeError(
                f"unsupported operand type(s) for /: '{type(self)}' \
                    and '{type(other)}'")

    def __idiv__(self, other: object) -> 'darray':
        """Non-Efficient division of a darray with another object.
        This division is element-wise.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to divide by
        :type other: [scalar, darray]
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """
        return self.divide(other, dst=self)

    __truediv__ = __div__
    __itruediv__ = __idiv__
    __rtruediv__ = __rdiv__

    def __len__(self) -> int:
        """Returns the length of the first dimension of the array.

        :return: Size of the first dimension of the array
        :rtype: int
        """

        if isinstance(self._shape, tuple) and len(self._shape) == 0:
            raise TypeError("len() of unsized object")

        return self._shape[0]

    def __abs__(self) -> 'darray':
        """Returns the absolute value of the array.

        :return: Absolute value of the array
        :rtype: darray
        """

        return self.absolute()

    def absolute(self, dst: 'darray' = None) -> 'darray':
        """Returns the absolute value of the array.

        :return: Absolute value of the array
        :rtype: darray
        """

        if dst is None:
            dst = self.copy()
        else:
            if dst.shape != self.shape:
                raise ValueError(
                    f"Shape mismatch : {dst.shape} != {self.shape}")
            if dst.dtype != self.dtype:
                raise ValueError(
                    f"Type mismatch : {dst.dtype} != {self.dtype}")

        self._cu_eltwise_abs(
            x_array=self,
            z_array=dst,
            size=self._size,
            block=self._block,
            grid=self._grid,
            stream=self._stream)

        return dst

    def __pow__(self, other: object) -> 'darray':
        raise NotImplementedError("Power operator not implemented")


def transpose(axes: tuple,
              src: darray,
              dst: darray = None) -> darray:
    """Returns a darray with the axes transposed.

    :param axes: Axes to transpose
    :type axes: tuple
    :param src: darray to transpose
    :type src: darray
    :return: Transposed darray
    :rtype: darray
    """
    return src.transpose(*axes, dst=dst)


def multiply(src: darray,
             other: object,
             dst: darray = None) -> darray:
    """Returns the multiplication of two darrays.
    It works that way::

        result = src * other

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Multiplication of the two darrays
    :rtype: darray
    """
    return src.multiply(other, dst)


def add(src: darray,
        other: object,
        dst: darray = None) -> darray:
    """Returns the addition of two darrays.
    It works that way::

        result = src + other

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Addition of the two darrays
    :rtype: darray
    """
    return src.add(other, dst)


def substract(src: darray,
              other: object,
              dst: darray = None) -> darray:
    """Returns the substraction of two darrays.
    It works that way::

        result = src - other

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Substraction of the two darrays
    :rtype: darray
    """
    return src.substract(other, dst)


def divide(src: darray,
           other: object,
           dst: darray = None) -> darray:
    """Returns the division of a darray by an object.
    It works that way::

        result = src / other

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Division of the two darrays
    :rtype: darray
    """
    return src.divide(other, dst)


def reversed_divide(
        src: darray,
        other: object,
        dst: darray = None) -> darray:
    """Returns the division of a darray and an object.
    It works that way::

        result = other / src

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Division of the two darrays
    :rtype: darray
    """
    return src.reversed_divide(other, dst)


def reversed_substract(
        src: darray,
        other: object,
        dst: darray = None) -> darray:
    """Returns the substraction of a darray and an object.
    It works that way::

        result = other - src

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Substraction of the two darrays
    :rtype: darray
    """
    return src.reversed_substract(other, dst)


def zeros(
        shape: tuple,
        dtype: dolphin.dtype = dolphin.dtype.float32) -> darray:
    """Returns a darray for a given shape and dtype filled with zeros.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param shape: Shape of the array
    :type shape: tuple
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with zeros
    :rtype: darray
    """

    return darray(numpy.zeros(shape, dtype=dtype.numpy_dtype))


def zeros_like(other: Union[darray, numpy.array]) -> darray:
    """Returns a darray filled with zeros with the same shape and dtype as
    another darray.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param other: darray to copy the shape and type from
    :type other: darray
    :return: darray filled with zeros
    :rtype: darray
    """
    if isinstance(other, darray):
        return zeros(shape=other.shape, dtype=other.dtype)
    return zeros(shape=other.shape,
                 dtype=dolphin.dtype.from_numpy_dtype(other.dtype))


def ones(shape: tuple, dtype: dolphin.dtype = dolphin.dtype.float32) -> darray:
    """Returns a darray for a given shape and dtype filled with ones.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param shape: Shape of the array
    :type shape: tuple
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with ones
    :rtype: darray
    """

    return darray(numpy.ones(shape, dtype=dtype.numpy_dtype))


def ones_like(other: Union[darray, numpy.array]) -> darray:
    """Returns a darray filled with ones with the same shape and dtype as
    another darray.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param other: darray to copy the shape and type from
    :type other: darray
    :return: darray filled with ones
    :rtype: darray
    """

    if isinstance(other, darray):
        return ones(shape=other.shape, dtype=other.dtype)
    return ones(shape=other.shape,
                dtype=dolphin.dtype.from_numpy_dtype(other.dtype))


def empty(
        shape: tuple,
        dtype: dolphin.dtype = dolphin.dtype.float32) -> darray:
    """Returns a darray of a given shape and dtype without
    initializing entries.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param shape: Shape of the array
    :type shape: tuple
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with random values
    :rtype: darray
    """

    return darray(numpy.empty(shape, dtype=dtype.numpy_dtype))


def empty_like(other: Union[darray, numpy.array]) -> darray:
    """Returns a darray without initializing entries with the same shape
    and dtype as another darray.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param other: darray to copy the shape and type from
    :type other: darray
    :return: darray filled with random values
    :rtype: darray
    """

    if isinstance(other, darray):
        return empty(shape=other.shape, dtype=other.dtype)
    return empty(shape=other.shape,
                 dtype=dolphin.dtype.from_numpy_dtype(other.dtype))


def absolute(array: darray, dst: darray = None) -> darray:
    """Returns the absolute value of a darray.

    :param array: darray to take the absolute value of
    :type array: darray
    :return: Absolute value of the darray
    :rtype: darray
    """

    return array.absolute(dst)


abs = absolute

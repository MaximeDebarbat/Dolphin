
import pycuda.driver as cuda  # pylint: disable=import-error
import tensorrt as trt
import math
import numbers

import numpy
import dolphin

import time


class darray(dolphin.CudaBase):
    """_summary_
    """

    def __init__(self,
                 array: numpy.ndarray = None,
                 shape: tuple = None,
                 dtype: dolphin.dtype = None,
                 stream: cuda.Stream = None,
                 ) -> None:

        super(darray, self).__init__()

        if array is not None:
            _a_dtype = dolphin.dtype.from_numpy_dtype(array.dtype)
            _a_shape = array.shape

            if shape is not None and shape != _a_shape:
                raise ValueError(f"Shape mismatch : {shape} != {_a_shape}")
            if dtype is not None and dtype != _a_dtype:
                raise ValueError(f"Type mismatch : {dtype.numpy_dtype} \
!= {_a_dtype}")
            if shape is None:
                shape = _a_shape
            if dtype is None:
                dtype = _a_dtype

        self._stream: cuda.Stream = stream
        self._dtype: dolphin.dtype = dtype
        self._shape: tuple = shape
        self._size: int = trt.volume(self._shape)
        self._nbytes: int = int(self._size * self._dtype.itemsize)

        self._allocation: cuda.DeviceAllocation = cuda.mem_alloc(self._nbytes)

        if self._stream is None:
            if array is not None:
                cuda.memcpy_htod(self._allocation, array.flatten(order="C"))
        else:
            if array is not None:
                cuda.memcpy_htod_async(self._allocation,
                                       array,
                                       self._stream)

        self._block = (int(min(self.MAX_THREADS_PER_BLOCKS, self._size)), 1, 1)
        self._grid = (int(math.ceil(self._size / self._block[0])), 1)

        self._cu_axpbz = dolphin.cufunc.CU_AXPBZ
        self._cu_axpbyz = dolphin.cufunc.CU_AXPBYZ
        self._cu_eltwise_mult = dolphin.cufunc.CU_ELTWISE_MULT
        self._cu_eltwise_div = dolphin.cufunc.CU_ELTWISE_DIV
        self._cu_scal_div = dolphin.cufunc.CU_SCAL_DIV
        self._cu_invscal_div = dolphin.cufunc.CU_INVSCAL_DIV
        self._cu_eltwise_cast = dolphin.cufunc.CU_ELTWISE_CAST
        self._cu_eltwise_abs = dolphin.cufunc.CU_ELTWISE_ABS
        self._cu_transpose = dolphin.cufunc.CU_TRANSPOSE

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
                strides.append(strides[-1]*max(1, s))
            return tuple(strides[::-1])
        else:
            return ()

    def from_ndarray(self, array: numpy.ndarray) -> None:
        """Creates a darray from a numpy array.

        :param array: Numpy array create the darray from
        :type array: numpy.ndarray
        """

        self.__init__(array=array, stream=self._stream)

    def astype(self, dtype: dolphin.dtype) -> None:
        """Converts the darray to a different dtype.
        Note that a copy from device to device is performed.

        :param dtype: Dtype to convert the darray to
        :type dtype: dolphin.dtype
        """
        if dtype == self._dtype:
            return self.copy()

        res = darray(shape=self._shape,
                     dtype=dtype,
                     stream=self._stream)

        # print(f"ASTYPE DEBUG : \nres.nbytes = {res._nbytes}\nself.nbytes = {self._nbytes}\nres.dtype = {res._dtype}\nself.dtype = {self._dtype}\nres.shape = {res._shape}\nself.shape = {self._shape}\nres.size = {res._size}\nself.size = {self._size}\nres.allocation = {res._allocation}\nself.allocation = {self._allocation}\nres.block = {res._block}\nself.block = {self._block}\nres.grid = {res._grid}\nself.grid = {self._grid}\nres.stream = {res._stream}\nself.stream = {self._stream}\nres.cu_eltwise_cast = {res._cu_eltwise_cast}\nself.cu_eltwise_cast = {self._cu_eltwise_cast}\n")
        if self._stream is None:
            self._cu_eltwise_cast(self,
                                  res,
                                  self._size,
                                  block=self._block,
                                  grid=self._grid)
        else:
            self._cu_eltwise_cast(self,
                                  res,
                                  self._size,
                                  block=self._block,
                                  grid=self._grid,
                                  stream=self._stream)

        return res

    def transpose(self, *axes: int, dst: 'darray' = None) -> 'darray':
        """Transposes the darray according to the axes.

        :param axes: Axes to permute
        :type axes: Tuple[int]
        :return: Transposed darray
        :rtype: darray
        """

        if len(axes) != len(self._shape):
            raise ValueError(f"axes don't match array")

        if not all(isinstance(v, int) for v in axes):
            raise ValueError(f"axes must be integers")

        strides = self.strides
        new_shape = [self.shape[i] for i in axes]
        new_strides = [strides[i] for i in axes]

        if dst is not None:
            if dst.shape != tuple(new_shape):
                raise ValueError(f"dst shape doesn't match")
            if dst.dtype != self.dtype:
                raise ValueError(f"dst dtype doesn't match")

        new_shape = numpy.array(new_shape,
                                dtype=numpy.uint32)
        new_strides = numpy.array(new_strides,
                                  dtype=numpy.uint32)

        new_shape_allocation = cuda.mem_alloc(new_shape.nbytes)
        new_strides_allocation = cuda.mem_alloc(new_strides.nbytes)

        if self._stream is None:
            cuda.memcpy_htod(new_shape_allocation, new_shape)
            cuda.memcpy_htod(new_strides_allocation, new_strides)
        else:
            cuda.memcpy_htod_async(new_shape_allocation,
                                   new_shape,
                                   self._stream)
            cuda.memcpy_htod_async(new_strides_allocation,
                                   new_strides,
                                   self._stream)

        if dst is not None:
            res = dst
        else:
            res = darray(shape=tuple(new_shape),
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
        res = darray(shape=self._shape,
                     dtype=self._dtype,
                     stream=self._stream)

        if self._stream is None:
            cuda.memcpy_dtod(res._allocation,
                             self._allocation,
                             self._nbytes)
        else:
            cuda.memcpy_dtod_async(res._allocation,
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
        if self._stream is None:
            cuda.memcpy_dtoh(res, self._allocation)
        else:
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
            return self + other

        if dst.nbytes != self._nbytes:
            raise ValueError(f"Size mismatch : \
{dst.nbytes} != {self._nbytes}")
        if dst.dtype != self._dtype:
            raise ValueError(f"dtype mismatch : \
{dst.dtype} != {self._dtype}")
        if dst.shape != self._shape:
            raise ValueError(f"Shape mismatch : \
{dst.shape} != {self._shape}")

        if isinstance(other, int) or isinstance(other, float) or \
            (isinstance(other, numpy.ndarray) and
             other.shape == () and
             numpy.issubdtype(other.dtype, numpy.number)):
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
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

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

        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 0:
                return self.copy()
            else:

                result = self.copy()

                self._cu_axpbz(
                            x_array=self,
                            z_array=result,
                            a_scalar=1,
                            b_scalar=other,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return result

        elif isinstance(other, darray):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : \
{other.shape} != {self._shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : \
{other.dtype} != {self._dtype}")

            result = self.copy()
            self._cu_axpbyz(
                            x_array=self,
                            y_array=other,
                            z_array=result,
                            a_scalar=1,
                            b_scalar=1,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return result

        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

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

        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 0:
                return self
            else:

                self._cu_axpbz(
                            x_array=self,
                            z_array=self,
                            a_scalar=1,
                            b_scalar=other,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return self

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
                            z_array=self,
                            a_scalar=1,
                            b_scalar=1,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return self

        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

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
            return self - other

        if dst.nbytes != self._nbytes:
            raise ValueError(f"Size mismatch : \
{dst.nbytes} != {self._nbytes}")
        if dst.dtype != self._dtype:
            raise ValueError(f"dtype mismatch : \
{dst.dtype} != {self._dtype}")
        if dst.shape != self._shape:
            raise ValueError(f"Shape mismatch : \
{dst.shape} != {self._shape}")

        if isinstance(other, int) or isinstance(other, float) or \
            (isinstance(other, numpy.ndarray) and
             other.shape == () and
             numpy.issubdtype(other.dtype, numpy.number)):
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
            raise TypeError(f"unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")

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
        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 0:
                return self.copy()
            else:

                result = self.copy()
                self._cu_axpbz(
                            x_array=self,
                            z_array=result,
                            a_scalar=1,
                            b_scalar=-other,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return result

        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

            result = self.copy()
            self._cu_axpbyz(
                            x_array=self,
                            y_array=other,
                            z_array=result,
                            a_scalar=1,
                            b_scalar=-1,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return result

        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")

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
            return self.__rsub__(other)

        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):
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
        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            result = self.copy()
            self._cu_axpbz(
                        x_array=self,
                        z_array=result,
                        a_scalar=-1,
                        b_scalar=other,
                        size=self._size,
                        block=self._block,
                        grid=self._grid,
                        stream=self._stream)

            return result

        elif isinstance(other, darray):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : \
{other.shape} != {self._shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : \
{other.dtype} != {self._dtype}")

            result = self.copy()
            self._cu_axpbyz(
                            x_array=self,
                            y_array=other,
                            z_array=result,
                            a_scalar=-1,
                            b_scalar=1,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return result

        else:
            raise TypeError(f"unsupported operand type(s) for -: \
'{type(self).__name__}' and '{type(other).__name__}'")

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
        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 0:
                return self
            else:

                self._cu_axpbz(
                            x_array=self,
                            z_array=self,
                            a_scalar=1,
                            b_scalar=-other,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return self

        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

            self._cu_axpbyz(
                            x_array=self,
                            y_array=other,
                            z_array=self,
                            a_scalar=1,
                            b_scalar=-1,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return self

        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")

    def multiply(self, other: object,
                 dst: 'darray' = None) -> 'darray':  # array.multiply(object)
        """Efficient multiplication of a darray with another object.
        Can be a darray or a scalar. If dst is None, normal __add__
        is called. Also this multiplication is efficient.
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

        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 1:
                cuda.memcpy_dtod_async(dst.allocation,
                                       self._allocation,
                                       self._nbytes,
                                       self._stream)
            else:

                self._cu_axpbz(
                            x_array=self,
                            z_array=dst,
                            a_scalar=other,
                            b_scalar=0,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)

        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

            self._cu_eltwise_mult(
                            x_array=self,
                            y_array=other,
                            z_array=dst,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")
        return dst

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
        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 1:
                return self.copy()
            else:

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

        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

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
            raise TypeError(f"unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")

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

        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 1:
                return self
            else:

                self._cu_axpbz(
                            x_array=self,
                            z_array=self,
                            a_scalar=other,
                            b_scalar=0,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return self

        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

            self._cu_eltwise_mult(
                            x_array=self,
                            y_array=other,
                            z_array=self,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return self

        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")

    def divide(self, other: object, dst: 'darray') -> 'darray':
        """Efficient division of a darray with another object.

        :param other: _description_
        :type other: object
        :param dst: _description_
        :type dst: darray
        :return: _description_
        :rtype: darray
        """
        if dst is None:
            return self / other
        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

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
        elif type(other) == type(self):
            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")
            self._cu_eltwise_div(
                            x_array=self,
                            y_array=other,
                            z_array=dst,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)

        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")
        return dst

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

        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 1:
                return self.copy()
            elif other == 0:
                raise ZeroDivisionError("Division by zero")
            else:

                result = self.copy()
                self._cu_scal_div(
                            x_array=self,
                            z_array=result,
                            a_scalar=numpy.float32(other),
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return result

        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

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
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")

    def reversed_divide(self, other: object, dst: 'darray') -> 'darray':
        """Efficient reversed division of a darray with another object.

        :param other: _description_
        :type other: object
        :param dst: _description_
        :type dst: darray
        :return: _description_
        :rtype: darray
        """
        if dst is None:
            return other / self

        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):
            self._cu_invscal_div(
                        x_array=self,
                        z_array=dst,
                        a_scalar=other,
                        size=self._size,
                        block=self._block,
                        grid=self._grid,
                        stream=self._stream)
        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

            result = self.copy()
            self._cu_eltwise_div(
                            x_array=self,
                            y_array=other,
                            z_array=dst,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")
        return dst

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

        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

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

        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

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
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")

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
        if isinstance(other, int) or isinstance(other, float) or \
           (isinstance(other, numpy.ndarray) and
           other.shape == () and
           numpy.issubdtype(other.dtype, numpy.number)):

            if other == 1:
                return self
            elif other == 0:
                raise ZeroDivisionError("Division by zero")
            else:

                self._cu_scal_div(
                            x_array=self,
                            z_array=self,
                            a_scalar=numpy.float32(other),
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return self

        elif type(other) == type(self):

            if other.shape != self._shape:
                raise ValueError(f"Shape mismatch : {other.shape} != {self.shape}")
            if other.dtype != self._dtype:
                raise ValueError(f"Type mismatch : {other.dtype} != {self.dtype}")

            self._cu_eltwise_div(
                            x_array=self,
                            y_array=other,
                            z_array=self,
                            size=self._size,
                            block=self._block,
                            grid=self._grid,
                            stream=self._stream)
            return self

        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")

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

        self._cu_eltwise_abs(
                    x_array=self,
                    size=self._size,
                    block=self._block,
                    grid=self._grid,
                    stream=self._stream)

        return self

    def __pow__(self, other: object) -> 'darray':
        raise NotImplementedError("Power operator not implemented")

    def __del__(self):
        try:
            self._allocation.free()
        except AttributeError:
            pass


def transpose(axes: tuple, array: darray, dst: darray = None) -> darray:
    """Returns a darray with the axes transposed.

    :param axes: Axes to transpose
    :type axes: tuple
    :param array: darray to transpose
    :type array: darray
    :return: Transposed darray
    :rtype: darray
    """
    return array.transpose(*axes, dst=dst)


def multiply(array1: darray, other: object, dst: darray = None) -> darray:
    """Returns the multiplication of two darrays.

    :param array1: First darray
    :type array1: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Multiplication of the two darrays
    :rtype: darray
    """
    return array1.multiply(other, dst)


def add(array1: darray, other: object, dst: darray = None) -> darray:
    """Returns the addition of two darrays.

    :param array1: First darray
    :type array1: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Addition of the two darrays
    :rtype: darray
    """
    return array1.add(other, dst)


def substract(array1: darray, other: object, dst: darray = None) -> darray:
    """Returns the substraction of two darrays.

    :param array1: First darray
    :type array1: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Substraction of the two darrays
    :rtype: darray
    """
    return array1.substract(other, dst)


def divide(array1: darray, other: object, dst: darray = None) -> darray:
    """Returns the division of two darrays.

    :param array1: First darray
    :type array1: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Division of the two darrays
    :rtype: darray
    """
    return array1.divide(other, dst)


def reversed_divide(array1: darray, other: object, dst: darray = None) -> darray:
    """Returns the division of two darrays.

    :param array1: First darray
    :type array1: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Division of the two darrays
    :rtype: darray
    """
    return array1.reversed_divide(other, dst)


def reversed_substract(array1: darray, other: object, dst: darray = None) -> darray:
    """Returns the substraction of two darrays.

    :param array1: First darray
    :type array1: darray
    :param other: Second darray or scalar
    :type other: [darray, scalar]
    :return: Substraction of the two darrays
    :rtype: darray
    """
    return array1.reversed_substract(other, dst)


def zeros(shape: tuple, dtype: dolphin.dtype = dolphin.dtype.float32) -> darray:
    """Returns a darray filled with zeros.

    :param shape: Shape of the array
    :type shape: tuple
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with zeros
    :rtype: darray
    """

    return darray(numpy.zeros(shape, dtype=dtype.numpy_dtype))


def zeros_like(other: darray) -> darray:
    """Returns a darray filled with zeros.

    :param other: darray to copy the shape and type from
    :type other: darray
    :return: darray filled with zeros
    :rtype: darray
    """

    return zeros(shape=other.shape, dtype=other.dtype)


def ones(shape: tuple, dtype: dolphin.dtype = dolphin.dtype.float32) -> darray:
    """Returns a darray filled with ones.

    :param shape: Shape of the array
    :type shape: tuple
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with ones
    :rtype: darray
    """

    return darray(numpy.ones(shape, dtype=dtype.numpy_dtype))


def empty(shape: tuple, dtype: dolphin.dtype = dolphin.dtype.float32) -> darray:
    """Returns a darray filled with random values.

    :param shape: Shape of the array
    :type shape: tuple
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with random values
    :rtype: darray
    """

    return darray(numpy.empty(shape, dtype=dtype.numpy_dtype))


def test(dolphin_dtype: dolphin.dtype = dolphin.dtype.float32,
         shape: tuple = (1000, 1000), n_iter: int = 1000):

    print(f"\nRunning tests for {dolphin_dtype}...\n")

    N_ITER = int(n_iter)

    #######################
    #    CREATION TEST    #
    #######################

    dummy = numpy.random.rand(*shape).astype(dolphin_dtype.numpy_dtype)
    cuda_array = darray(dummy)
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Creation test 1 failed"

    print(f"Creation test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    cuda_array = darray(shape=shape, dtype=dolphin_dtype)
    cuda_array.from_ndarray(dummy)
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Creation test 2 failed"

    print(f"Creation test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # zeros

    dummy = numpy.zeros(shape=shape, dtype=dolphin_dtype)
    cuda_array = zeros(shape=shape, dtype=dolphin_dtype)

    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Creation test 3 failed"

    print(f"Creation test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # zeros_like

    dummy = numpy.zeros(shape=shape, dtype=dolphin_dtype)
    cuda_array_1 = zeros_like(cuda_array)

    #######################
    #    ADD SCAL TEST    #
    #######################

    # array + scal
    dummy = dummy + 8
    cuda_array = cuda_array + 8
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add scalar test 1 failed"

    print(f"Addscal  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # scal + array
    dummy = 8 + dummy
    cuda_array = 8 + cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add scalar test 2 failed"

    print(f"Addscal  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array += scal
    dummy += 8
    cuda_array += 8
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add scalar test 3 failed"

    print(f"Addscal  test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array.add(scal)
    dummy += 8
    cuda_array = cuda_array.add(8)
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add scalar test 4 failed"

    print(f"Addscal  test 4 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array.add(scal, dst)
    dummy += 8
    cuda_array = cuda_array.add(8)
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add scalar test 5 failed"

    print(f"Addscal  test 5 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    #######################
    #    ADD ARRA TEST    #
    #######################

    # array + array
    dummy = dummy + dummy
    cuda_array = cuda_array + cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add array test 1 failed"

    print(f"Addarra  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # scal + array
    dummy = dummy + dummy
    cuda_array = cuda_array + cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add array test 2 failed"

    print(f"Addarra  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array += scal
    dummy += dummy
    cuda_array += cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add array test 3 failed"

    print(f"Addarra  test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    #######################
    #    ADD FAIL TEST    #
    #######################

    try:
        cuda_array = cuda_array + "test"
        raise AssertionError("Addfail test 1 failed")
    except TypeError:
        print(f"Addfail  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    #######################
    #    SUB SCAL TEST    #
    #######################

    # array - scal
    dummy = dummy - 8
    cuda_array = cuda_array - 8
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Sub scalar test 1 failed"

    print(f"Subscal  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # scal - array
    dummy = 8 - dummy
    cuda_array = 8 - cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Sub scalar test 2 failed"

    print(f"Subscal  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array - scal
    dummy -= 8
    cuda_array -= 8
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Sub scalar test 3 failed"

    print(f"Subscal  test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    #######################
    #    SUB ARRA TEST    #
    #######################

    # array - other
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)
    dother = darray(other)

    dummy = dummy - other
    cuda_array = cuda_array - dother
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Sub array test 1 failed"

    print(f"Subarra  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # other - array
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)
    dother = darray(other)

    dummy = other - dummy
    cuda_array = dother - cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Sub array test 2 failed"

    print(f"Subarra  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array -= other
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)
    dother = darray(other)

    dummy -= other
    cuda_array -= dother
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Sub array test 3 failed"

    print(f"Subarra  test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    #######################
    #    MUL SCAL TEST    #
    #######################

    # array * scal
    dummy = dummy * 8
    cuda_array = cuda_array * 8
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Mul scalar test 1 failed"

    print(f"Mulscal  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # scal * array
    dummy = 8 * dummy
    cuda_array = 8 * cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Mul scalar test 2 failed"

    print(f"Mulscal  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array *= scal
    dummy *= 8
    cuda_array *= 8
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Mul scalar test 3 failed"

    print(f"Mulscal  test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    #######################
    #    MUL ARRA TEST    #
    #######################

    # array * other
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)*2
    dother = darray(other)

    dummy = dummy * other
    cuda_array = cuda_array * dother
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Mul array test 1 failed"

    print(f"Mularra  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # other * array
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)*2
    dother = darray(other)

    dummy = other * dummy
    cuda_array = dother * cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Mul array test 2 failed"

    print(f"Mularra  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array *= other
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)*2
    dother = darray(other)

    dummy *= other
    cuda_array *= dother
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Mul array test 3 failed"

    print(f"Mularra  test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    #######################
    #    DIV SCAL TEST    #
    #######################

    # array / scal
    dummy = dummy / 8
    cuda_array = cuda_array / 8
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dolphin_dtype.numpy_dtype))

    assert diff < 1e-5, "Div scalar test 1 failed"

    print(f"Divscal  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0].astype(dolphin_dtype.numpy_dtype)}")

    # scal / array
    dummy = 8 / (dummy+1)
    cuda_array = 8 / (cuda_array+1)
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dolphin_dtype.numpy_dtype))

    assert diff < 1e-5, "Div scalar test 2 failed"

    print(f"Divscal  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0].astype(dolphin_dtype.numpy_dtype)}")

    # array /= scal
    dummy /= 8
    cuda_array /= 8
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dolphin_dtype.numpy_dtype))

    assert diff < 1e-5, "Div scalar test 3 failed"

    print(f"Divscal  test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0].astype(dolphin_dtype.numpy_dtype)}")

    #######################
    #    DIV ARRA TEST    #
    #######################

    dummy = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)*64
    cuda_array = darray(dummy)

    # array / other
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)*2
    dother = darray(other)

    #print(f"other : {other} | dother : {dother}")

    dummy = dummy / other
    cuda_array = cuda_array / dother
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dolphin_dtype.numpy_dtype))

    assert diff < 1e-5, "Div array test 1 failed"

    print(f"Divarra  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0].astype(dolphin_dtype.numpy_dtype)}")

    dummy = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)*2
    cuda_array = darray(dummy)

    # other / array
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)*8
    dother = darray(other)

    dummy = other / dummy
    cuda_array = dother / cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dolphin_dtype.numpy_dtype))

    assert diff < 1e-5, "Div array test 2 failed"

    print(f"Divarra  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0].astype(dolphin_dtype.numpy_dtype)}")

    # array /= other
    other = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)*2
    dother = darray(other)

    dummy /= other
    cuda_array /= dother
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dolphin_dtype.numpy_dtype))

    print(f"Divarra  test 3 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0].astype(dolphin_dtype.numpy_dtype)}")

    assert diff < 1e-5, "Div array test 3 failed"

    dummy = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)
    cuda_array = darray(dummy)

    other = numpy.zeros(shape, dtype=dolphin_dtype.numpy_dtype)
    dother = darray(other)

    try:
        dummy /= other
        cuda_array /= dother
        diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dolphin_dtype.numpy_dtype))
    except ZeroDivisionError:
        print(f"Divarra  test 4 : {diff} | Division by zero test passed")
    except Exception as e:
        pass

    #######################
    #      CAST TEST      #
    #######################

    dummy = numpy.ones(shape, dtype=dolphin_dtype.numpy_dtype)
    cuda_array = darray(dummy)

    for dtype in dolphin.dtype:
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = cuda_array.astype(dtype)
        diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dtype.numpy_dtype))

        assert diff < 1e-5, f"Cast test failed for {dtype}"

        print(f"Cast  test : {diff} | {dolphin_dtype} -> {dtype} test passed")

    #######################
    #      ABS TEST       #
    #######################

    dummy = numpy.random.rand(*shape)*10
    dummy = dummy.astype(dolphin_dtype.numpy_dtype)
    cuda_array = darray(dummy)

    dummy = numpy.abs(dummy)
    cuda_array = abs(cuda_array)
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy.astype(dtype.numpy_dtype))

    print(f"Absolute test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0].astype(dolphin_dtype.numpy_dtype)}")

    assert diff < 1e-5, "Absolute test 1 failed"

    #######################
    #    TRANSPOSE TEST   #
    #######################

    dummy = numpy.random.rand(*shape)*10
    dummy = dummy.astype(dolphin_dtype.numpy_dtype)
    perm = tuple([i for i in range(len(shape))][::-1])

    transposed_dummy = dummy.transpose(*perm)

    cuda_array = darray(dummy)
    transposed_cuda_array = cuda_array.transpose(*perm)

    diff = numpy.linalg.norm(transposed_cuda_array.ndarray - transposed_dummy.astype(dtype.numpy_dtype))

    print(f"Transpose test 1 : {diff} | first number : \
{transposed_cuda_array.ndarray[0, 0]} == {transposed_dummy[0, 0].astype(dolphin_dtype.numpy_dtype)}")

    assert diff < 1e-5, "Transpose test 1 failed"

    #######################
    #       SUCCESS       #
    #######################

    print(f"All tests passed for {dolphin_dtype}")

    #######################
    #     TIME  TESTS     #
    #######################

    dummy = numpy.random.rand(*shape)*10
    cuda_array = darray(dummy)

    t1 = time.time()
    for _ in range(N_ITER):
        dummy_res = 15*dummy.transpose(1, 0)/2 + 5*dummy.transpose(1, 0)/3
    numpy_time = 1000*(time.time() - t1)/N_ITER

    print(f"numpy time              : {numpy_time}")

    t1 = time.time()
    for _ in range(N_ITER):
        cuda_res = 15*cuda_array.transpose(1, 0)/2 + 5*cuda_array.transpose(1, 0)/3
    cuda_time = 1000*(time.time() - t1)/N_ITER

    print(f"Non efficient cuda time : {cuda_time}")

    res_cuda_array_1 = darray(dummy.transpose(1, 0))
    res_cuda_array_2 = darray(dummy.transpose(1, 0))
    res_cuda_array_3 = darray(dummy.transpose(1, 0))

    t1 = time.time()
    for _ in range(N_ITER):
        transpose((1, 0), cuda_array, res_cuda_array_1)
        divide(res_cuda_array_1, 2, res_cuda_array_1)
        multiply(res_cuda_array_1, 15, res_cuda_array_1)

        transpose((1, 0), cuda_array, res_cuda_array_2)
        divide(res_cuda_array_2, 3, res_cuda_array_2)
        multiply(res_cuda_array_2, 5, res_cuda_array_2)

        add(res_cuda_array_1, res_cuda_array_2, res_cuda_array_3)
    cuda_time = 1000*(time.time() - t1)/N_ITER

    print(f"efficient cuda time     : {cuda_time}")

    diff1 = numpy.linalg.norm(cuda_res.ndarray - dummy_res)
    diff2 = numpy.linalg.norm(res_cuda_array_3.ndarray - dummy_res)

    assert diff1 < 1e-5, "Time test 1 failed"
    assert diff2 < 1e-5, "Time test 2 failed"

if __name__ == "__main__":

    shape = (100, 100)

    test(dolphin.dtype.float32, shape=shape, n_iter=1e3)
    test(dolphin.dtype.float64, shape=shape, n_iter=1e3)
    test(dolphin.dtype.uint8, shape=shape, n_iter=1e3)
    test(dolphin.dtype.uint16, shape=shape, n_iter=1e3)
    test(dolphin.dtype.uint32, shape=shape, n_iter=1e3)
    test(dolphin.dtype.int8, shape=shape, n_iter=1e3)
    test(dolphin.dtype.int16, shape=shape, n_iter=1e3)
    test(dolphin.dtype.int32, shape=shape, n_iter=1e3)

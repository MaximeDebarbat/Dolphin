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

from typing import Union, Tuple

import pycuda.driver as cuda  # pylint: disable=import-error
import tensorrt as trt  # pylint: disable=import-error

import numpy
import dolphin


class CuFill(dolphin.CuFillCompiler):
    __CU_FUNC_NAME: str = "fill_"

    def __init__(self):
        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                "P" + numpy.dtype(dtype.numpy_dtype).char + "PPII")

    def __call__(self,
                 array: 'darray',
                 value: Union[int, float, numpy.number],
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None):

        self._func[array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            array.allocation,
            value,
            array.shape_allocation,
            array.strides_allocation,
            array.ndim,
            numpy.uint32(size))


class AXpBZ(dolphin.AXpBZCompiler):
    __CU_FUNC_NAME: str = "axpbz_"

    def __init__(self):
        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                "PP" + numpy.dtype(dtype.numpy_dtype).char + numpy.dtype(
                    dtype.numpy_dtype).char + "PPII")

    def __call__(self,
                 x_array: 'darray',
                 z_array: 'darray',
                 a_scalar: numpy.number,
                 b_scalar: numpy.number,
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            x_array.allocation,
            z_array.allocation,
            x_array.dtype.numpy_dtype(a_scalar),
            x_array.dtype.numpy_dtype(b_scalar),
            x_array.shape_allocation,
            x_array.strides_allocation,
            x_array.ndim,
            numpy.uint32(size))


class AXpBYZ(dolphin.AXpBYZCompiler):
    __CU_FUNC_NAME: str = "axpbyz_"

    def __init__(self):
        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                "PPP" + numpy.dtype(dtype.numpy_dtype).char + numpy.dtype(
                    dtype.numpy_dtype).char + "PPPPPPII")

    def __call__(self,
                 x_array: 'darray',
                 y_array: 'darray',
                 z_array: 'darray',
                 a_scalar: numpy.number,
                 b_scalar: numpy.number,
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            x_array.allocation,
            y_array.allocation,
            z_array.allocation,
            x_array.dtype.numpy_dtype(a_scalar),
            x_array.dtype.numpy_dtype(b_scalar),
            x_array.shape_allocation,
            x_array.strides_allocation,
            y_array.shape_allocation,
            y_array.strides_allocation,
            z_array.shape_allocation,
            z_array.strides_allocation,
            x_array.ndim,
            numpy.uint32(size))


class EltwiseMult(dolphin.EltwiseMultCompiler):
    __CU_FUNC_NAME: str = "elt_wise_mul_"

    def __init__(self):

        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare("PPPPPPPPPII")

    def __call__(self,
                 x_array: 'darray',
                 y_array: 'darray',
                 z_array: 'darray',
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            x_array.allocation,
            y_array.allocation,
            z_array.allocation,
            x_array.shape_allocation,
            x_array.strides_allocation,
            y_array.shape_allocation,
            y_array.strides_allocation,
            z_array.shape_allocation,
            z_array.strides_allocation,
            x_array.ndim,
            numpy.uint32(size))


class EltwiseDiv(dolphin.EltwiseDivCompiler):
    __CU_FUNC_NAME: str = "elt_wise_div_"

    def __init__(self):

        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare("PPPPPPPPPIIP")

    def __call__(self,
                 x_array: 'darray',
                 y_array: 'darray',
                 z_array: 'darray',
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        cuda.memcpy_htod_async(self._error, numpy.uint8(0), stream)
        self._func[x_array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            x_array.allocation,
            y_array.allocation,
            z_array.allocation,
            x_array.shape_allocation,
            x_array.strides_allocation,
            y_array.shape_allocation,
            y_array.strides_allocation,
            z_array.shape_allocation,
            z_array.strides_allocation,
            x_array.ndim,
            numpy.uint32(size),
            self._error)

        error = numpy.zeros(1, dtype=numpy.uint8)
        cuda.memcpy_dtoh_async(error, self._error, stream)

        if error[0] == 1:
            raise ZeroDivisionError("Division by zero")

        cuda.memcpy_htod_async(self._error, numpy.uint8(0), stream)


class ScalDiv(dolphin.ScalDivCompiler):
    __CU_FUNC_NAME: str = "scal_div_"

    def __init__(self):

        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                    "PPPPPP" + numpy.dtype(dtype.numpy_dtype).char + "II")

    def __call__(self,
                 x_array: 'darray',
                 z_array: 'darray',
                 a_scalar: numpy.number,
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            x_array.allocation,
            z_array.allocation,
            x_array.shape_allocation,
            x_array.strides_allocation,
            z_array.shape_allocation,
            z_array.strides_allocation,
            x_array.dtype.numpy_dtype(a_scalar),
            x_array.ndim,
            numpy.uint32(size))


class InvScalDiv(dolphin.InvScalDivCompiler):
    __CU_FUNC_NAME: str = "invscal_div_"

    def __init__(self):

        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                    "PPPPPP" + numpy.dtype(dtype.numpy_dtype).char + "IIP")

    def __call__(self,
                 x_array: 'darray',
                 z_array: 'darray',
                 a_scalar: numpy.number,
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        cuda.memcpy_htod_async(self._error, numpy.uint8(0), stream)
        self._func[x_array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            x_array.allocation,
            z_array.allocation,
            x_array.shape_allocation,
            x_array.strides_allocation,
            z_array.shape_allocation,
            z_array.strides_allocation,
            x_array.dtype.numpy_dtype(a_scalar),
            x_array.ndim,
            numpy.uint32(size),
            self._error)

        error = numpy.empty((1,), dtype=numpy.uint8)
        error.fill(0)
        cuda.memcpy_dtoh_async(error, self._error, stream)

        if error[0] == 1:
            raise ZeroDivisionError("Division by zero")

        cuda.memcpy_htod_async(self._error, numpy.uint8(0), stream)


class EltWiseCast(dolphin.EltWiseCastCompiler):
    __CU_FUNC_NAME: str = "_to_"

    def __init__(self):

        super().__init__()

        for dtype in dolphin.dtype:
            for dtype2 in dolphin.dtype:

                self._func[dtype.cuda_dtype +
                           self.__CU_FUNC_NAME +
                           dtype2.cuda_dtype] = \
                            self.compiled_source.get_function(
                                dtype.cuda_dtype +
                                self.__CU_FUNC_NAME +
                                dtype2.cuda_dtype).prepare("PPPPPPII")

    def __call__(self,
                 x_array: 'darray',
                 z_array: 'darray',
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype + self.__CU_FUNC_NAME +
                   z_array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            x_array.allocation,
            z_array.allocation,
            x_array.shape_allocation,
            x_array.strides_allocation,
            z_array.shape_allocation,
            z_array.strides_allocation,
            x_array.ndim,
            numpy.uint32(size))


class EltwiseAbs(dolphin.EltwiseAbsCompiler):
    __CU_FUNC_NAME: str = "elt_wise_abs_"

    def __init__(self):
        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare("PPPPPPII")

    def __call__(self,
                 x_array: 'darray',
                 z_array: 'darray',
                 size: numpy.uint32,
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            x_array.allocation,
            z_array.allocation,
            x_array.shape_allocation,
            x_array.strides_allocation,
            z_array.shape_allocation,
            z_array.strides_allocation,
            x_array.ndim,
            numpy.uint32(size))


class DiscontiguousCopy(dolphin.DiscontiguousCopyCompiler):
    __CU_FUNC_NAME: str = "discontiguous_copy_"

    def __init__(self):
        super().__init__()

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = self.compiled_source.get_function(
                self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare("PPPPPPIIII")

    def __call__(self,
                 src: 'darray',
                 dst: 'darray',
                 block: Tuple[int, int, int],
                 grid: Tuple[int, int],
                 stream: cuda.Stream = None) -> None:

        self._func[src.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.shape_allocation,
            src.strides_allocation,
            dst.shape_allocation,
            dst.strides_allocation,
            src.ndim,
            dst.ndim,
            src.size,
            dst.size)


class darray:
    """
    This class implements a generic numpy style array that can be used
    with the `dolphin` library. It implements common features available
    with numpy arrays such as `astype`, `transpose`, `copy`...

    `darray` is made with the same philosophy as `numpy.ndarray`. The usability
    is really close to numpy arrays. However, `darray` is meant to be much more
    performant than `numpy.ndarray` since it is GPU accelerated.

    :param shape: Shape of the darray, defaults to None
    :type shape: Tuple[int, ...], optional
    :param dtype: dtype of the darray, defaults to None
    :type dtype: dolphin.dtype, optional
    :param stream: CUDA stream to use, defaults to None
    :type stream: cuda.Stream, optional
    :param array: numpy array to copy, defaults to None
    :type array: numpy.ndarray, optional
    :param strides: strides of the darray, defaults to None
    :type strides: Tuple[int, ...], optional
    :param allocation: CUDA allocation to use, defaults to None
    :type allocation: cuda.DeviceAllocation, optional
    :param allocation_size: Size of the allocation, defaults to None
    :type allocation_size: int, optional
    """

    _cu_axpbz = AXpBZ()
    _cu_axpbyz = AXpBYZ()
    _cu_eltwise_mult = EltwiseMult()
    _cu_eltwise_div = EltwiseDiv()
    _cu_scal_div = ScalDiv()
    _cu_invscal_div = InvScalDiv()
    _cu_eltwise_cast = EltWiseCast()
    _cu_eltwise_abs = EltwiseAbs()
    _cu_fill = CuFill()
    _cu_discontiguous_copy = DiscontiguousCopy()

    def __init__(self,
                 shape: Tuple[int, ...] = None,
                 dtype: dolphin.dtype = dolphin.dtype.float32,
                 stream: cuda.Stream = None,
                 array: numpy.ndarray = None,
                 strides: Tuple[int, ...] = None,
                 allocation: cuda.DeviceAllocation = None,
                 allocation_size: cuda.DeviceAllocation = None
                 ) -> None:

        if array is not None:
            dtype = dtype.from_numpy_dtype(array.dtype)
            shape = array.shape

        self._stream: cuda.Stream = stream
        self._dtype: dolphin.dtype = dtype
        self._shape: Tuple[int, ...] = shape
        self._size: int = trt.volume(self._shape)
        self._nbytes: int = int(self._size * self._dtype.itemsize)

        if strides is None:
            self._strides: Tuple[int, ...] = self.compute_strides(self._shape)
        else:
            self._strides: Tuple[int, ...] = strides

        if allocation_size is not None:
            self._allocation_size: int = allocation_size
        else:
            self._allocation_size: int = self._nbytes

        if allocation is not None:
            self._allocation: cuda.DeviceAllocation = allocation
        else:
            self._allocation: cuda.DeviceAllocation = cuda.mem_alloc(
                self._allocation_size)

        if array is not None:
            cuda.memcpy_htod_async(self._allocation,
                                   array.flatten(order="C"),
                                   self._stream)

        self._block, self._grid = dolphin.CudaBase.GET_BLOCK_GRID_1D(
            self._size)

        self._strides_allocation = cuda.mem_alloc(self.ndim * 4)
        self._shape_allocation = cuda.mem_alloc(self.ndim * 4)

        cuda.memcpy_htod_async(self._strides_allocation,
                               numpy.array(self._strides, dtype=numpy.uint32),
                               self._stream)
        cuda.memcpy_htod_async(self._shape_allocation,
                               numpy.array(self._shape, dtype=numpy.uint32),
                               self._stream)

    @staticmethod
    def broadcastable(shape_1: Tuple[int, ...], shape_2: Tuple[int, ...]
                      ) -> bool:
        """Checks if two shapes are broadcastable.

        :param shape_1: First shape
        :type shape_1: Tuple[int, ...]
        :param shape_2: Second shape
        :type shape_2: Tuple[int, ...]
        :return: True if the shapes are broadcastable, False otherwise
        :rtype: bool
        """
        for i in range(min(len(shape_1), len(shape_2))):
            if (shape_1[-i] != shape_2[-i] and
               shape_1[-i] != 1 and shape_2[-i] != 1):
                return False
        return True

    @staticmethod
    def compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Computes the strides of an array from the shape.
        The strides are the number of elements to skip to get to the next
        element. Also, the strides are in elements, not bytes.

        :param shape: shape of the ndarray
        :type shape: Tuple[int, ...]
        :return: Strides
        :rtype: Tuple[int, ...]
        """
        if shape:
            strides = [1]
            for s in shape[:0:-1]:
                strides.append(strides[-1] * max(1, s))
            return tuple(strides[::-1])

        return ()

    @property
    def shape_allocation(self) -> cuda.DeviceAllocation:
        """Property to access the cuda allocation of the shape.

        :return: The cuda allocation of the shape
        :rtype: cuda.DeviceAllocation
        """
        return self._shape_allocation

    @property
    def strides_allocation(self) -> cuda.DeviceAllocation:
        """Property to access the cuda allocation of the strides.

        :return: The cuda allocation of the strides
        :rtype: cuda.DeviceAllocation
        """
        return self._strides_allocation

    @property
    def ndim(self) -> numpy.uint32:
        """Computes the number of dimensions of the array.

        :return: Number of dimensions of the array
        :rtype: numpy.uint32
        """
        return len(self.shape)

    @property
    def strides(self) -> Tuple[int, ...]:
        """Property to access the strides of the array.

        :return: Strides of the array
        :rtype: Tuple[int, ...]
        """
        return self._strides

    @property
    def allocation(self) -> cuda.DeviceAllocation:
        """Property to access (Read/Write) the cuda
        allocation of the array

        :return: The cuda allocation of the array
        :rtype: cuda.DeviceAllocation
        """
        return self._allocation

    @property
    def size(self) -> numpy.uint32:
        """Property to access the size of the array.
        Size is defined as the number of elements in the array.

        :return: The size of the array, in terms of number of elements
        :rtype: numpy.uint32
        """
        return numpy.uint32(self._size)

    @property
    def dtype(self) -> dolphin.dtype:
        """Property to access the dolphin.dtype of the array

        :return: dolphin.dtype of the array
        :rtype: dolphin.dtype
        """
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Property to access the shape of the array

        :return: Shape of the array
        :rtype: Tuple[int, ...]
        """
        return self._shape

    @property
    def np(self) -> numpy.ndarray:
        """
        Alias for to_numpy()

        :return: numpy.ndarray of the darray
        :rtype: numpy.ndarray
        """

        return self.to_numpy()

    @property
    def nbytes(self) -> int:
        """Property to access the number of bytes of the array

        :return: Number of bytes of the array
        :rtype: int
        """
        return self._nbytes

    @property
    def stream(self) -> cuda.Stream:
        """Property to access (Read/Write) the cuda stream of the array

        :return: Stream used by the darray
        :rtype: cuda.Stream
        """
        return self._stream

    @property
    def T(self) -> 'darray':
        """Performs a transpose operation on the darray.
        This transpose reverse the order of the axes::

            a = darray(shape=(2, 3, 4))
            a.T.shape
            >>> (4, 3, 2)

        Also, please note that this function is not efficient as
        it performs a copy of the `darray`.

        :return: Transposed darray
        :rtype: darray
        """

        return self.transpose(*list(range(len(self._shape)))[::-1])

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

    @np.setter
    def np(self, array: numpy.ndarray) -> None:
        """Calls from_numpy() to create a darray from a numpy array.

        :param array: _description_
        :type array: numpy.ndarray
        """
        self.from_numpy(array)

    def to_numpy(self) -> numpy.ndarray:
        """Converts the darray to a numpy.ndarray.
        Note that a copy from the device to the host is performed.

        :return: numpy.ndarray of the darray
        :rtype: numpy.ndarray
        """
        res = numpy.empty((self._allocation_size//self.dtype.itemsize,),
                          dtype=self._dtype.numpy_dtype)

        cuda.memcpy_dtoh_async(res,
                               self._allocation,
                               self._stream)

        temp_s = tuple([s*self._dtype.itemsize for s in self._strides])
        return numpy.lib.stride_tricks.as_strided(
            res,
            shape=self._shape,
            strides=temp_s)

    def from_numpy(self, array: numpy.ndarray) -> None:
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

        :param dtype: dtype to convert to
        :type dtype: dolphin.dtype
        :param dst: darray to write the result of the operation,
          defaults to None
        :type dst: darray, optional
        :raises ValueError: In case the dst shape or dtype doesn't match
        :return: darray with the new dtype
        :rtype: darray
        """

        if dst is not None and self._shape != dst.shape:
            raise ValueError(
                f"dst shape doesn't match darray : \
{self.shape} != {dst.shape}")

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

    def transpose(self, *axes: int) -> 'darray':
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

        if len(set(axes)) != len(axes):
            raise ValueError("repeated axis in transpose")

        strides = self._strides
        new_shape = [self.shape[i] for i in axes]
        new_strides = [strides[i] for i in axes]

        res = self.__class__(shape=tuple(new_shape),
                             dtype=self._dtype,
                             stream=self._stream,
                             strides=new_strides,
                             allocation=self._allocation)

        return res

    def copy(self) -> 'darray':
        """Returns a copy of the current darray.
        Note that a copy from device to device is performed.

        :return: Copy of the array with another cuda allocation
        :rtype: darray
        """
        res = self.__class__(shape=self._shape,
                             dtype=self._dtype,
                             strides=self._strides,
                             allocation_size=self._allocation_size,
                             stream=self._stream)

        cuda.memcpy_dtod_async(res.allocation,
                               self._allocation,
                               self._allocation_size,
                               stream=self._stream)

        return res

    def __getitem__(self, index: Union[int, slice, tuple]) -> 'darray':
        """Returns a view of the darray with the given index.

        :param index: Index to use
        :type index: Union[int, slice, tuple]
        :raises IndexError: Too many indexes. The number of indexes
            must be less than the number of dimensions of the array.
        :raises IndexError: Axes must be in the range of the number
            of dimensions of the array.
        :return: View of the darray
        :rtype: darray
        """
        if not isinstance(index, tuple):
            index = (index,)

        new_shape = []
        new_offset = 0
        new_strides = []

        seen_ellipsis = False

        index_axis = 0
        array_axis = 0
        while index_axis < len(index):
            index_entry = index[index_axis]

            if array_axis > len(self.shape):
                raise IndexError("too many axes in index")

            if isinstance(index_entry, slice):
                start, stop, idx_stride = index_entry.indices(
                    self.shape[array_axis])

                array_stride = self.strides[array_axis]

                new_shape.append((abs(stop - start) - 1) //
                                 abs(idx_stride) + 1)
                new_strides.append(idx_stride * array_stride)
                new_offset += array_stride * start

                index_axis += 1
                array_axis += 1

            elif isinstance(index_entry, (int, numpy.integer)):
                array_shape = self.shape[array_axis]
                if index_entry < 0:
                    index_entry += array_shape

                if not (0 <= index_entry < array_shape):
                    raise IndexError(f"subindex in axis {index_axis} \
out of range")

                new_offset += self.strides[array_axis] * index_entry

                index_axis += 1
                array_axis += 1

            elif index_entry is Ellipsis:
                index_axis += 1

                remaining_index_count = len(index) - index_axis
                new_array_axis = len(self.shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(self.shape[array_axis])
                    new_strides.append(self.strides[array_axis])
                    array_axis += 1

                if seen_ellipsis:
                    raise IndexError("more than one ellipsis \
not allowed in index")
                seen_ellipsis = True

            elif index_entry is numpy.newaxis:
                new_shape.append(1)
                new_strides.append(0)
                index_axis += 1

            else:
                raise IndexError(f"invalid subindex in axis {index_axis}")

        while array_axis < len(self.shape):
            new_shape.append(self.shape[array_axis])
            new_strides.append(self.strides[array_axis])

            array_axis += 1

        return darray(
            shape=tuple(new_shape),
            dtype=self.dtype,
            strides=tuple(new_strides),
            allocation=int(self.allocation) + new_offset * self.dtype.itemsize,
            allocation_size=self._allocation_size - new_offset
        )

    def __setitem__(self, index: Union[int, slice, tuple],
                    other: Union[int, float, numpy.number, 'darray']) -> None:
        """Sets the value of the darray with the given index.

        :param index: Index to use
        :type index: Union[int, slice, tuple]
        :param other: Value to set
        :type other: Union[int, float, numpy.number, darray]
        :return: View of the darray
        :rtype: darray
        """

        if isinstance(other, (numpy.number, int, float)):
            print("other is a number")
            self[index].fill(other)
        elif isinstance(other, darray):
            if not self.broadcastable(self.shape, other.shape):
                raise ValueError(f"operands could not be broadcast together \
with shapes {self.shape}, {other.shape}")
            self._cu_discontiguous_copy(src=other,
                                        dst=self[index],
                                        block=self._block,
                                        grid=self._grid,
                                        stream=self._stream)

    def __str__(self) -> str:
        """Returns the string representation of the numpy array.
        Note that a copy from the device to the host is performed.

        :return: String representation of the numpy array
        :rtype: str
        """
        return self.to_numpy().__str__()  # pylint: disable=E1120

    def __repr__(self) -> str:
        """Returns the representation of the numpy array.
        Note that a copy from the device to the host is performed.

        :return: Representation of the numpy array
        :rtype: str
        """
        return self.to_numpy().__repr__()  # pylint: disable=E1120

    def flatten(self, dst: 'darray' = None) -> 'darray':
        """Returns a flattened view of the darray. Useful for
        reordering the array in memory.

        :param dst: Destination darray
        :type dst: darray
        :return: Flattened view of the darray
        :rtype: darray
        """
        if dst is None:
            dst = darray(shape=(self.size,),
                         dtype=self.dtype,
                         allocation=self._allocation,
                         allocation_size=self._allocation_size)

        self._cu_discontiguous_copy(src=self,
                                    dst=dst,
                                    block=self._block,
                                    grid=self._grid,
                                    stream=self._stream)

        return dst

    def fill(self, value: Union[int, float, numpy.number]) -> 'darray':
        """
        Fills the darray with the value of value.

        :param value: Value to fill the array with
        :type value: Union[int, float, numpy.number]
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

    def add(self, other: Union[int, float, numpy.number, 'darray'],
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
        :type other: Union[int, float, numpy.number, 'darray']
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

    def __add__(self, other: Union[int, float, numpy.number, 'darray']
                ) -> 'darray':
        """Non-Efficient addition of a darray with another object.
        It is not efficient because it implies a copy of the array
        where the result is written. cuda.memalloc is really time
        consuming (up to 95% of the total latency is spent in
        cuda.memalloc only)

        :param other: scalar or darray to add
        :type other: Union[int, float, numpy.number, 'darray']
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """
        return self.add(other)

    __radd__ = __add__

    def __iadd__(self, other: Union[int, float, numpy.number, 'darray']
                 ) -> 'darray':
        """Implements += operator. As __add__, this method is not
        efficient because it implies a copy of the array where the
        usage of cuda.memalloc which is really time consuming
        (up to 95% of the total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to add
        :type other: Union[int, float, numpy.number, 'darray']
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """

        return self.add(other, self)

    def substract(self, other: Union[int, float, numpy.number, 'darray'],
                  dst: 'darray' = None) -> 'darray':
        """Efficient substraction of a darray with another object.

        :param other: scalar or darray to substract
        :type other: Union[int, float, numpy.number, 'darray']
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

    def __sub__(self, other: Union[int, float, numpy.number, 'darray']
                ) -> 'darray':  # array - object
        """Non-Efficient substraction of a darray with another object.
        It is not efficient because it implies a copy of the array
        where the result is written. cuda.memalloc is really time
        consuming (up to 95% of the total latency is spent in
        cuda.memalloc only)

        :param other: scalar or darray to substract
        :type other: Union[int, float, numpy.number, 'darray']
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """

        return self.substract(other)

    def reversed_substract(self,
                           other: Union[int, float, numpy.number, 'darray'],
                           dst: 'darray' = None) -> 'darray':
        """Efficient reverse substraction of an object with darray.
        It is efficient if dst is provided because it does not
        invoke cuda.memalloc.
        If dst is not provided, normal __rsub__ is called.

        :param other: scalar or darray to substract
        :type other: Union[int, float, numpy.number, 'darray']
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

    def __rsub__(self, other: Union[int, float, numpy.number, 'darray']
                 ) -> 'darray':  # object - array
        """Non-Efficient substraction of another object with darray.
        It is not efficient because it implies a copy of the array
        where the result is written. cuda.memalloc is really time
        consuming (up to 95% of the total latency is spent in
        cuda.memalloc only)

        :param other: scalar or darray to substract
        :type other: Union[int, float, numpy.number, 'darray']
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """
        return self.reversed_substract(other)

    def __isub__(self, other: Union[int, float, numpy.number, 'darray']
                 ) -> 'darray':  # array -= object
        """Non-Efficient -= operation.
        It is not efficient because it implies a copy of the array
        where the result is written. cuda.memalloc is really time
        consuming (up to 95% of the total latency is spent in
        cuda.memalloc only)

        :param other: scalar or darray to substract
        :type other: Union[int, float, numpy.number, 'darray']
        :raises ValueError: If other is not a scalar or a darray
        :return: A copy of the darray where the result is written
        :rtype: darray
        """
        return self.substract(other, dst=self)

    def multiply(self, other: Union[int, float, numpy.number, 'darray'],
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
        :type other: Union[int, float, numpy.number, 'darray']
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

    def __mul__(self, other: Union[int, float, numpy.number, 'darray']
                ) -> 'darray':  # array * object
        """Non-Efficient multiplication of a darray with another object.
        This multiplication is element-wise multiplication, not matrix
        multiplication. For matrix multiplication please refer to matmul.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to multiply
        :type other: Union[int, float, numpy.number, 'darray']
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

    def __imul__(self, other: Union[int, float, numpy.number, 'darray']
                 ) -> 'darray':
        """Non-Efficient multiplication of a darray with another object.
        This multiplication is element-wise multiplication, not matrix
        multiplication. For matrix multiplication please refer to matmul.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to multiply
        :type other: Union[int, float, numpy.number, 'darray']
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """
        return self.multiply(other, dst=self)

    def divide(self, other: Union[int, float, numpy.number, 'darray'],
               dst: 'darray' = None) -> 'darray':
        """Efficient division of a darray with another object.
        Can be a darray or a scalar. If dst is None, normal __div__
        is called.
        This method is much more efficient than the __div__ method
        because __div__ implies a copy of the array.
        cuda.memalloc is really time consuming (up to 95% of the total
        latency is spent in cuda.memalloc only).

        :param other: scalar or darray to divide by
        :type other: Union[int, float, numpy.number, 'darray']
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

    def __div__(self, other: Union[int, float, numpy.number, 'darray']
                ) -> 'darray':  # array / object
        """Non-Efficient division of a darray with another object.
        This division is element-wise.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to divide by
        :type other: Union[int, float, numpy.number, 'darray']
        :raises ValueError: If other is not a scalar or a darray
        :return: The darray where the result is written
        :rtype: darray
        """
        return self.divide(other)

    def reversed_divide(self, other: Union[int, float, numpy.number, 'darray'],
                        dst: 'darray' = None) -> 'darray':
        """Efficient division of a darray with another object.

        Can be a darray or a scalar. If dst is None, normal __rdiv__
        is called.
        This method is much more efficient than the __rdiv__ method
        because __rdiv__ implies a copy of the array.
        cuda.memalloc is really time consuming (up to 95% of the total
        latency is spent in cuda.memalloc only).

        :param other: scalar or darray to divide by
        :type other: Union[int, float, numpy.number, 'darray']
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

    def __rdiv__(self, other: Union[int, float, numpy.number, 'darray']
                 ) -> 'darray':  # object / array
        """Non-Efficient reverse division of an object by darray.
        This division is element-wise.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to divide by
        :type other: Union[int, float, numpy.number, 'darray']
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

    def __idiv__(self, other: Union[int, float, numpy.number, 'darray']
                 ) -> 'darray':
        """Non-Efficient division of a darray with another object.
        This division is element-wise.
        This operation is not efficient because it implies a copy of the array
        using cuda.memalloc. This is really time consuming (up to 95% of the
        total latency is spent in cuda.memalloc only)

        :param other: scalar or darray to divide by
        :type other: Union[int, float, numpy.number, 'darray']
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


def transpose(src: darray,
              axes: Tuple[int, ...]) -> darray:
    """Returns a darray with the axes transposed.

    :param axes: Axes to transpose
    :type axes: Tuple[int, ...]
    :param src: darray to transpose
    :type src: darray
    :return: Transposed darray
    :rtype: darray
    """
    return src.transpose(*axes)


def multiply(src: darray,
             other: Union[int, float, numpy.number, 'darray'],
             dst: darray = None) -> darray:
    """Returns the multiplication of two darrays.
    It works that way::

        result = src * other

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: Union[int, float, numpy.number, 'darray']
    :return: Multiplication of the two darrays
    :rtype: darray
    """
    return src.multiply(other, dst)


def add(src: darray,
        other: Union[int, float, numpy.number, 'darray'],
        dst: darray = None) -> darray:
    """Returns the addition of two darrays.
    It works that way::

        result = src + other

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: Union[int, float, numpy.number, 'darray']
    :return: Addition of the two darrays
    :rtype: darray
    """
    return src.add(other, dst)


def substract(src: darray,
              other: Union[int, float, numpy.number, 'darray'],
              dst: darray = None) -> darray:
    """Returns the substraction of two darrays.
    It works that way::

        result = src - other

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: Union[int, float, numpy.number, 'darray']
    :return: Substraction of the two darrays
    :rtype: darray
    """
    return src.substract(other, dst)


def divide(src: darray,
           other: Union[int, float, numpy.number, 'darray'],
           dst: darray = None) -> darray:
    """Returns the division of a darray by an object.
    It works that way::

        result = src / other

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: Union[int, float, numpy.number, 'darray']
    :return: Division of the two darrays
    :rtype: darray
    """
    return src.divide(other, dst)


def reversed_divide(
        src: darray,
        other: Union[int, float, numpy.number, 'darray'],
        dst: darray = None) -> darray:
    """Returns the division of a darray and an object.
    It works that way::

        result = other / src

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: Union[int, float, numpy.number, 'darray']
    :return: Division of the two darrays
    :rtype: darray
    """
    return src.reversed_divide(other, dst)


def reversed_substract(
        src: darray,
        other: Union[int, float, numpy.number, 'darray'],
        dst: darray = None) -> darray:
    """Returns the substraction of a darray and an object.
    It works that way::

        result = other - src

    :param src: First darray
    :type src: darray
    :param other: Second darray or scalar
    :type other: Union[int, float, numpy.number, 'darray']
    :return: Substraction of the two darrays
    :rtype: darray
    """
    return src.reversed_substract(other, dst)


def zeros(
        shape: Tuple[int, ...],
        dtype: dolphin.dtype = dolphin.dtype.float32) -> darray:
    """Returns a darray for a given shape and dtype filled with zeros.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param shape: Shape of the array
    :type shape: Tuple[int, ...]
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with zeros
    :rtype: darray
    """

    return darray(array=numpy.zeros(shape, dtype=dtype.numpy_dtype))


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


def ones(shape: Tuple[int, ...], dtype: dolphin.dtype = dolphin.dtype.float32
         ) -> darray:
    """Returns a darray for a given shape and dtype filled with ones.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param shape: Shape of the array
    :type shape: Tuple[int, ...]
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with ones
    :rtype: darray
    """

    return darray(array=numpy.ones(shape, dtype=dtype.numpy_dtype))


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
        shape: Tuple[int, ...],
        dtype: dolphin.dtype = dolphin.dtype.float32) -> darray:
    """Returns a darray of a given shape and dtype without
    initializing entries.

    This function is a creation function, thus, it does not take an optional
    destination `darray` as argument.

    :param shape: Shape of the array
    :type shape: Tuple[int, ...]
    :param dtype: Type of the array
    :type dtype: dolphin.dtype
    :return: darray filled with random values
    :rtype: darray
    """

    return darray(shape=shape, dtype=dtype)


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


def from_numpy(array: numpy.ndarray) -> darray:
    """Returns a darray from a numpy array.

    :param array: numpy array to convert
    :type array: numpy.ndarray
    :return: _description_
    :rtype: darray
    """

    return darray(array=array)

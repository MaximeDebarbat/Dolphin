
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

    def transpose(self, *axes: int) -> 'darray':
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
        new_shape = numpy.array([self.shape[i] for i in axes],
                                dtype=numpy.uint32)
        new_strides = numpy.array([strides[i] for i in axes],
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

    def __add__(self, other: object) -> 'darray':

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
    __iadd__ = __add__  # array += object

    def __sub__(self, other: object) -> 'darray':  # array - object

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

    def __rsub__(self, other: object) -> 'darray':  # object - array
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

    __isub__ = __sub__  # array -= object

    def __mul__(self, other: object) -> 'darray':  # array * object
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
    __imul__ = __mul__  # array *= object

    def __div__(self, other: object) -> 'darray':  # array / object
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

    def __rdiv__(self, other: object) -> 'darray':  # object / array
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

    __idiv__ = __div__  # array /= object

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


def test(dolphin_dtype: dolphin.dtype = dolphin.dtype.float32,
         shape: tuple = (1000, 1000)):

    print(f"\nRunning tests for {dolphin_dtype}...\n")

    N_ITER = int(1e3)

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

    #######################
    #    ADD ARRA TEST    #
    #######################

    # array + array
    dummy = dummy + dummy
    cuda_array = cuda_array + cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add scalar test 1 failed"

    print(f"Addarra  test 1 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # scal + array
    dummy = dummy + dummy
    cuda_array = cuda_array + cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add scalar test 2 failed"

    print(f"Addarra  test 2 : {diff} | first number : \
{cuda_array.ndarray[0, 0]} == {dummy[0, 0]}")

    # array += scal
    dummy += dummy
    cuda_array += cuda_array
    diff = numpy.linalg.norm(cuda_array.ndarray - dummy)

    assert diff < 1e-5, "Add scalar test 3 failed"

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

    print(f"numpy time : {numpy_time}")

    t1 = time.time()
    for _ in range(N_ITER):
        cuda_res = 15*cuda_array.transpose(1, 0)/2 + 5*cuda_array.transpose(1, 0)/3
    cuda_time = 1000*(time.time() - t1)/N_ITER

    print(f"cuda time : {cuda_time}")


if __name__ == "__main__":

    shape = (3000, 1000)

    test(dolphin.dtype.float32, shape=shape)
    test(dolphin.dtype.float64, shape=shape)
    test(dolphin.dtype.uint8, shape=shape)
    test(dolphin.dtype.uint16, shape=shape)
    test(dolphin.dtype.uint32, shape=shape)
    test(dolphin.dtype.int8, shape=shape)
    test(dolphin.dtype.int16, shape=shape)
    test(dolphin.dtype.int32, shape=shape)

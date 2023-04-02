"""_summary_
"""

import os
import math
from time import time

import numpy

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error
from jinja2 import Template
import dolphin


class AXpBZ(dolphin.CudaBase):
    __CU_FILENAME: str = "axpbz.cu"
    __CU_FUNC_NAME: str = "axpbz_"

    def __init__(self):

        super(AXpBZ, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = SourceModule(
                Template(self._cuda_source).render(
                    dtype=dtype.cuda_dtype)).get_function(
                        self.__CU_FUNC_NAME + dtype.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 z_array: dolphin.darray,
                 a_scalar: numpy.number,
                 b_scalar: numpy.number,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype](
                    x_array.allocation,
                    z_array.allocation,
                    x_array.dtype.numpy_dtype(a_scalar),
                    x_array.dtype.numpy_dtype(b_scalar),
                    numpy.uint32(size),
                    block=block,
                    grid=grid,
                    stream=stream)


class AXpBYZ(dolphin.CudaBase):
    __CU_FILENAME: str = "axpbyz.cu"
    __CU_FUNC_NAME: str = "axpbyz_"

    def __init__(self):

        super(AXpBYZ, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        for dtype in dolphin.dtype:
            s = Template(self._cuda_source).render(
                    dtype=dtype.cuda_dtype)
            self._func[dtype.cuda_dtype] = SourceModule(s).get_function(
                        self.__CU_FUNC_NAME + dtype.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 y_array: dolphin.darray,
                 z_array: dolphin.darray,
                 a_scalar: numpy.number,
                 b_scalar: numpy.number,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype](
                    x_array.allocation,
                    y_array.allocation,
                    z_array.allocation,
                    x_array.dtype.numpy_dtype(a_scalar),
                    x_array.dtype.numpy_dtype(b_scalar),
                    numpy.uint32(size),
                    block=block,
                    grid=grid,
                    stream=stream)


class EltwiseMult(dolphin.CudaBase):
    __CU_FILENAME: str = "elt_wise_mul.cu"
    __CU_FUNC_NAME: str = "elt_wise_mul_"

    def __init__(self):

        super(EltwiseMult, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = SourceModule(
                Template(self._cuda_source).render(
                    dtype=dtype.cuda_dtype)).get_function(
                        self.__CU_FUNC_NAME + dtype.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 y_array: dolphin.darray,
                 z_array: dolphin.darray,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype](
                    x_array.allocation,
                    y_array.allocation,
                    z_array.allocation,
                    numpy.uint32(size),
                    block=block,
                    grid=grid,
                    stream=stream)


class EltwiseDiv(dolphin.CudaBase):
    __CU_FILENAME: str = "elt_wise_div.cu"
    __CU_FUNC_NAME: str = "elt_wise_div_"

    def __init__(self):

        super(EltwiseDiv, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        self._error = cuda.mem_alloc(numpy.dtype(numpy.uint8).itemsize)

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = SourceModule(
                Template(self._cuda_source).render(
                    dtype=dtype.cuda_dtype)).get_function(
                        self.__CU_FUNC_NAME + dtype.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 y_array: dolphin.darray,
                 z_array: dolphin.darray,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        cuda.memcpy_htod(self._error, numpy.uint8(0))

        self._func[x_array.dtype.cuda_dtype](
                    x_array.allocation,
                    y_array.allocation,
                    z_array.allocation,
                    numpy.uint32(size),
                    self._error,
                    block=block,
                    grid=grid,
                    stream=stream)

        error = numpy.zeros(1, dtype=numpy.uint8)
        if stream is None:
            cuda.memcpy_dtoh(error, self._error)
        else:
            cuda.memcpy_dtoh_async(error, self._error, stream)

        if error[0] == 1:
            raise ZeroDivisionError("Division by zero")

        if stream is None:
            cuda.memcpy_htod(self._error, numpy.uint8(0))
        else:
            cuda.memcpy_htod_async(self._error, numpy.uint8(0), stream)


class ScalDiv(dolphin.CudaBase):
    __CU_FILENAME: str = "scal_div.cu"
    __CU_FUNC_NAME: str = "scal_div_"

    def __init__(self):

        super(ScalDiv, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = SourceModule(
                Template(self._cuda_source).render(
                    dtype=dtype.cuda_dtype)).get_function(
                        self.__CU_FUNC_NAME + dtype.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 z_array: dolphin.darray,
                 a_scalar: numpy.number,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype](
                    x_array.allocation,
                    z_array.allocation,
                    x_array.dtype.numpy_dtype(a_scalar),
                    numpy.uint32(size),
                    block=block,
                    grid=grid,
                    stream=stream)


class InvScalDiv(dolphin.CudaBase):
    __CU_FILENAME: str = "invscal_div.cu"
    __CU_FUNC_NAME: str = "invscal_div_"

    def __init__(self):

        super(InvScalDiv, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        self._error = cuda.mem_alloc(numpy.dtype(numpy.uint8).itemsize)

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = SourceModule(
                Template(self._cuda_source).render(
                    dtype=dtype.cuda_dtype)).get_function(
                        self.__CU_FUNC_NAME + dtype.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 z_array: dolphin.darray,
                 a_scalar: numpy.number,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        cuda.memcpy_htod(self._error, numpy.uint8(0))

        self._func[x_array.dtype.cuda_dtype](
                    x_array.allocation,
                    z_array.allocation,
                    x_array.dtype.numpy_dtype(a_scalar),
                    numpy.uint32(size),
                    self._error,
                    block=block,
                    grid=grid,
                    stream=stream)

        error = numpy.zeros(1, dtype=numpy.uint8)
        if stream is None:
            cuda.memcpy_dtoh(error, self._error)
        else:
            cuda.memcpy_dtoh_async(error, self._error, stream)

        if error[0] == 1:
            raise ZeroDivisionError("Division by zero")

        if stream is None:
            cuda.memcpy_htod(self._error, numpy.uint8(0))
        else:
            cuda.memcpy_htod_async(self._error, numpy.uint8(0), stream)


class EltWiseCast(dolphin.CudaBase):
    __CU_FILENAME: str = "elt_wise_cast.cu"
    __CU_FUNC_NAME: str = "_to_"

    def __init__(self):

        super(EltWiseCast, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        for dtype in dolphin.dtype:
            for dtype2 in dolphin.dtype:

                s = Template(self._cuda_source).render(
                        indtype=dtype.cuda_dtype,
                        outdtype=dtype2.cuda_dtype)

                self._func[dtype.cuda_dtype + self.__CU_FUNC_NAME + dtype2.cuda_dtype] = SourceModule(s).get_function(
                            dtype.cuda_dtype + self.__CU_FUNC_NAME + dtype2.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 z_array: dolphin.darray,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype + self.__CU_FUNC_NAME + z_array.dtype.cuda_dtype](
                    x_array.allocation,
                    z_array.allocation,
                    numpy.uint32(size),
                    block=block,
                    grid=grid,
                    stream=stream)


class EltwiseAbs(dolphin.CudaBase):
    __CU_FILENAME: str = "elt_wise_abs.cu"
    __CU_FUNC_NAME: str = "elt_wise_abs_"

    def __init__(self):

        super(EltwiseAbs, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = SourceModule(
                Template(self._cuda_source).render(
                    dtype=dtype.cuda_dtype)).get_function(
                        self.__CU_FUNC_NAME + dtype.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype](
                    x_array.allocation,
                    numpy.uint32(size),
                    block=block,
                    grid=grid,
                    stream=stream)


class Transpose(dolphin.CudaBase):
    __CU_FILENAME: str = "transpose.cu"
    __CU_FUNC_NAME: str = "transpose_"

    def __init__(self):
        super(Transpose, self).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        for dtype in dolphin.dtype:
            self._func[dtype.cuda_dtype] = SourceModule(
                Template(self._cuda_source).render(
                    dtype=dtype.cuda_dtype)).get_function(
                        self.__CU_FUNC_NAME + dtype.cuda_dtype)

    def __call__(self,
                 x_array: dolphin.darray,
                 y_array: dolphin.darray,
                 shape: cuda.DeviceAllocation,
                 strides: cuda.DeviceAllocation,
                 ndim: numpy.uint32,
                 size: numpy.uint32,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        self._func[x_array.dtype.cuda_dtype](
                    x_array.allocation,
                    y_array.allocation,
                    shape,
                    strides,
                    numpy.uint32(ndim),
                    numpy.uint32(size),
                    block=block,
                    grid=grid,
                    stream=stream)


CU_AXPBZ = AXpBZ()
CU_AXPBYZ = AXpBYZ()
CU_ELTWISE_MULT = EltwiseMult()
CU_ELTWISE_DIV = EltwiseDiv()
CU_INVSCAL_DIV = InvScalDiv()
CU_SCAL_DIV = ScalDiv()
CU_ELTWISE_CAST = EltWiseCast()
CU_ELTWISE_ABS = EltwiseAbs()
CU_TRANSPOSE = Transpose()

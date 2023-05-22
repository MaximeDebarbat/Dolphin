"""
Module that contains the CUDA kernels for darray operations.
Each class compiles the kernel for a specific operation with
specific data types.
"""

import numpy

import pycuda.driver as cuda  # pylint: disable=import-error
from jinja2 import Template

import dolphin
from .compiler_base import CompilerBase


class CuFillCompiler(CompilerBase):
    __CU_FILENAME: str = "fill.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source


class AXpBZCompiler(CompilerBase):
    __CU_FILENAME: str = "axpbz.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source


class AXpBYZCompiler(CompilerBase):
    __CU_FILENAME: str = "axpbyz.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source


class EltwiseMultCompiler(CompilerBase):
    __CU_FILENAME: str = "elt_wise_mul.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source


class EltwiseDivCompiler(CompilerBase):
    __CU_FILENAME: str = "elt_wise_div.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self._error = cuda.mem_alloc(numpy.dtype(numpy.uint8).itemsize)
        cuda.memcpy_htod(self._error, numpy.uint8(0))

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source


class ScalDivCompiler(CompilerBase):
    __CU_FILENAME: str = "scal_div.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source


class InvScalDivCompiler(CompilerBase):
    __CU_FILENAME: str = "invscal_div.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self._error = cuda.mem_alloc(numpy.dtype(numpy.uint8).itemsize)
        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source


class EltWiseCastCompiler(CompilerBase):
    __CU_FILENAME: str = "elt_wise_cast.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            for dtype2 in dolphin.dtype:

                source += Template(self._cuda_source).render(
                    indtype=dtype.cuda_dtype,
                    outdtype=dtype2.cuda_dtype)

        source = self.append_utils(source)
        return source


class EltwiseAbsCompiler(CompilerBase):
    __CU_FILENAME: str = "elt_wise_abs.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self._error = cuda.mem_alloc(numpy.dtype(numpy.uint8).itemsize)
        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source


class DiscontiguousCopyCompiler(CompilerBase):
    __CU_FILENAME: str = "discontiguous_copy.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self._error = cuda.mem_alloc(numpy.dtype(numpy.uint8).itemsize)
        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        source = self.append_utils(source)
        return source

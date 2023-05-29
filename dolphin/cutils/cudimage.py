"""
Module that contains the CUDA kernels for dimage operations.
Each class compiles the kernel for a specific operation with
specific data types.
"""

from jinja2 import Template

import dolphin
from .compiler_base import CompilerBase


class CuResizeCompiler(CompilerBase):
    __CU_FILENAME: str = "resize.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype,
                )
        return source


class CuNormalizeCompiler(CompilerBase):
    __CU_FILENAME: str = "normalize.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype_in in dolphin.dtype:
            for dtype_out in dolphin.dtype:
                source += Template(self._cuda_source).render(
                    intype=dtype_in.cuda_dtype,
                    outtype=dtype_out.cuda_dtype,
                    )
        return source


class CuCvtColorCompiler(CompilerBase):
    __CU_FILENAME: str = "cvt_color.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.compiled_source = self.try_load_cubin()

    def generate_source(self):
        source: str = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype
                )

        return source

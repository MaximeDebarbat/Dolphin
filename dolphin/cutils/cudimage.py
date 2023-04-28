"""
Module that contains the CUDA kernels for dimage operations.
Each class compiles the kernel for a specific operation with
specific data types.
"""

from pycuda.compiler import SourceModule  # pylint: disable=import-error
from jinja2 import Template

import dolphin
from .compiler_base import CompilerBase


class CuResizeCompiler(CompilerBase):
    __CU_FILENAME: str = "resize.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.source: str = ""
        for dtype in dolphin.dtype:
            self.source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype,
                )
        self.compiled_source = SourceModule(self.source)


class CuNormalizeCompiler(CompilerBase):
    __CU_FILENAME: str = "normalize.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)

        self.source: str = ""
        for dtype_in in dolphin.dtype:
            for dtype_out in dolphin.dtype:
                self.source += Template(self._cuda_source).render(
                    intype=dtype_in.cuda_dtype,
                    outtype=dtype_out.cuda_dtype,
                    )
        self.compiled_source = SourceModule(self.source)


class CuCvtColorCompiler(CompilerBase):
    __CU_FILENAME: str = "cvt_color.cu"

    def __init__(self):
        super().__init__(self.__CU_FILENAME)
        self.source = ""
        for dtype in dolphin.dtype:
            self.source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype
                )
        self.compiled_source = SourceModule(self.source)

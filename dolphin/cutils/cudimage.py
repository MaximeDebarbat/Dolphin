"""_summary_
"""

import os
import math
from time import time
from typing import Union

import numpy

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error
from jinja2 import Template
import dolphin


class CuResizeNearest(dolphin.CudaBase):
    __CU_FILENAME: str = "resize_nearest.cu"
    __CU_FUNC_NAME: str = "_resize_nearest_"

    def __init__(self):
        super(CuResizeNearest).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        source = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        compiled_source = SourceModule(source)

        for mode in ["CHW", "HWC"]:
            for dtype in dolphin.dtype:
                self._func[mode+dtype.cuda_dtype] = compiled_source.get_function(mode+self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                    "PPHHHHB")

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        if input.image_dim_format.value == dolphin.dimage_dim_format.DOLPHIN_CHW.value:
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode+input.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            input.allocation,
            output.allocation,
            input.width,
            input.height,
            output.width,
            output.height,
            input.channel
            )


class CuResizePadding(dolphin.CudaBase):
    __CU_FILENAME: str = "resize_padding.cu"
    __CU_FUNC_NAME: str = "_resize_padding_"

    def __init__(self):
        super(CuResizePadding).__init__()
        self._cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CU_FILENAME), "rt", encoding="utf-8").read()
        self._func = {}

        source = ""
        for dtype in dolphin.dtype:
            source += Template(self._cuda_source).render(
                dtype=dtype.cuda_dtype)

        compiled_source = SourceModule(source)

        for mode in ["CHW", "HWC"]:
            for dtype in dolphin.dtype:
                self._func[mode+dtype.cuda_dtype] = compiled_source.get_function(mode+self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                    "PPHHHHB"+numpy.dtype(dtype.numpy_dtype).char)

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 padding: Union[float, int],
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        if input.image_dim_format.value == dolphin.dimage_dim_format.DOLPHIN_CHW.value:
            mode = "CHW"
        else:
            mode = "HWC"

        # print(f"mode : {mode}, input.shape : {input.shape}, output.shape : {output.shape}, padding : {padding}")
        # exit(0)
        self._func[mode+input.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            input.allocation,
            output.allocation,
            input.width,
            input.height,
            output.width,
            output.height,
            input.channel,
            input.dtype.numpy_dtype(padding)
            )

CU_RESIZE_LINEAR = CuResizeNearest()
CU_RESIZE_PADDING = CuResizePadding()
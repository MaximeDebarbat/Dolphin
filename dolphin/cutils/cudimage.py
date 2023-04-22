"""_summary_
"""

import os
from typing import Union

import numpy

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error
from jinja2 import Template
import dolphin


class CuResizeCompiler(dolphin.CudaBase):
    __CU_FILENAME: str = "resize.cu"
    cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            __CU_FILENAME), "rt", encoding="utf-8").read()
    source: str = ""
    for dtype in dolphin.dtype:
        source += Template(cuda_source).render(
            dtype=dtype.cuda_dtype,
            )
    compiled_source = SourceModule(source)


class CuResizeNearest(CuResizeCompiler):
    __CU_FUNC_NAME: str = "_resize_nearest_"

    def __init__(self):
        super(CuResizeNearest).__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype in dolphin.dtype:
                self._func[mode+dtype.cuda_dtype] = self.compiled_source.get_function(mode+self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
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


class CuResizePadding(CuResizeCompiler):
    __CU_FUNC_NAME: str = "_resize_padding_"

    def __init__(self):
        super(CuResizePadding).__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype in dolphin.dtype:
                self._func[mode+dtype.cuda_dtype] = self.compiled_source.get_function(mode+self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
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


class CuNormalizeCompiler(dolphin.CudaBase):
    __CU_FILENAME: str = "normalize.cu"
    cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            __CU_FILENAME), "rt", encoding="utf-8").read()
    source: str = ""
    for dtype_in in dolphin.dtype:
        for dtype_out in dolphin.dtype:
            source += Template(cuda_source).render(
                intype=dtype_in.cuda_dtype,
                outtype=dtype_out.cuda_dtype,
                )
    compiled_source = SourceModule(source)


class CuNormalizeMeanStd(CuNormalizeCompiler):
    __CU_FUNC_NAME: str = "_normalize_mean_std_"

    def __init__(self):
        super(CuNormalizeMeanStd).__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                for dtype_out in dolphin.dtype:
                    self._func[mode+dtype_in.cuda_dtype+dtype_out.cuda_dtype] = self.compiled_source.get_function(mode+self.__CU_FUNC_NAME + dtype_in.cuda_dtype + "_" + dtype_out.cuda_dtype).prepare(
                        "PPHHBPP")

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 mean: Union[float, int],
                 std: Union[float, int],
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        if input.image_dim_format.value == dolphin.dimage_dim_format.DOLPHIN_CHW.value:
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode+input.dtype.cuda_dtype+output.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            input.allocation,
            output.allocation,
            input.width,
            input.height,
            input.channel,
            mean,
            std
            )


class CuNormalize255(CuNormalizeCompiler):
    __CU_FUNC_NAME: str = "normalize_255_"

    def __init__(self):
        super(CuNormalize255).__init__()

        self._func: dict = {}

        for dtype_in in dolphin.dtype:
            for dtype_out in dolphin.dtype:
                self._func[dtype_in.cuda_dtype+dtype_out.cuda_dtype] = self.compiled_source.get_function(self.__CU_FUNC_NAME + dtype_in.cuda_dtype + "_" + dtype_out.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        self._func[input.dtype.cuda_dtype+output.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            input.allocation,
            output.allocation,
            input.width,
            input.height,
            input.channel
            )


class CuNormalizeTF(CuNormalizeCompiler):
    __CU_FUNC_NAME: str = "normalize_tf_"

    def __init__(self):
        super(CuNormalizeTF).__init__()

        self._func: dict = {}

        for dtype_in in dolphin.dtype:
            for dtype_out in dolphin.dtype:
                self._func[dtype_in.cuda_dtype+dtype_out.cuda_dtype] = self.compiled_source.get_function(self.__CU_FUNC_NAME + dtype_in.cuda_dtype + "_" + dtype_out.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        self._func[input.dtype.cuda_dtype+output.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            input.allocation,
            output.allocation,
            input.width,
            input.height,
            input.channel
            )


class CuCvtColorCompiler(dolphin.CudaBase):
    __CU_FILENAME: str = "cvt_color.cu"

    cuda_source: str = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            __CU_FILENAME), "rt", encoding="utf-8").read()
    source = ""
    for dtype in dolphin.dtype:
        source += Template(cuda_source).render(
            dtype=dtype.cuda_dtype
            )
    compiled_source = SourceModule(source)


class CuCvtColorRGB2GRAY(CuCvtColorCompiler):
    __CU_FUNC_NAME: str = "_cvt_color_rgb2gray_"

    def __init__(self):
        super(CuCvtColorRGB2GRAY).__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                self._func[mode+dtype_in.cuda_dtype] = self.compiled_source.get_function(mode + self.__CU_FUNC_NAME + dtype_in.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
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
            input.channel
            )


class CuCvtColorBGR2GRAY(CuCvtColorCompiler):
    __CU_FUNC_NAME: str = "_cvt_color_bgr2gray_"

    def __init__(self):
        super(CuCvtColorBGR2GRAY).__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                self._func[mode+dtype_in.cuda_dtype] = self.compiled_source.get_function(mode + self.__CU_FUNC_NAME + dtype_in.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
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
            input.channel
            )


class CuCvtColorBGR2RGB(CuCvtColorCompiler):
    __CU_FUNC_NAME: str = "_cvt_color_bgr2rgb_"

    def __init__(self):
        super(CuCvtColorBGR2RGB).__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                self._func[mode+dtype_in.cuda_dtype] = self.compiled_source.get_function(mode + self.__CU_FUNC_NAME + dtype_in.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
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
            input.channel
            )


class CuCvtColorRGB2BGR(CuCvtColorCompiler):
    __CU_FUNC_NAME: str = "_cvt_color_rgb2bgr_"

    def __init__(self):
        super(CuCvtColorRGB2BGR).__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                self._func[mode+dtype_in.cuda_dtype] = self.compiled_source.get_function(mode + self.__CU_FUNC_NAME + dtype_in.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 input: dolphin.dimage,
                 output: dolphin.dimage,
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
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
            input.channel
            )


CU_RESIZE_LINEAR = CuResizeNearest()
CU_RESIZE_PADDING = CuResizePadding()
CU_NORMALIZE_MEAN_STD = CuNormalizeMeanStd()
CU_NORMALIZE_255 = CuNormalize255()
CU_NORMALIZE_TF = CuNormalizeTF()
CU_CVT_COLOR_RGB2GRAY = CuCvtColorRGB2GRAY()
CU_CVT_COLOR_BGR2GRAY = CuCvtColorBGR2GRAY()
CU_CVT_COLOR_BGR2RGB = CuCvtColorBGR2RGB()
CU_CVT_COLOR_RGB2BGR = CuCvtColorRGB2BGR()

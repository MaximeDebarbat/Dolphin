"""_summary_
"""

import math

import os
import sys

from typing import Union
import numpy as np
from jinja2 import Template

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CudaBinding  # pylint: disable=import-error
from Data import ImageDimension  # pylint: disable=import-error


class CuLetterBox(CUDA_BASE):
    """CuLetterBox is a class wrapping the CUDA implementation of LetterBox
    preprocessing.
    This preprocessing function is used to resize an image to a given size
    without modifying the aspect ratio of the image. The image is then padded.

    :param out_image_dimension: Output image size
    :type out_image_dimension: ImageDimension
    :param padding_value: Padding value, defaults to 127
    :type padding_value: int, optional
    """

    __CUDA_LETTERBOX_FILE_NAME = "letterbox.cu"
    __CUDA_LETTERBOX_FCT_NAME = "letterbox"

    def __init__(self, out_image_dimension: Union[ImageDimension, tuple],
                 padding_value: int = 127):
        # pylint: disable=redefined-outer-name

        super().__init__()

        if isinstance(out_image_dimension, tuple):
            self._out_image_dimension = ImageDimension(width=out_image_dimension[1],
                                                       height=out_image_dimension[0],
                                                       channels=out_image_dimension[2],
                                                       dtype=np.uint8)
        else:
            self._out_image_dimension = out_image_dimension

        if self._out_image_dimension.dtype == np.uint8:
            _cuda_out_type = "uint8_t"
        elif self._out_image_dimension.dtype == np.uint16:
            _cuda_out_type = "uint16_t"
        elif self._out_image_dimension.dtype == np.int16:
            _cuda_out_type = "int16_t"
        elif (self._out_image_dimension.dtype == np.float32 or
              self._out_image_dimension.dtype == np.float16 or
              self._out_image_dimension.dtype == np.float64):
            _cuda_out_type = "float"
        else:
            raise ValueError(f"Unsupported dtype : \
{self._out_image_dimension.dtype}")

        self._padding_value = padding_value

        self._letterbox_cuda_sm = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CUDA_LETTERBOX_FILE_NAME), "rt", encoding="utf-8")

        processed_template = Template(self._letterbox_cuda_sm.read()).render(
            out_type=_cuda_out_type)

        self._letterbox_cuda_sm = SourceModule(processed_template)

        self._letterbox_fct = self._letterbox_cuda_sm.get_function(
            self.__CUDA_LETTERBOX_FCT_NAME)

        self._out_image_dimension_binding = CudaBinding()

        self._out_image_dimension_binding.allocate(shape=(3,),
                                                   dtype=self._out_image_dimension.shape_dtype)
        self._out_image_dimension_binding.write(data=self.
                                                _out_image_dimension.ndarray)
        self._out_image_dimension_binding.h2d()

        self._padding_value_binding = CudaBinding()

        self._padding_value_binding.allocate(shape=(1,), dtype=np.uint8)
        self._padding_value_binding.write(data=np.array([self._padding_value],
                                                        dtype=np.uint8))
        self._padding_value_binding.h2d()

        self._block = (self.MAX_BLOCK_X, self.MAX_BLOCK_Y, 1)
        self._grid = (max(1, math.ceil(self._out_image_dimension.width /
                                       self._block[0])),
                      max(1, math.ceil(self._out_image_dimension.height /
                                       self._block[1])))

    def __call__(self, binding_in_image: CudaBinding,
                 binding_in_image_size: CudaBinding,
                 binding_out_image: CudaBinding,
                 stream: cuda.Stream() = None) -> None:
        """_summary_

        :param binding_in_image: _description_
        :type binding_in_image: CudaBinding
        :param binding_in_image_size: _description_
        :type binding_in_image_size: CudaBinding
        :param binding_out_image: _description_
        :type binding_out_image: CudaBinding
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        self._letterbox_fct(binding_in_image.device,
                            binding_out_image.device,
                            binding_in_image_size.device,
                            self._out_image_dimension_binding.device,
                            self._padding_value_binding.device,
                            block=self._block, grid=self._grid,
                            stream=stream)


if __name__ == "__main__":

    import cv2
    import time

    N_ITER = int(1e4)
    in_image = cv2.imread("dog.jpg")

    in_image_size = ImageDimension(width=in_image.shape[1],
                                   height=in_image.shape[0],
                                   channels=in_image.shape[2],
                                   dtype=np.uint8)

    in_image_binding = CudaBinding()
    in_image_binding.allocate(shape=in_image_size.shape, dtype=np.uint8)
    in_image_binding.write(data=in_image)
    in_image_binding.h2d()

    in_image_size_binding = CudaBinding()
    in_image_size_binding.allocate(shape=(3,),
                                   dtype=ImageDimension.shape_dtype)
    in_image_size_binding.write(data=in_image_size.ndarray)
    in_image_size_binding.h2d()

    out_image_dimension = ImageDimension(width=640, height=640, channels=3,
                                         dtype=np.uint8)

    out_image_binding = CudaBinding()
    out_image_binding.allocate(shape=out_image_dimension.shape,
                               dtype=out_image_dimension.dtype)

    letterbox = CuLetterBox(out_image_dimension)

    start = time.time()
    for _ in range(N_ITER):
        letterbox(in_image_binding, in_image_size_binding, out_image_binding)
    end = time.time()
    cuda_time = 1000/N_ITER*(end-start)
    print(f"Time to process: {cuda_time}ms/iter over {N_ITER} iteration.")

    out_image_binding.d2h()

    cv2.imwrite("out.jpg", out_image_binding.value)

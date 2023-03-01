
import math

import os
import sys

import numpy as np

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CUDA_Binding
from Data import ImageSize


class CuLetterBox(CUDA_BASE):
    """CuLetterBox is a class wrapping the CUDA implementation of LetterBox
    preprocessing.
    This preprocessing function is used to resize an image to a given size
    without modifying the aspect ratio of the image. The image is then padded.

    :param out_image_size: Output image size
    :type out_image_size: ImageSize
    :param padding_value: Padding value, defaults to 127
    :type padding_value: int, optional
    """

    __CUDA_LETTERBOX_FILE_NAME = "letterbox.cu"
    __CUDA_LETTERBOX_FCT_NAME = "letterbox"

    def __init__(self, out_image_size: ImageSize,
                 padding_value: int = 127):
        # pylint: disable=redefined-outer-name

        super().__init__()

        self._out_image_size = out_image_size
        self._padding_value = padding_value

        self._letterbox_cuda_sm = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CUDA_LETTERBOX_FILE_NAME), "rt", encoding="utf-8")

        self._letterbox_cuda_sm = SourceModule(self._letterbox_cuda_sm.read())
        self._letterbox_fct = self._letterbox_cuda_sm.get_function(
            self.__CUDA_LETTERBOX_FCT_NAME)

        self._out_image_size_binding = CUDA_Binding()

        self._out_image_size_binding.allocate(shape=(3,), dtype=np.uint16)
        self._out_image_size_binding.write(data=self._out_image_size.ndarray)
        self._out_image_size_binding.H2D()

        self._padding_value_binding = CUDA_Binding()

        self._padding_value_binding.allocate(shape=(1,), dtype=np.uint8)
        self._padding_value_binding.write(data=np.array([self._padding_value],
                                                        dtype=np.uint8))
        self._padding_value_binding.H2D()

        self._block = (self.MAX_BLOCK_X, self.MAX_BLOCK_Y, 1)
        self._grid = (max(1, math.ceil(self._out_image_size.width /
                                       self._block[0])),
                      max(1, math.ceil(self._out_image_size.height /
                                       self._block[1])))

    def __call__(self, binding_in_image: CUDA_Binding,
                 binding_in_image_size: CUDA_Binding,
                 binding_out_image: CUDA_Binding,
                 stream: cuda.Stream() = None) -> None:
        """_summary_

        :param binding_in_image: _description_
        :type binding_in_image: CUDA_Binding
        :param binding_in_image_size: _description_
        :type binding_in_image_size: CUDA_Binding
        :param binding_out_image: _description_
        :type binding_out_image: CUDA_Binding
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        self._letterbox_fct(binding_in_image.device,
                            binding_out_image.device,
                            binding_in_image_size.device,
                            self._out_image_size_binding.device,
                            self._padding_value_binding.device,
                            block=self._block, grid=self._grid,
                            stream=stream)


if __name__ == "__main__":

    import cv2
    import time

    N_ITER = int(1e4)
    in_image = cv2.imread("dog.jpg")
    in_image_size = ImageSize(width=in_image.shape[1],
                              height=in_image.shape[0],
                              channels=in_image.shape[2],
                              dtype=np.uint16)

    in_image_binding = CUDA_Binding()
    in_image_binding.allocate(shape=in_image_size.shape, dtype=np.uint8)
    in_image_binding.write(data=in_image)
    in_image_binding.H2D()

    in_image_size_binding = CUDA_Binding()
    in_image_size_binding.allocate(shape=(3,), dtype=np.uint16)
    in_image_size_binding.write(data=in_image_size.ndarray)
    in_image_size_binding.H2D()

    out_image_size = ImageSize(width=640, height=640, channels=3,
                               dtype=np.uint16)

    out_image_binding = CUDA_Binding()
    out_image_binding.allocate(shape=out_image_size.shape, dtype=np.uint8)

    letterbox = CuLetterBox(out_image_size)

    start = time.time()
    for _ in range(N_ITER):
        letterbox(in_image_binding, in_image_size_binding, out_image_binding)
    end = time.time()

    print(f"Time to process: {1000/N_ITER*(end-start)}ms/iter \
          over {N_ITER} iteration.")

    out_image_binding.D2H()

    cv2.imwrite("out.jpg", out_image_binding.value)

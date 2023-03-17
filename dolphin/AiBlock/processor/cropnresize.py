
import math

import os
import sys

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error

import numpy as np

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CudaBinding # pylint: disable=import-error
from Data import ImageDimension # pylint: disable=import-error


class CuCropNResize(CUDA_BASE):
    """CuCropNResize is a class wrapping the CUDA implementation of
    Crop and Resize preprocessing. I reads an image and a list of
    bounding boxes and performs an efficient crop and resize operation.

    :param out_image_size: Expected output image size
    :type out_image_size: ImageDimension
    :param n_max_bboxes: Maximum number of bounding boxes
    :type n_max_bboxes: int
    """

    __CUDA_CROPNRESIZE_FILE_NAME = "cropnresize.cu"
    __CUDA_CROPNRESIZE_FCT_NAME = "cropnresize"

    def __init__(self, out_image_size: ImageDimension,
                 n_max_bboxes: int):
        # pylint: disable=redefined-outer-name

        super().__init__()

        self._out_image_size = out_image_size
        self._n_max_bboxes = n_max_bboxes

        # Here, we import and compile self.__CUDA_FILE_NAME
        self._cnr_cuda_sm = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CUDA_CROPNRESIZE_FILE_NAME), "rt", encoding="utf-8")

        self._cnr_cuda_sm = SourceModule(self._cnr_cuda_sm.read())
        self._cnr_cuda_f = self._cnr_cuda_sm.get_function(
            self.__CUDA_CROPNRESIZE_FCT_NAME)

        self._binding_n_max_bboxes = CudaBinding()
        self._out_image_binding_size = CudaBinding()
        self._binding_max_width = CudaBinding()
        self._binding_max_height = CudaBinding()

        self._binding_n_max_bboxes.allocate(shape=(), dtype=np.uint16)
        self._out_image_binding_size.allocate(shape=(3,),
                                              dtype=self._out_image_size.dtype)
        self._binding_max_width.allocate(shape=(), dtype=np.float32)
        self._binding_max_height.allocate(shape=(), dtype=np.float32)

        self._out_image_binding_size.write(data=self._out_image_size.ndarray)
        self._out_image_binding_size.h2d()

        self._binding_n_max_bboxes.write(data=self._n_max_bboxes)
        self._binding_n_max_bboxes.d2h()

        self._block = self._GET_BLOCK_X_Y(Z=self._n_max_bboxes)
        self._grid = (math.ceil(self._out_image_size.width / self._block[0]),
                      math.ceil(self._out_image_size.height / self._block[1]))

    def __call__(self, in_image_binding: CudaBinding,
                 in_image_binding_size: CudaBinding,
                 binding_bounding_boxes: CudaBinding,
                 out_image_binding_batch: CudaBinding,
                 stream: cuda.Stream = None
                 ) -> None:
        """Callable method to call the CUDA function that performs the crop
        and resize operation.
        For a given image and a list of bounding boxes, this function will
        crop the image and resize it to the desired output size.

        All CudaBinding objects must be allocated and written
        before calling this function.

        F.e.:
          >>> in_image_binding = CudaBinding()
          >>> in_image_binding.allocate(shape=image.shape, dtype=np.uint8)
          >>> in_image_binding.write(data=image)
          >>> in_image_binding.h2d(stream=stream)

        :param in_image_binding: The CudaBinding object containing \
        the input image.
        Must be allocated and written before calling this function.
        :type in_image_binding: CudaBinding
        :param in_image_binding_size: The CudaBinding object containing
        the input image size.
        Must be allocated and written before calling this function.
        :type in_image_binding_size: CudaBinding
        :param binding_bounding_boxes: The CudaBinding object containing
        the bounding boxes.
        Must be allocated and written before calling this function.
        :type binding_bounding_boxes: CudaBinding
        :param out_image_binding_batch: The CudaBinding object containing
        the output image batch. Must be allocated before calling this function.
        :type out_image_binding_batch: CudaBinding
        :param stream: The CUDA stream to perform the operation, defaults to None
        :type stream: cuda.Stream, optional
        :return: None
        :rtype: None
        """

        self._cnr_cuda_f(in_image_binding.device,
                         out_image_binding_batch.device,
                         in_image_binding_size.device,
                         self._out_image_binding_size.device,
                         binding_bounding_boxes.device,
                         block=self._block,
                         grid=self._grid,
                         stream=stream)


if __name__ == "__main__":

    import cv2
    import time

    in_image_binding = CudaBinding()
    in_image_size_binding = CudaBinding()
    bounding_boxes_binding = CudaBinding()

    image = np.random.randint(0, 255,
                              size=(1080, 1920, 3),
                              dtype=np.uint8)

    bboxes_list = [[200, 200, 500, 500],
                   [100, 100, 250, 250],
                   [200, 200, 500, 500],
                   [100, 100, 250, 250],
                   [200, 200, 500, 500],
                   [100, 100, 250, 250],
                   [200, 200, 500, 500],
                   [100, 100, 250, 250],
                   [200, 200, 500, 500],
                   [100, 100, 250, 250]]
    N_MAX_BBOXES = len(bboxes_list)
    N_ITER = int(1e3)

    in_image_binding.allocate(shape=image.shape, dtype=np.uint8)
    in_image_binding.write(data=image)
    in_image_binding.h2d()

    in_image_size_binding.allocate(shape=(3,), dtype=np.uint16)
    in_image_size_binding.write(np.array(image.shape, dtype=np.uint16))
    in_image_size_binding.h2d()

    bounding_boxes_binding.allocate(shape=(N_MAX_BBOXES, 4), dtype=np.uint16)
    bounding_boxes_binding.write(bboxes_list)
    bounding_boxes_binding.h2d()

    out_image_size = ImageDimension(width=500, height=500,
                               channels=3, dtype=np.uint16)

    out_image_binding = CudaBinding()
    out_image_binding.allocate(shape=(N_MAX_BBOXES, out_image_size.height,
                                      out_image_size.width,
                                      out_image_size.channels),
                               dtype=np.uint8)

    cropnrezise = CuCropNResize(out_image_size=out_image_size,
                                n_max_bboxes=N_MAX_BBOXES)

    t1 = time.time()
    for _ in range(N_ITER):
        cropnrezise(in_image_binding=in_image_binding,
                    in_image_binding_size=in_image_size_binding,
                    binding_bounding_boxes=bounding_boxes_binding,
                    out_image_binding_batch=out_image_binding)
    cuda_time = 1000/N_ITER*(time.time()-t1)
    print(f"AVG CUDA Time : {cuda_time}ms/iter over {N_ITER} iterations")

    t1 = time.time()
    for _ in range(N_ITER):
        for i in range(N_MAX_BBOXES):
            x1, y1, x2, y2 = bboxes_list[i]
            cv2.resize(image[y1:y2, x1:x2], (out_image_size.width,
                                             out_image_size.height))
    opencv_time = 1000/N_ITER*(time.time()-t1)
    print(f"OpenCV Time : {opencv_time}ms/iter over {N_ITER} iterations")

    print(f"Speedup : {opencv_time/cuda_time}")

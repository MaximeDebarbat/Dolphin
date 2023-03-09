import os
import sys

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CudaBinding # pylint: disable=import-error
from Data import ImageSize, BoundingBox # pylint: disable=import-error


class CuRescaleBbox(CUDA_BASE):
    """CuRescaleBbox is a class wrapping the CUDA implementation of
    rescaling bounding boxes to a new image size. For instance,
    after TRT:EfficientNMS, we have a list of bounding boxes that
    are relative to the model input size. We need to rescale them
    to the original image size.

    :param in_image_size: Image size of the model's input
    :type in_image_size: ImageSize
    :param in_image_size: Image size of the original image
    :type in_image_size: ImageSize
    """

    __CUDA_RESCALEBBOX_FILE_NAME = "rescalebbox.cu"
    __CUDA_RESCALEBBOX_FCT_NAME = "rescalebbox"

    def __init__(self, in_image_size: ImageSize,
                 rescaled_image_size: ImageSize,
                 n_max_bboxes: int):
        # pylint: disable=redefined-outer-name

        super().__init__()

        if (n_max_bboxes <= 1):
            raise AssertionError(f"n_max_bboxes argument should be >=1. \
                                 Here n_max_bboxes={n_max_bboxes}.")

        self._in_image_size = in_image_size
        self._rescaled_image_size = rescaled_image_size
        self._n_max_bboxes = n_max_bboxes

        # Here, we import and compile self.__CUDA_FILE_NAME
        self.__rb_cuda_sm = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
                                        self.__CUDA_RESCALEBBOX_FILE_NAME),
                                 "rt", encoding="utf-8")
        self.__rb_cuda_sm = SourceModule(self.__rb_cuda_sm.read())
        self.__rb_cuda_f = self.__rb_cuda_sm.get_function(
            self.__CUDA_RESCALEBBOX_FCT_NAME)

        self._binding_in_image_size = CudaBinding()
        self._binding_rescaled_image_size = CudaBinding()

        self._binding_in_image_size.allocate(shape=(3,),
                                             dtype=self._in_image_size.dtype)
        self._binding_rescaled_image_size.allocate(shape=(3,),
                                                   dtype=self._in_image_size.
                                                   dtype)

        ########
        # COPY #
        ########

        self._binding_in_image_size.write(data=self._in_image_size.ndarray)
        self._binding_in_image_size.h2d()

        self._binding_rescaled_image_size.write(
            data=self._rescaled_image_size.ndarray)
        self._binding_rescaled_image_size.h2d()

    def __call__(self, binding_bounding_boxes: CudaBinding,
                 binding_out_bboxes: CudaBinding,
                 stream: cuda.Stream = None) -> None:
        # pylint : disable=redefined-outer-name
        """__call__ is the function called when the instance is called as a
        function. It calls the CUDA kernel for rescaling bounding boxes.

        All CudaBinding objects must be allocated and written
        before calling this function.

        F.e.:
            binding_in_image = CudaBinding()
            binding_in_image.allocate(shape=image.shape, dtype=np.uint8)
            binding_in_image.write(data=image)
            binding_in_image.h2d(stream=stream)

        :param binding_bounding_boxes: Bounding boxes to rescale
        :type binding_bounding_boxes: CudaBinding
        :param binding_out_bboxes: Rescaled bounding boxes
        :type binding_out_bboxes: CudaBinding
        :param stream: CUDA stream, defaults to None
        :type stream: cuda.Stream, optional
        """

        self.__rb_cuda_f(binding_bounding_boxes.device,
                         binding_out_bboxes.device,
                         self._binding_in_image_size.device,
                         self._binding_rescaled_image_size.device,
                         block=(self._n_max_bboxes, 1, 1),
                         grid=(1, 1), stream=stream)


if __name__ == "__main__":

    import time

    in_image_size = ImageSize(width=1920, height=1080, channels=3,
                              dtype=np.uint16)
    rescale_image_size = ImageSize(width=640, height=640, channels=3,
                                   dtype=np.uint16)

    bboxes = [BoundingBox(x0=100, y0=100, x1=600, y1=600, relative=False),
              BoundingBox(x0=0, y0=0, x1=640, y1=640, relative=False),
              BoundingBox(x0=0, y0=0, x1=640, y1=640, relative=False),
              BoundingBox(x0=0, y0=0, x1=640, y1=640, relative=False),
              BoundingBox(x0=0, y0=0, x1=640, y1=640, relative=False),
              BoundingBox(x0=0, y0=0, x1=640, y1=640, relative=False),
              BoundingBox(x0=0, y0=0, x1=640, y1=640, relative=False),
              BoundingBox(x0=0, y0=0, x1=640, y1=640, relative=False),
              BoundingBox(x0=0, y0=0, x1=640, y1=640, relative=False)]

    N_BBOXES = len(bboxes)
    N_ITER = int(1e5)

    bboxes_binding = CudaBinding()
    bboxes_binding.allocate(shape=(N_BBOXES, 4), dtype=np.uint16)
    bboxes_binding.write(data=np.array([e.ndarray for e in bboxes]))
    bboxes_binding.h2d()

    bboxes_out_binding = CudaBinding()
    bboxes_out_binding.allocate(shape=(N_BBOXES, 4), dtype=np.uint16)

    rescaler = CuRescaleBbox(in_image_size=ImageSize(1920, 1080, 3,
                                                     np.uint16),
                             rescaled_image_size=ImageSize(640, 640, 3,
                                                           np.uint16),
                             n_max_bboxes=N_BBOXES)

    t1 = time.time()
    for _ in range(N_ITER):
        rescaler(binding_bounding_boxes=bboxes_binding,
                 binding_out_bboxes=bboxes_out_binding)
    cuda_time = 1000/N_ITER*(time.time()-t1)
    print(f"CUDA AVG time : {cuda_time}ms/iter")
    bboxes_out_binding.d2h()

    bbox_array = np.array([e.ndarray for e in bboxes])
    res = np.zeros_like(bbox_array)
    t1 = time.time()
    for _ in range(N_ITER):
        res[:, 0] = in_image_size.width*(bbox_array[:, 0] /
                                         rescale_image_size.width)
        res[:, 1] = in_image_size.height*(bbox_array[:, 1] /
                                          rescale_image_size.height)
        res[:, 2] = in_image_size.width*(bbox_array[:, 2] /
                                         rescale_image_size.width)
        res[:, 3] = in_image_size.height*(bbox_array[:, 3] /
                                          rescale_image_size.height)
    numpy_time = 1000/N_ITER*(time.time()-t1)
    print(f"CPU AVG time : {numpy_time}ms/iter")

    print(f"Speedup : {numpy_time/cuda_time}")

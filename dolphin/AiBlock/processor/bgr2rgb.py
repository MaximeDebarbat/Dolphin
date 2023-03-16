import os
import sys
import math

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CudaBinding # pylint: disable=import-error
from Data import ImageDimension # pylint: disable=import-error
from image_processor import ImageProcessor


class CuBGR2RGB(ImageProcessor):
    """Class that wraps the CUDA implementation of channel swapping
    from BGR to RGB. It is used to convert an BGR/RGB image to BGR/RGB
    respectively.

    This wrapper is for now only managing images with the format
    HWC.
    """

    _CUDA_FILE_NAME = "bgr2rgb.cu"
    _CUDA_FCT_NAME = "bgr2rgb"

    def __init__(self):
        super().__init__()

        self._cuda_sm = SourceModule(self._cuda_sm.read())
        self._cuda_f = self._cuda_sm.get_function(
            self._CUDA_FCT_NAME)

    def __call__(self, binding_in_image: CudaBinding,
                 binding_in_image_size: CudaBinding,
                 binding_out_image: CudaBinding,
                 stream: cuda.Stream = None):
        # pylint: disable=redefined-outer-name
        """Callable method to call the CUDA function that performs
        channel swapping from BGR to RGB. It reads an image size,
        an image and performs the channel swapping.
        Note that you can use it in the two directions. Such as
        BGR -> RGB and RGB -> BGR.

        All CudaBinding objects must be allocated and written
        before calling this function.

        F.e.:
            binding_in_image = CudaBinding()
            binding_in_image.allocate(shape=image.shape, dtype=np.uint8)
            binding_in_image.write(data=image)
            binding_in_image.h2d(stream=stream)

        :param image_size_binding: Binding containing the image size
        :type image_size_binding: CudaBinding
        :param in_out_image_binding: Image whose channels will be swapped
        :type in_out_image_binding: CudaBinding
        :param stream: Cuda stream to perform async operation, defaults to None
        :type stream: cuda.Stream, optional
        """

        block = (min(binding_in_image_size.value[1], self.MAX_BLOCK_X),
                 min(binding_in_image_size.value[0], self.MAX_BLOCK_Y), 1)

        grid = (max(1, math.ceil(binding_in_image_size.value[1] /
                                 block[0])),
                max(1, math.ceil(binding_in_image_size.value[0] /
                                 block[1])))

        self._cuda_f(binding_in_image_size.device,
                          binding_in_image.device,
                          binding_out_image.device,
                          block=block, grid=grid, stream=stream)


if __name__ == "__main__":

    import time
    import cv2

    stream = cuda.Stream()

    channel_swapper = CuBGR2RGB()

    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    in_image_binding = CudaBinding()
    in_image_binding.allocate(shape=image.shape, dtype=np.uint8)
    in_image_binding.write(data=image.flatten(order="C"))
    in_image_binding.h2d(stream=stream)

    in_image_size_binding = CudaBinding()
    in_image_size_binding.allocate(shape=(3,), dtype=np.uint16)
    in_image_size_binding.write(np.array(image.shape))
    in_image_size_binding.h2d(stream=stream)

    out_image_binding = CudaBinding()
    out_image_binding.allocate(shape=image.shape,
                               dtype=np.uint8)

    N_ITER = int(np.ceil(1e5) // 2 * 2 + 1)

    t1 = time.time()
    for _ in range(N_ITER):
        channel_swapper(binding_in_image=in_image_binding,
                        binding_in_image_size=in_image_size_binding,
                        binding_out_image=out_image_binding,
                        stream=stream)
    cuda_time = 1000/N_ITER*(time.time()-t1)
    print(f"AVG CUDA Time : {cuda_time}ms/iter over {N_ITER} iterations")

    t1 = time.time()
    for _ in range(N_ITER):
        new = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    opencv_time = 1000/N_ITER*(time.time()-t1)
    print(f"OpenCV Time : {opencv_time}ms/iter over {N_ITER} iterations")

    print(f"Speedup : {opencv_time/cuda_time}")

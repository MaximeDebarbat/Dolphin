import os
import sys
import math

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error

import numpy as np

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CUDA_Binding  # pylint: disable=import-error
from Data import ImageSize  # pylint: disable=import-error


class CuHWC2CHW(CUDA_BASE):
    """CuHWC2CWH is a class wrapping the CUDA implementation of
    HWC to CWH preprocessing. It reads an image and performs an efficient
    dimension swapping. It is the equivalent of the operation :

    ...
    np.transpose(image, (2, 1, 0))
    ...

    Assuming `image` is here an HWC image (Default in OpenCV).
    """

    __CUDA_HWC2CHW_FILE_NAME = "hwc2chw.cu"
    __CUDA_HWC2CHW_FCT_NAME = "hwc2chw"

    def __init__(self) -> None:
        super().__init__()

        self._hwc2chw_cuda_sm = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CUDA_HWC2CHW_FILE_NAME), "rt", encoding="utf-8")
        self._hwc2chw_cuda_sm = SourceModule(self._hwc2chw_cuda_sm.read())
        self._hwc2chw_cuda_f = self._hwc2chw_cuda_sm.get_function(
            self.__CUDA_HWC2CHW_FCT_NAME)

        self._block = (self.MAX_BLOCK_X, self.MAX_BLOCK_Y, 1)

    def __call__(self,
                 in_image_binding: CUDA_Binding,
                 in_image_size_binding: CUDA_Binding,
                 out_image_binding: CUDA_Binding,
                 stream: cuda.Stream = None
                 ) -> None:
        # pylint: disable=redefined-outer-name
        """Calls the CUDA implementation of the HWC to CWH
        preprocessing. It reads an image and performs an efficient
        dimension swapping.

        F.e.:
        binding_in_image = CUDA_Binding()
        binding_in_image.allocate(shape=image.shape, dtype=np.uint8)
        binding_in_image.write(data=image)
        binding_in_image.H2D(stream=stream)

        :param image_size_binding: Binding containing the image size
        :type image_size_binding: CUDA_Binding
        :param in_out_image_binding: Image whose channels will be swapped
        :type in_out_image_binding: CUDA_Binding
        :param stream: Cuda stream to perform async operation, defaults to None
        :type stream: cuda.Stream, optional
        """

        grid = (max(1, math.ceil(image_size_binding.value[1] /
                                 self._block[0])),
                max(1, math.ceil(image_size_binding.value[0] /
                                 self._block[1])))

        self._hwc2chw_cuda_f(
            in_image_size_binding.device,
            in_image_binding.device,
            out_image_binding.device,
            block=self._block,
            grid=grid,
            stream=stream)


if __name__ == "__main__":

    import time

    transposer = CuHWC2CHW()
    N_ITER = int(1e5)

    stream = cuda.Stream()

    image_in_shape = ImageSize(1080, 1920, 3, np.uint16)
    image_in = np.random.randint(0, 255, size=(image_in_shape.height,
                                               image_in_shape.width,
                                               image_in_shape.channels),
                                 dtype=np.uint8)

    image_in_binding = CUDA_Binding()
    image_out_binding = CUDA_Binding()
    image_size_binding = CUDA_Binding()

    image_in_binding.allocate(shape=image_in_shape.shape,
                              dtype=np.uint8)
    image_in_binding.write(data=image_in)
    image_in_binding.H2D(stream=stream)

    image_out_binding.allocate(shape=(image_in_shape.channels,
                                      image_in_shape.height,
                                      image_in_shape.width),
                               dtype=np.uint8)

    image_size_binding.allocate(shape=(3,),
                                dtype=np.uint16)
    image_size_binding.write(data=image_in_shape.ndarray)
    image_size_binding.H2D(stream=stream)

    t1 = time.time()
    for _ in range(N_ITER):
        out = np.transpose(image_in, (2, 0, 1))
    numpy_time = 1000/N_ITER*(time.time()-t1)
    print(f"OpenCV Time : {numpy_time}ms/iter over {N_ITER} iterations")

    t1 = time.time()
    for _ in range(N_ITER):
        transposer(image_in_binding, image_size_binding, image_out_binding)
    cuda_time = 1000/N_ITER*(time.time()-t1)
    print(f"CUDA Time : {cuda_time}ms/iter over {N_ITER} iterations")

    print(f"Speedup : {numpy_time/cuda_time}x")

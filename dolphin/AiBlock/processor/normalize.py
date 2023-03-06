import os
import sys
import math

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CUDA_Binding  # pylint: disable=import-error
from Data import ImageSize  # pylint: disable=import-error


class CuNormalize(CUDA_BASE):
    """CuHWC2CWH is a class wrapping the CUDA implementation of
    HWC to CWH preprocessing. It reads an image and performs an efficient
    dimension swapping. It is the equivalent of the operation :

    ...
    np.transpose(image, (2, 1, 0))
    ...

    Assuming `image` is here an HWC image (Default in OpenCV).
    """

    __CUDA_NORMALIZE_FILE_NAME = "normalize.cu"
    __CUDA_NORMALIZE_FCT_NAME = "normalize"

    def __init__(self) -> None:
        super().__init__()

        self._normalize_cuda_sm = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self.__CUDA_NORMALIZE_FILE_NAME), "rt", encoding="utf-8")
        self._normalize_cuda_sm = SourceModule(self._normalize_cuda_sm.read())
        self._normalize_cuda_f = self._normalize_cuda_sm.get_function(
            self.__CUDA_NORMALIZE_FCT_NAME)

        self._block = (self.MAX_BLOCK_X, self.MAX_BLOCK_Y, 1)

    def __call__(self, image_binding: CUDA_Binding,
                 out_image_binding: CUDA_Binding,
                 image_size_binding: CUDA_Binding) -> None:
        # pylint: disable=redefined-outer-name
        """_summary_
        """
        grid = (max(1, math.ceil(image_size_binding.value[1] /
                                 self._block[0])),
                max(1, math.ceil(image_size_binding.value[1] /
                                 self._block[1])))

        self._normalize_cuda_f(image_binding.device,
                               out_image_binding.device,
                               image_size_binding.device,
                               block=self._block,
                               grid=grid)


if __name__ == "__main__":

    import time

    normalizer = CuNormalize()
    N_ITER = int(1e4)

    stream = cuda.Stream()

    image_in_shape = ImageSize(width=1920,
                               height=1080,
                               channels=3,
                               dtype=np.uint16)

    image_in = np.random.randint(0, 255, size=(image_in_shape.channels,
                                               image_in_shape.height,
                                               image_in_shape.width),
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
                               dtype=np.float32)

    image_size_binding.allocate(shape=(3,),
                                dtype=np.uint16)
    image_size_binding.write(data=image_in_shape.ndarray)
    image_size_binding.H2D(stream=stream)

    t1 = time.time()
    for _ in range(N_ITER):
        out = image_in.astype(np.float32) / 255
    numpy_time = 1000/N_ITER*(time.time()-t1)
    print(f"Numpy Time : {numpy_time}ms/iter over {N_ITER} iterations")

    t1 = time.time()
    for _ in range(N_ITER):
        normalizer(image_in_binding, image_out_binding, image_size_binding)
    cuda_time = 1000/N_ITER*(time.time()-t1)
    print(f"CUDA Time : {cuda_time}ms/iter over {N_ITER} iterations")

    print(f"Speedup : {numpy_time/cuda_time}x")

    image_out_binding.D2H(stream=stream)

    print(f"norm : {np.linalg.norm(out-image_out_binding.value)}")

import os
import sys
import math

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error

import numpy as np

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CudaBase, CudaBinding  # pylint: disable=import-error
from Data import ImageDimension  # pylint: disable=import-error
from .image_processor import ImageProcessor


class CuHWC2CHW(ImageProcessor):
    """CuHWC2CWH is a class wrapping the CUDA implementation of
    HWC to CWH preprocessing. It reads an image and performs an efficient
    dimension swapping. It is the equivalent of the operation :

      >>> import numpy as np
      >>> image = cv2.imread(...)
      >>> np.transpose(image, (2, 1, 0))

    Assuming `image` is here an HWC image (Default in OpenCV).
    """

    _CUDA_FILE_NAME: str = "hwc2chw.cu"
    _CUDA_FCT_NAME: str = "hwc2chw"

    def __init__(self) -> None:
        super().__init__()

        self._cuda_sm = SourceModule(self._cuda_sm.read())
        self._cuda_f = self._cuda_sm.get_function(self._CUDA_FCT_NAME)
        self._block = (self.MAX_BLOCK_X, self.MAX_BLOCK_Y, 1)

    def forward(self,
                in_image_binding: CudaBinding,
                in_image_size_binding: CudaBinding,
                out_image_binding: CudaBinding,
                stream: cuda.Stream = None
                ) -> None:
        # pylint: disable=redefined-outer-name
        """Calls the CUDA implementation of the HWC to CWH
        preprocessing. It reads an image and performs an efficient
        dimension swapping.

        F.e.::
          >>> in_image_binding = CudaBinding()
          >>> in_image_binding.allocate(shape=image.shape, dtype=np.uint8)
          >>> in_image_binding.write(data=image)
          >>> in_image_binding.h2d(stream=stream)

        :param in_image_size_binding: Binding containing the image size
        :type in_image_size_binding: CudaBinding
        :param in_out_image_binding: Image whose channels will be swapped
        :type in_out_image_binding: CudaBinding
        :param stream: Cuda stream to perform async operation, defaults to None
        :type stream: cuda.Stream, optional
        """

        grid = (max(1, math.ceil(in_image_size_binding.value[1] /
                                 self._block[0])),
                max(1, math.ceil(in_image_size_binding.value[0] /
                                 self._block[1])))

        self._cuda_f(
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

    image_in_shape = ImageDimension(1080, 1920, 3, np.uint16)
    image_in = np.random.randint(0, 255, size=(image_in_shape.height,
                                               image_in_shape.width,
                                               image_in_shape.channels),
                                 dtype=np.uint8)

    image_in_binding = CudaBinding()
    image_out_binding = CudaBinding()
    image_size_binding = CudaBinding()

    image_in_binding.allocate(shape=image_in_shape.shape,
                              dtype=np.uint8)
    image_in_binding.write(data=image_in)
    image_in_binding.h2d(stream=stream)

    image_out_binding.allocate(shape=(image_in_shape.channels,
                                      image_in_shape.height,
                                      image_in_shape.width),
                               dtype=np.uint8)

    image_size_binding.allocate(shape=(3,),
                                dtype=np.uint16)
    image_size_binding.write(data=image_in_shape.ndarray)
    image_size_binding.h2d(stream=stream)

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

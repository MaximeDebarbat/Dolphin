
import os
import sys
import math
import time
from enum import Enum

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error

import numpy as np

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CudaBinding  # pylint: disable=import-error
from Data import ImageDimension  # pylint: disable=import-error
from image_processor import ImageProcessor


class NormalizeMode(Enum):
    """NORMALIZE_MODES is an enum class defining the different
    normalization modes available in the CUDA implementation.
    """
    MEAN_STD = 0
    _255 = 1
    _128 = 2


class CuNormalize(ImageProcessor):
    """CuHWC2CWH is a class wrapping the CUDA implementation of
    HWC to CWH preprocessing. It reads an image and performs an efficient
    dimension swapping. It is the equivalent of the operation :

    ...
    np.transpose(image, (2, 1, 0))
    ...

    Assuming `image` is here an HWC image (Default in OpenCV).
    """

    _CUDA_FILE_NAME = "normalize.cu"
    _CUDA_NORMALIZE_FCT_NAME_MEAN_STD = "normalize_mean_std"
    _CUDA_NORMALIZE_FCT_NAME_255 = "normalize_255"
    _CUDA_NORMALIZE_FCT_NAME_128 = "normalize_128"

    def __init__(self, norm_type: NormalizeMode =
                 NormalizeMode._255,
                 mean: np.ndarray = None,
                 std: np.ndarray = None) -> None:

        super().__init__()

        self._type = norm_type

        if (mean is not None and std is not None):

            if mean.shape != std.shape:
                raise ValueError(f"mean and std must have the same shape. \
                                 Found {mean.shape} and {std.shape}")

            if (mean.dtype != np.float32 or std.dtype != np.float32):
                raise ValueError(f"mean and std must be float32. Found \
                                    {mean.dtype} and {std.dtype}")

            self._mean_binding = CudaBinding()
            self._std_binding = CudaBinding()

            self._mean_binding.allocate(shape=mean.shape,
                                        dtype=np.float32)
            self._std_binding.allocate(shape=std.shape,
                                       dtype=np.float32)

            self._mean_binding.write(data=mean)
            self._std_binding.write(data=std)

            self._mean_binding.h2d()
            self._std_binding.h2d()

        self._cuda_sm = SourceModule(self._cuda_sm.read())

        self._fct_binder = {
            NormalizeMode.MEAN_STD: self._cuda_sm.get_function(
                                    self._CUDA_NORMALIZE_FCT_NAME_MEAN_STD),
            NormalizeMode._255: self._cuda_sm.get_function(
                                    self._CUDA_NORMALIZE_FCT_NAME_255),
            NormalizeMode._128: self._cuda_sm.get_function(
                                    self._CUDA_NORMALIZE_FCT_NAME_128)
        }

        self._block = (self.MAX_BLOCK_X, self.MAX_BLOCK_Y, 1)

    def __call__(self, binding_in_image: CudaBinding,
                 binding_in_image_size: CudaBinding,
                 binding_out_image: CudaBinding,
                 stream: cuda.Stream = None) -> None:
        # pylint: disable=redefined-outer-name
        """_summary_
        """
        grid = (max(1, math.ceil(binding_in_image_size.value[1] /
                                 self._block[0])),
                max(1, math.ceil(binding_in_image_size.value[1] /
                                 self._block[1])))

        if self._type == NormalizeMode.MEAN_STD:
            self._fct_binder[self._type](binding_in_image.device,
                                         binding_out_image.device,
                                         binding_in_image_size.device,
                                         self._mean_binding.device,
                                         self._std_binding.device,
                                         block=self._block,
                                         grid=grid,
                                         stream=stream)
        else:
            self._fct_binder[self._type](binding_in_image.device,
                                         binding_out_image.device,
                                         binding_in_image_size.device,
                                         block=self._block,
                                         grid=grid,
                                         stream=stream)


def test_255():
    """
    Function to run tests on the CuNormalize class
    using the _255 normalization mode
    """

    normalizer = CuNormalize(norm_type=NormalizeMode._255)
    # pylint: disable=W0212

    n_iter = int(1e3)

    stream = cuda.Stream()

    image_in_shape = ImageDimension(width=1920,
                                    height=1080,
                                    channels=3,
                                    dtype=np.uint16)

    image_in = np.random.randint(0, 255, size=(image_in_shape.channels,
                                               image_in_shape.height,
                                               image_in_shape.width),
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
                               dtype=np.float32)

    image_size_binding.allocate(shape=(3,),
                                dtype=np.uint16)
    image_size_binding.write(data=image_in_shape.ndarray)
    image_size_binding.h2d(stream=stream)

    t_1 = time.time()
    for _ in range(n_iter):
        out = image_in.astype(np.float32) / 255
    numpy_time = 1000/n_iter*(time.time()-t_1)
    print(f"Numpy Time : {numpy_time}ms/iter over {n_iter} iterations")

    t_1 = time.time()
    for _ in range(n_iter):
        normalizer(image_in_binding,
                   image_size_binding,
                   image_out_binding,
                   stream)
    cuda_time = 1000/n_iter*(time.time()-t_1)
    print(f"CUDA Time : {cuda_time}ms/iter over {n_iter} iterations")

    print(f"Speedup : {numpy_time/cuda_time}x")

    image_out_binding.d2h(stream=stream)

    print(f"norm : {np.linalg.norm(out-image_out_binding.value)}")


def test_mean_std():
    """
    Function to run tests on the CuNormalize class
    using the MEAN_STD normalization mode
    """

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    normalizer = CuNormalize(norm_type=NormalizeMode.MEAN_STD,
                             mean=imagenet_mean,
                             std=imagenet_std)
    n_iter = int(1e3)

    stream = cuda.Stream()

    image_in_shape = ImageDimension(width=1920,
                               height=1080,
                               channels=3,
                               dtype=np.uint16)

    image_in = np.random.randint(0, 255, size=(image_in_shape.channels,
                                               image_in_shape.height,
                                               image_in_shape.width),
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
                               dtype=np.float32)

    image_size_binding.allocate(shape=(3,),
                                dtype=np.uint16)
    image_size_binding.write(data=image_in_shape.ndarray)
    image_size_binding.h2d(stream=stream)

    t_1 = time.time()
    for _ in range(n_iter):
        intermediate = image_in.astype(np.float32) / 255
        intermediate[0, :, :] -= imagenet_mean[0]
        intermediate[1, :, :] -= imagenet_mean[1]
        intermediate[2, :, :] -= imagenet_mean[2]

        intermediate[0, :, :] /= imagenet_std[0]
        intermediate[1, :, :] /= imagenet_std[1]
        intermediate[2, :, :] /= imagenet_std[2]

    numpy_time = 1000/n_iter*(time.time()-t_1)
    print(f"Numpy Time : {numpy_time}ms/iter over {n_iter} iterations")

    t_1 = time.time()
    for _ in range(n_iter):
        normalizer(image_in_binding,
                   image_size_binding,
                   image_out_binding,
                   stream)
    cuda_time = 1000/n_iter*(time.time()-t_1)
    print(f"CUDA Time : {cuda_time}ms/iter over {n_iter} iterations")

    print(f"Speedup : {numpy_time/cuda_time}x")

    image_out_binding.d2h(stream=stream)

    print(f"norm : {np.linalg.norm(intermediate-image_out_binding.value)}")


def test_128():
    """
    Function to run tests on the CuNormalize class
    with the _128 normalization mode.
    """

    normalizer = CuNormalize(norm_type=NormalizeMode._128)  # pylint: disable=W0212

    n_iter = int(1e3)

    stream = cuda.Stream()

    image_in_shape = ImageDimension(width=1920,
                               height=1080,
                               channels=3,
                               dtype=np.uint16)

    image_in = np.random.randint(0, 255, size=(image_in_shape.channels,
                                               image_in_shape.height,
                                               image_in_shape.width),
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
                               dtype=np.float32)

    image_size_binding.allocate(shape=(3,),
                                dtype=np.uint16)
    image_size_binding.write(data=image_in_shape.ndarray)
    image_size_binding.h2d(stream=stream)

    t_1 = time.time()
    for _ in range(n_iter):
        out = (image_in.astype(np.float32) - 128) / 128
    numpy_time = 1000/n_iter*(time.time()-t_1)
    print(f"Numpy Time : {numpy_time}ms/iter over {n_iter} iterations")

    t_1 = time.time()
    for _ in range(n_iter):
        normalizer(image_in_binding,
                   image_size_binding,
                   image_out_binding,
                   stream)
    cuda_time = 1000/n_iter*(time.time()-t_1)
    print(f"CUDA Time : {cuda_time}ms/iter over {n_iter} iterations")

    print(f"Speedup : {numpy_time/cuda_time}x")

    image_out_binding.d2h(stream=stream)

    print(f"norm : {np.linalg.norm(out-image_out_binding.value)}")


if __name__ == "__main__":

    print("Test 255")
    test_255()

    print("Test 128")
    test_128()

    print("Test mean std")
    test_mean_std()

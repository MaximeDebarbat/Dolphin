
import math
import time
from enum import Enum
from typing import List, Union, Tuple
import numpy
import pycuda.driver as cuda  # pylint: disable=import-error
import dolphin


class dimage_dim_format(Enum):
    """Image dimension format.

    Attributes:
        CHW: CHW format.
        HWC: HWC format.
        HW: HW format.
    """

    DOLPHIN_CHW = 0
    DOLPHIN_HWC = 1
    DOLPHIN_HW = 2


class dimage_channel_format(Enum):
    """Image channel format.

    Attributes:
        RGB: RGB format.
        BGR: BGR format.
        GRAY_SCALE: GRAY_SCALE format.
    """

    DOLPHIN_RGB = 0
    DOLPHIN_BGR = 1
    DOLPHIN_GRAY_SCALE = 2


class dimage_resize_type(Enum):
    """Image resize type.

    Attributes:
        NEAREST: Nearest neighbor interpolation.
        PADDED: Padded image in order to maintain the aspect ratio.
    """

    DOLPHIN_NEAREST = 0
    DOLPHIN_PADDED = 1


class dimage_normalize_type(Enum):
    """Image normalize type.

    Attributes:
        MEAN_STD: Normalize the image with the mean and standard deviation.
        MIN_MAX: Normalize the image with the minimum and maximum value.
    """

    DOLPHIN_MEAN_STD = 0
    DOLPHIN_255 = 1
    DOLPHIN_128 = 2


class dimage(dolphin.darray):

    def __init__(self,
                 array: numpy.ndarray = None,
                 shape: tuple = None,
                 dtype: dolphin.dtype = dolphin.dtype.uint8,
                 stream: cuda.Stream = None,
                 channel_format: dimage_channel_format = None
                 ) -> None:

        super(dimage, self).__init__(array=array,
                                     shape=shape,
                                     dtype=dtype,
                                     stream=stream)

        if len(self._shape) != 2 and len(self._shape) != 3:
            raise ValueError("The shape of the image must be 2 or 3.")

        self._image_channel_format = channel_format

        if len(self._shape) == 2:
            if self._image_channel_format is not None and \
                    self._image_channel_format != dimage_channel_format.DOLPHIN_GRAY_SCALE:
                raise ValueError(
                    "With a shape of 2 the channel format must be GRAY_SCALE.")

            self._image_channel_format = dimage_channel_format.DOLPHIN_GRAY_SCALE
            self._image_dim_format = dimage_dim_format.DOLPHIN_HW

        else:
            if self._shape[2] == 3:
                if self._image_channel_format is not None and (\
                        self._image_channel_format.value != dimage_channel_format.DOLPHIN_RGB.value and \
                        self._image_channel_format.value != dimage_channel_format.DOLPHIN_BGR.value):
                    raise ValueError(
                        "With a shape of 3 the channel format must be RGB or BGR.")
                self._image_channel_format = self._image_channel_format or dimage_channel_format.DOLPHIN_RGB
                self._image_dim_format = dimage_dim_format.DOLPHIN_HWC

            elif self._shape[0] == 3:
                if self._image_channel_format is not None and \
                        self._image_channel_format.value != dimage_channel_format.DOLPHIN_RGB.value and \
                        self._image_channel_format.value != dimage_channel_format.DOLPHIN_BGR.value:
                    raise ValueError(
                        "With a shape of 3 the channel format must be RGB or BGR.")
                self._image_channel_format = self._image_channel_format or dimage_channel_format.DOLPHIN_RGB
                self._image_dim_format = dimage_dim_format.DOLPHIN_CHW

            elif self._shape[2] == 1:
                if self._image_channel_format is not None and \
                        self._image_channel_format.value != dimage_channel_format.DOLPHIN_GRAY_SCALE.value:
                    raise ValueError(
                        "With a shape of 2 the channel format must be GRAY_SCALE.")

                self._image_channel_format = dimage_channel_format.DOLPHIN_GRAY_SCALE
                self._image_dim_format = dimage_dim_format.DOLPHIN_HW
                self._shape = (self._shape[0], self._shape[1])

            elif self._shape[0] == 1:
                if self._image_channel_format is not None and \
                        self._image_channel_format != dimage_channel_format.DOLPHIN_GRAY_SCALE:
                    raise ValueError(
                        "With a shape of 2 the channel format must be GRAY_SCALE.")

                self._image_channel_format = dimage_channel_format.DOLPHIN_GRAY_SCALE
                self._image_dim_format = dimage_dim_format.DOLPHIN_HW
                self._shape = (self._shape[1], self._shape[2])
            else:
                raise ValueError(f"The shape of the image is not valid. \
Supported shape : (H, W), (H, W, 1), (1, H, W), (H, W, 3), (3, H, W). Got : {self._shape}")

        self.cu_resize_linear = dolphin.cudimage.CU_RESIZE_LINEAR
        self.cu_resize_padding = dolphin.cudimage.CU_RESIZE_PADDING

    def copy(self) -> 'dimage':
        """Returns a copy of the image.

        :return: The copy of the image
        :rtype: dimage
        """

        res = self.__class__(shape=self._shape,
                             dtype=self._dtype,
                             stream=self._stream,
                             channel_format=self._image_channel_format)

        cuda.memcpy_dtod_async(res.allocation,
                               self._allocation,
                               self._nbytes,
                               self._stream)

        return res

    @property
    def image_channel_format(self) -> dimage_channel_format:
        """Returns the image channel format.

        :return: The image channel format
        :rtype: dimage_channel_format
        """
        return self._image_channel_format

    @property
    def image_dim_format(self) -> dimage_dim_format:
        """Returns the image dimension format.

        :return: The image dimension format
        :rtype: dimage_dim_format
        """
        return self._image_dim_format

    @property
    def height(self) -> numpy.uint16:
        """Returns the height of the image.

        :return: The height of the image
        :rtype: numpy.uint16
        """
        if self._image_dim_format in (dimage_dim_format.DOLPHIN_HW,
                                      dimage_dim_format.DOLPHIN_HWC):
            return numpy.uint16(self._shape[0])
        return numpy.uint16(self._shape[1])

    @property
    def width(self) -> numpy.uint16:
        """Returns the width of the image.

        :return: The width of the image
        :rtype: numpy.uint16
        """
        if self._image_dim_format in (dimage_dim_format.DOLPHIN_HW,
                                      dimage_dim_format.DOLPHIN_HWC):
            return numpy.uint16(self._shape[1])
        return numpy.uint16(self._shape[2])

    @property
    def channel(self) -> numpy.uint8:
        """Returns the number of channels of the image.

        :return: The number of channels of the image
        :rtype: numpy.uint8
        """
        if self._image_channel_format == dimage_channel_format.DOLPHIN_RGB:
            return numpy.uint8(3)
        return numpy.uint8(1)

    def astype(self, dtype: dolphin.dtype,
               dst: 'dimage' = None) -> 'dimage':
        """Converts the dimage to a different dtype.
        Note that a copy from device to device is performed.

        :param dtype: Dtype to convert the darray to
        :type dtype: dolphin.dtype
        """

        return super().astype(dtype, dst)

    def resize(self, shape: Tuple[int],
               resize_type: dimage_resize_type,
               dst: 'dimage' = None,
               padding_value: Union[int, float] = 127) -> 'dimage':
        """Resize the image and returns the result.

        :param height: (width, height) where width and height are the new size of the image
        :type height: tuple[int]
        :param resize_type: The type of resize to perform
        :type width: dolphin.dimage_resize_type
        """
        if len(shape) != 2:
            raise ValueError(
                "The shape must be a tuple of 2 elements (width, height)")

        width, height = shape

        if dst is not None and (
           dst.width != width or
           dst.height != height or
           dst.dtype != self.dtype or
           dst.channel != self.channel
           ):
            raise ValueError(
                "The destination image must have the same shape as the source image.")

        if (width == self.width and
           height == self.height):
            if dst is None:
                return self.copy()
            cuda.memcpy_dtod_async(dst.allocation,
                                   self.allocation,
                                   self._stream)
            return dst

        if dst is None:
            if self._image_dim_format == dimage_dim_format.DOLPHIN_HW:
                new_shape = (height, width)
            elif self._image_dim_format == dimage_dim_format.DOLPHIN_HWC:
                new_shape = (height, width, self.channel)
            elif self._image_dim_format == dimage_dim_format.DOLPHIN_CHW:
                new_shape = (self.channel, height, width)
            else:
                raise ValueError(
                    "The image dimension format is not valid.")

            dst = self.__class__(shape=new_shape,
                                 dtype=self.dtype,
                                 stream=self._stream,
                                 channel_format=self._image_channel_format)

        if resize_type.value == dimage_resize_type.DOLPHIN_NEAREST.value:
            block = (32, 32, 1)
            grid = (
                math.ceil(
                    width /
                    block[0]),
                math.ceil(
                    height /
                    block[1]),
                1)

            self.cu_resize_linear(input=self,
                                  output=dst,
                                  block=block,
                                  grid=grid,
                                  stream=self._stream)

        if resize_type.value == dimage_resize_type.DOLPHIN_PADDED.value:
            block = (32, 32, 1)
            grid = (math.ceil(width/block[0]), math.ceil(height/block[1]), 1)

            self.cu_resize_padding(input=self,
                                   output=dst,
                                   padding=padding_value,
                                   block=block,
                                   grid=grid,
                                   stream=self._stream)

        return dst

    def normalize(self, normalize_type: dimage_normalize_type,
                  mean: List[Union[int, float]] = None,
                  std: List[Union[int, float]] = None) -> 'dimage':
        """Normalize the image.

        :param mean: The mean value of the image
        :type mean: float
        :param std: The standard deviation of the image
        :type std: float
        """
        pass

    def cvtColor(self, color_format: dimage_channel_format) -> 'dimage':
        """Convert the image color format.

        :param color_format: The color format
        :type color_format: dimage_channel_format
        """
        pass

    def transpose(self, *axes: int, dst: 'dimage' = None) -> 'dimage':
        """Transpose the image.

        :param axes: The permutation of the axes
        :type axes: int
        :param dst: The destination image
        :type dst: dimage
        :return: The transposed image
        :rtype: dimage
        """
        if dst is not None and (dst.dtype != self.dtype or
                                dst.channel != self.channel or
                                dst.width != self.width or
                                dst.height != self.height):
            raise ValueError(
                "The destination image must have the same shape as the source image.")

        if dst is None:
            new_shape = tuple([self._shape[i] for i in axes])
            dst = self.__class__(shape=new_shape,
                                 dtype=self.dtype,
                                 stream=self._stream,
                                 channel_format=self._image_channel_format)

        return super().transpose(*axes, dst=dst)

def resize(shape: tuple, resize_type: dimage_resize_type, dst: dimage) -> None:
    """Resize the image.

    :param height: The height of the image
    :type height: int
    :param width: The width of the image
    :type width: int
    """
    pass


def normalize(normalize_type: dimage_normalize_type,
              mean: List[Union[int, float]] = None,
              std: List[Union[int, float]] = None,
              dst: dimage = None) -> None:
    """Normalize the image.

    :param mean: The mean value of the image
    :type mean: float
    :param std: The standard deviation of the image
    :type std: float
    """
    pass


def cvtColor(color_format: dimage_channel_format,
             dst: dimage = None) -> None:
    """Convert the image color format.

    :param color_format: The color format
    :type color_format: dimage_channel_format
    """
    pass



##### TEST #####


def letterbox(im: numpy.ndarray,
              new_shape: tuple = (640, 640),
              padding_value: int = 114):

    import cv2
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_NEAREST)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(padding_value,)*len(im.shape))  # add border
    return im


def test(dtype: dolphin.dtype = dolphin.dtype.uint8,
         shape: tuple = (640, 640),
         n_iter: int = 1000):

    import cv2  # pylint: disable=import-outside-toplevel

    print(f"\nRunning test for {dtype.name} | {shape} image")

    ##############################
    # Create the source image
    ##############################

    dummy = numpy.random.randint(0, 255, shape).astype(dtype.numpy_dtype)
    cuda_image = dimage(dummy)

    diff = numpy.linalg.norm(dummy.flatten(order="C") - cuda_image.ndarray.flatten(order="C"))

    assert diff < 1e-5, f"The image is not correctly copied to the GPU. diff : {diff}"

    if len(shape) == 3:
        if shape[2] == 3:
            assert cuda_image.image_dim_format == dimage_dim_format.DOLPHIN_HWC
        elif shape[2] == 1 or shape[0] == 1:
            assert cuda_image.image_dim_format == dimage_dim_format.DOLPHIN_HW
    else:
        assert cuda_image.image_dim_format == dimage_dim_format.DOLPHIN_HW

    print(f"Creation test 1             : passed")

    dummy = numpy.random.randint(0, 255, shape).astype(dtype.numpy_dtype)
    cuda_image = dimage(shape=shape, dtype=dtype)
    cuda_image.ndarray = dummy

    diff = numpy.linalg.norm(dummy.flatten(order="C") - cuda_image.ndarray.flatten(order="C"))

    assert diff < 1e-5, f"The image is not correctly copied to the GPU. diff : {diff}"
    assert isinstance(cuda_image, dimage), f"The transposed image is not a dimage object : {type(cuda_image)}"

    print(f"Creation test 2             : passed")

    cuda_image = dimage(shape=shape, dtype=dtype)
    if len(cuda_image.shape) == 3:
        assert cuda_image.image_channel_format == dimage_channel_format.DOLPHIN_RGB
    else:
        assert cuda_image.image_channel_format == dimage_channel_format.DOLPHIN_GRAY_SCALE

    print(f"Creation test 3             : passed")

    ##############################
    # Copy test
    ##############################

    cuda_image = dimage(shape=shape, dtype=dtype)
    copy = cuda_image.copy()

    diff = numpy.linalg.norm(cuda_image.ndarray.flatten(order="C") - copy.ndarray.flatten(order="C"))

    assert diff < 1e-5, f"The image is not correctly copied to the GPU. diff : {diff}"
    assert isinstance(copy, dimage), f"The transposed image is not a dimage object : {type(cuda_image)}"

    print(f"Copy test 1                 : passed")

    ##############################
    # Transpose the image
    ##############################

    dummy = numpy.random.randint(0, 255, shape).astype(dtype.numpy_dtype)
    cuda_image = dimage(dummy)

    perm_dummy = tuple([i for i in range(len(dummy.shape))][::-1])
    perm_cuda_array = tuple([i for i in range(len(cuda_image.shape))][::-1])

    dummy = dummy.transpose(*perm_dummy)
    cuda_image = cuda_image.transpose(*perm_cuda_array)

    diff = numpy.linalg.norm(dummy - cuda_image.ndarray)

    assert diff < 1e-5, f"The image is not correctly transposed. diff : {diff}"
    assert isinstance(cuda_image, dimage), f"The transposed image is not a dimage object : {type(cuda_image)}"
    if len(cuda_image.shape) == 3:
        assert cuda_image.image_dim_format == dimage_dim_format.DOLPHIN_CHW, \
            f"The image dim format is not correctly set : {cuda_image.image_dim_format}"
    else:
        assert cuda_image.image_dim_format == dimage_dim_format.DOLPHIN_HW

    print(f"Transpose test 1            : passed")

    ##############################
    # Resize the image
    ##############################

    s_dtype = dtype.numpy_dtype
    if dtype.numpy_dtype not in [numpy.uint8, numpy.int8, numpy.uint16, numpy.int16, numpy.int32, numpy.float32, numpy.float64]:
        s_dtype = numpy.uint8

    new_shape = (shape[0] // 2, shape[1] // 2)
    dummy = numpy.random.randint(0, 255, shape).astype(s_dtype)
    cuda_image = dimage(dummy).astype(dtype)

    resized_dummy = cv2.resize(
        dummy, (new_shape[0], new_shape[1]), interpolation=cv2.INTER_NEAREST)

    resized_cuda_image = cuda_image.resize(
        new_shape, resize_type=dimage_resize_type.DOLPHIN_NEAREST)

    diff = numpy.linalg.norm(resized_dummy - resized_cuda_image.astype(dolphin.dtype.from_numpy_dtype(s_dtype)).ndarray)

    assert diff < 1e-5, f"The image is not correctly resized. diff : {diff}"
    assert isinstance(resized_cuda_image, dimage), f"The transposed image is not a dimage object : {type(resized_cuda_image)}"

    print(f"Resize test 1               : passed ({dolphin.dtype.from_numpy_dtype(s_dtype)}/{dtype})")

    s_dtype = dtype.numpy_dtype
    if dtype.numpy_dtype not in [numpy.uint8, numpy.int8, numpy.uint16, numpy.int16, numpy.int32, numpy.float32, numpy.float64]:
        s_dtype = numpy.uint8

    new_shape = (shape[0] // 2, shape[1] // 2)
    dummy = numpy.random.randint(0, 255, (640, 640, 3)).astype(s_dtype)
    cuda_image = dimage(dummy).astype(dtype)

    if len(cuda_image.shape) == 3:

        cuda_image = cuda_image.transpose(2, 0, 1)
        resized_cuda_image = cuda_image.resize(
            new_shape, resize_type=dimage_resize_type.DOLPHIN_NEAREST)
        resized_cuda_image = resized_cuda_image.transpose(1, 2, 0)

        resized_dummy = cv2.resize(dummy,
                                   (new_shape[1], new_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        diff = numpy.linalg.norm(resized_dummy - resized_cuda_image.astype(dolphin.dtype.from_numpy_dtype(s_dtype)).ndarray)

        assert diff < 1e-5, f"The image is not correctly resized. diff : {diff}"

    print(f"Resize test 2               : passed ({dolphin.dtype.from_numpy_dtype(s_dtype)}/{dtype})")

    s_dtype = dtype.numpy_dtype
    if dtype.numpy_dtype not in [numpy.uint8, numpy.int8, numpy.uint16, numpy.int16, numpy.int32, numpy.float32, numpy.float64]:
        s_dtype = numpy.uint8

    new_shape = (shape[0] // 2, shape[1] // 2)
    dummy = numpy.random.randint(0, 255, shape).astype(s_dtype)
    cuda_image = dimage(dummy).astype(dtype)

    resized_dummy = letterbox(dummy, (new_shape[0], new_shape[1]), 127)

    resized_cuda_image = cuda_image.resize(
        new_shape, resize_type=dimage_resize_type.DOLPHIN_PADDED)

    diff = numpy.linalg.norm(resized_dummy - resized_cuda_image.astype(dolphin.dtype.from_numpy_dtype(s_dtype)).ndarray)

    assert diff < 1e-5, f"The image is not correctly resized. diff : {diff}"

    print(f"Resize test 3               : passed ({dolphin.dtype.from_numpy_dtype(s_dtype)}/{dtype})")

if __name__ == "__main__":

    shapes = [(6, 6, 3),
              (640, 640),
              (640, 640, 1)]

    n_iter = int(1e2)

    for shape in shapes:

        test(dolphin.dtype.float32, shape=shape, n_iter=n_iter)
        test(dolphin.dtype.float64, shape=shape, n_iter=n_iter)
        test(dolphin.dtype.uint8, shape=shape, n_iter=n_iter)
        test(dolphin.dtype.uint16, shape=shape, n_iter=n_iter)
        test(dolphin.dtype.uint32, shape=shape, n_iter=n_iter)
        test(dolphin.dtype.int8, shape=shape, n_iter=n_iter)
        test(dolphin.dtype.int16, shape=shape, n_iter=n_iter)
        test(dolphin.dtype.int32, shape=shape, n_iter=n_iter)

"""
# dimage

This module implements `dimage` class.
It inherits from :func:`darray <dolphin.darray>` in order
to provide more functionalities for image processing.

Along with the `dimage` class, it also provides wrappers
in order to simply and fastly use the image processing
methods implemented in `dimage` class.

## dimage class

Main class for image processing inherited from :func:`darray <dolphin.darray>`.
You can find more information about this class in the documentation of
`dimage` class directly. To use it, you can simply import it with::

    from dolphin import dimage

    image = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)

or::

    import cv2
    from dolphin import dimage

    image = cv2.imread("image.jpg")
    dolphin_image = dimage(image)

## Wrappers

The purpose of the wrappers is to provide a simple and fast way
to perform OpenCV & Numpy style image processing functions to quickly
implement common image processing operations.

These operations are as follow:
- resize
- normalize
- cvtColor
and all the wrappers implemented in :func:`dimage <dolphin.darray>` class.

And can be used as follow::

    import cv2
    import dolphin as dp

    image = cv2.imread("image.jpg")
    image = dp.dimage(image)

    resize = dp.resize(image, (100, 100))
    normalize = dp.normalize(resize)
    rgb = dp.cvtColor(normalize, dp.DOLPHIN_RGB)

Also, as described in the documentation of `dimage` class,
you can perform these operations directly from the `dimage` class.

It is important to note that the most efficient way to perform these operations
is to already have allocated the destination object and to pass it
as a parameter.
"""

import math
from enum import Enum
from typing import List, Union, Tuple

import pycuda.driver as cuda  # pylint: disable=import-error

import numpy
import dolphin


class CuResizeNearest(dolphin.CuResizeCompiler):
    __CU_FUNC_NAME: str = "_resize_nearest_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype in dolphin.dtype:
                self._func[mode+dtype.cuda_dtype] = \
                    self.compiled_source.get_function(
                        mode+self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                    "PPHHHHB")

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        if (src.image_dim_format.value ==
           dimage_dim_format.DOLPHIN_CHW.value):
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode+src.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            dst.width,
            dst.height,
            src.channel
            )


class CuResizePadding(dolphin.CuResizeCompiler):
    __CU_FUNC_NAME: str = "_resize_padding_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype in dolphin.dtype:
                self._func[mode+dtype.cuda_dtype] = \
                    self.compiled_source.get_function(
                        mode+self.__CU_FUNC_NAME + dtype.cuda_dtype).prepare(
                    "PPHHHHB"+numpy.dtype(dtype.numpy_dtype).char)

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 padding: Union[float, int],
                 block: tuple,
                 grid: tuple,
                 stream: cuda.Stream = None) -> None:

        if (src.image_dim_format.value ==
           dimage_dim_format.DOLPHIN_CHW.value):
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode+src.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            dst.width,
            dst.height,
            src.channel,
            src.dtype.numpy_dtype(padding)
            )


class CuNormalizeMeanStd(dolphin.CuNormalizeCompiler):
    __CU_FUNC_NAME: str = "_normalize_mean_std_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                for dtype_out in dolphin.dtype:
                    self._func[mode + dtype_in.cuda_dtype +
                               dtype_out.cuda_dtype] = \
                        self.compiled_source.get_function(
                            mode+self.__CU_FUNC_NAME + dtype_in.cuda_dtype +
                            "_" + dtype_out.cuda_dtype).prepare(
                        "PPHHBPP")

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 mean: Union[float, int],
                 std: Union[float, int],
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        if (src.image_dim_format.value ==
           dimage_dim_format.DOLPHIN_CHW.value):
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode + src.dtype.cuda_dtype +
                   dst.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            src.channel,
            mean,
            std
            )


class CuNormalize255(dolphin.CuNormalizeCompiler):
    __CU_FUNC_NAME: str = "normalize_255_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for dtype_in in dolphin.dtype:
            for dtype_out in dolphin.dtype:
                self._func[dtype_in.cuda_dtype+dtype_out.cuda_dtype] = \
                    self.compiled_source.get_function(
                        self.__CU_FUNC_NAME + dtype_in.cuda_dtype +
                        "_" + dtype_out.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        self._func[src.dtype.cuda_dtype +
                   dst.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            src.channel
            )


class CuNormalizeTF(dolphin.CuNormalizeCompiler):
    __CU_FUNC_NAME: str = "normalize_tf_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for dtype_in in dolphin.dtype:
            for dtype_out in dolphin.dtype:
                self._func[dtype_in.cuda_dtype+dtype_out.cuda_dtype] = \
                    self.compiled_source.get_function(
                        self.__CU_FUNC_NAME + dtype_in.cuda_dtype +
                        "_" + dtype_out.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        self._func[src.dtype.cuda_dtype +
                   dst.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            src.channel
            )


class CuCvtColorRGB2GRAY(dolphin.CuCvtColorCompiler):
    __CU_FUNC_NAME: str = "_cvt_color_rgb2gray_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                self._func[mode+dtype_in.cuda_dtype] = \
                    self.compiled_source.get_function(
                        mode + self.__CU_FUNC_NAME +
                        dtype_in.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        if (src.image_dim_format.value ==
           dimage_dim_format.DOLPHIN_CHW.value):
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode+src.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            src.channel
            )


class CuCvtColorBGR2GRAY(dolphin.CuCvtColorCompiler):
    __CU_FUNC_NAME: str = "_cvt_color_bgr2gray_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                self._func[mode+dtype_in.cuda_dtype] = \
                    self.compiled_source.get_function(
                        mode + self.__CU_FUNC_NAME +
                        dtype_in.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        if (src.image_dim_format.value ==
           dimage_dim_format.DOLPHIN_CHW.value):
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode+src.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            src.channel
            )


class CuCvtColorBGR2RGB(dolphin.CuCvtColorCompiler):
    __CU_FUNC_NAME: str = "_cvt_color_bgr2rgb_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                self._func[mode+dtype_in.cuda_dtype] = \
                    self.compiled_source.get_function(
                        mode + self.__CU_FUNC_NAME +
                        dtype_in.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        if (src.image_dim_format.value ==
           dimage_dim_format.DOLPHIN_CHW.value):
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode+src.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            src.channel
            )


class CuCvtColorRGB2BGR(dolphin.CuCvtColorCompiler):
    __CU_FUNC_NAME: str = "_cvt_color_rgb2bgr_"

    def __init__(self):
        super().__init__()

        self._func: dict = {}

        for mode in ["CHW", "HWC"]:
            for dtype_in in dolphin.dtype:
                self._func[mode+dtype_in.cuda_dtype] = \
                    self.compiled_source.get_function(
                    mode + self.__CU_FUNC_NAME + dtype_in.cuda_dtype).prepare(
                    "PPHHB")

    def __call__(self,
                 src: 'dimage',
                 dst: 'dimage',
                 block: cuda.DeviceAllocation,
                 grid: cuda.DeviceAllocation,
                 stream: cuda.Stream = None) -> None:

        if (src.image_dim_format.value ==
           dimage_dim_format.DOLPHIN_CHW.value):
            mode = "CHW"
        else:
            mode = "HWC"

        self._func[mode+src.dtype.cuda_dtype].prepared_async_call(
            grid,
            block,
            stream,
            src.allocation,
            dst.allocation,
            src.width,
            src.height,
            src.channel
            )


class dimage_dim_format(Enum):
    """Image dimension format.

    Attributes:
        DOLPHIN_CHW: CHW format.
        DOLPHIN_HWC: HWC format.
        DOLPHIN_HW: HW format.
    """

    DOLPHIN_CHW: int = 0
    DOLPHIN_HWC: int = 1
    DOLPHIN_HW: int = 2


class dimage_channel_format(Enum):
    """Image channel format.

    Attributes:
        DOLPHIN_RGB: RGB format.
        DOLPHIN_BGR: BGR format.
        DOLPHIN_GRAY_SCALE: GRAY_SCALE format.
    """

    DOLPHIN_RGB = 0
    DOLPHIN_BGR = 1
    DOLPHIN_GRAY_SCALE = 2


class dimage_resize_type(Enum):
    """Image resize type.

    Attributes:
        DOLPHIN_NEAREST: Nearest neighbor interpolation.
        DOLPHIN_PADDED: Padded image in order to maintain the aspect ratio.
    """

    DOLPHIN_NEAREST = 0
    DOLPHIN_PADDED = 1


class dimage_normalize_type(Enum):
    """Image normalize type.

    Attributes:
        DOLPHIN_MEAN_STD: Normalize the image with the mean and standard
                          deviation.
        DOLPHIN_255: Normalize the image with 255.
        DOLPHIN_TF: Normalize the image with the TF method.
    """

    DOLPHIN_MEAN_STD = 0
    DOLPHIN_255 = 1
    DOLPHIN_TF = 2


class dimage(dolphin.darray):
    """
    ## dimage

    This class inherits from :func:`darray <dolphin.darray>` in order
    to provide image processing functionalities.

    ### Constructor

    `dimage` constructor can be called with the following parameters::
        array: numpy.ndarray = None
        shape: tuple = None
        dtype: dolphin.dtype = dolphin.dtype.uint8
        stream: cuda.Stream = None
        channel_format: dimage_channel_format = None

    ### Overview

    `dimage` is made with the same philosophy as `darray` but with
    additionnal functionalities for image processing gpu-accelerated.
    It supports all the methods defined in `darray` but has
    some specific attributes in order to better handle images.

    We tried to partly follow the same philosophy as OpenCV in order
    to make the transition easier.

    Important : dimage are assuming to always follow (height, width)
    order as per defined in OpenCV.

    ### Properties

    In addition to the properties of :func:`darray <dolphin.darray>`,
    `dimage` has the following properties::

        image_dim_format: dimage_dim_format
            The dimension format of the image.

        image_channel_format: dimage_channel_format
            The channel format of the image.

        height: int
            The height of the image.

        width: int
            The width of the image.

        channels: int
            The number of channels of the image.

    ### Methods

    In addition to the methods of :func:`darray <dolphin.darray>`,
    `dimage` has the following methods::

        resize: Resize the image.
        normalize: Normalize the image.
        cvtColor: Convert the image to another color space.
    """

    cu_resize_linear = CuResizeNearest()
    cu_resize_padding = CuResizePadding()
    cu_normalize_mean_std = CuNormalizeMeanStd()
    cu_normalize_255 = CuNormalize255()
    cu_normalize_tf = CuNormalizeTF()
    cu_cvtColor_rbg2gray = CuCvtColorRGB2GRAY()
    cu_cvtColor_bgr2gray = CuCvtColorBGR2GRAY()
    cu_cvtColor_bgr2rgb = CuCvtColorBGR2RGB()
    cu_cvtColor_rgb2bgr = CuCvtColorRGB2BGR()

    def __init__(self,
                 shape: tuple = None,
                 dtype: dolphin.dtype = dolphin.dtype.uint8,
                 stream: cuda.Stream = None,
                 array: numpy.ndarray = None,
                 channel_format: dimage_channel_format = None,
                 allocation: cuda.DeviceAllocation = None
                 ) -> None:

        super().__init__(array=array,
                         shape=shape,
                         dtype=dtype,
                         stream=stream,
                         allocation=allocation)

        if len(self._shape) != 2 and len(self._shape) != 3:
            raise ValueError("The shape of the image must be 2 or 3.")

        self._image_channel_format = channel_format

        if len(self._shape) == 2:
            if self._image_channel_format is not None and \
                    (self._image_channel_format !=
                     dimage_channel_format.DOLPHIN_GRAY_SCALE):
                raise ValueError(
                    "With a shape of 2 the channel format must be GRAY_SCALE.")

            self._image_channel_format = (
                dimage_channel_format.DOLPHIN_GRAY_SCALE)
            self._image_dim_format = dimage_dim_format.DOLPHIN_HW

        else:
            if self._shape[2] == 3:
                if (self._image_channel_format is not None and
                    self._image_channel_format.value not in (
                        dimage_channel_format.DOLPHIN_RGB.value,
                        dimage_channel_format.DOLPHIN_BGR.value)):
                    raise ValueError(
                        "With a shape of 3 the channel format must be RGB \
or BGR.")
                self._image_channel_format = (
                    self._image_channel_format or
                    dimage_channel_format.DOLPHIN_RGB)
                self._image_dim_format = dimage_dim_format.DOLPHIN_HWC

            elif self._shape[0] == 3:
                if (self._image_channel_format is not None and
                    self._image_channel_format.value not in (
                        dimage_channel_format.DOLPHIN_RGB.value,
                        dimage_channel_format.DOLPHIN_BGR.value)):
                    raise ValueError(
                        "With a shape of 3 the channel format must be RGB \
or BGR.")
                self._image_channel_format = (
                    self._image_channel_format or
                    dimage_channel_format.DOLPHIN_RGB)
                self._image_dim_format = dimage_dim_format.DOLPHIN_CHW

            elif self._shape[2] == 1:
                if self._image_channel_format is not None and \
                        self._image_channel_format.value != \
                        dimage_channel_format.DOLPHIN_GRAY_SCALE.value:
                    raise ValueError(
                        "With a shape of 2 the channel format must \
be GRAY_SCALE.")

                self._image_channel_format = (
                    dimage_channel_format.DOLPHIN_GRAY_SCALE)
                self._image_dim_format = dimage_dim_format.DOLPHIN_HW
                self._shape = (self._shape[0], self._shape[1])

            elif self._shape[0] == 1:
                if self._image_channel_format is not None and \
                        self._image_channel_format != \
                        dimage_channel_format.DOLPHIN_GRAY_SCALE:
                    raise ValueError(
                        "With a shape of 2 the channel format must \
be GRAY_SCALE.")

                self._image_channel_format = (
                    dimage_channel_format.DOLPHIN_GRAY_SCALE)
                self._image_dim_format = dimage_dim_format.DOLPHIN_HW
                self._shape = (self._shape[1], self._shape[2])
            else:
                raise ValueError(f"The shape of the image is not valid. \
Supported shape : (H, W), (H, W, 1), (1, H, W), (H, W, 3), (3, H, W). \
Got : {self._shape}")

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
        if self._image_channel_format.value in (
                dimage_channel_format.DOLPHIN_RGB.value,
                dimage_channel_format.DOLPHIN_BGR.value):
            return numpy.uint8(3)
        return numpy.uint8(1)

    def astype(self, dtype: dolphin.dtype,
               dst: 'dimage' = None) -> 'dimage':
        """
        ### Convert the image to a different type

        This function converts the image to a different type.
        To use it efficiently, the destination image must be
        preallocated and passed as an argument to the function::

            src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
            dst = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.float32)
            src.astype(dolphin.dtype.float32, dst)

        :param dtype: Dtype to convert the darray to
        :type dtype: dolphin.dtype
        :param dst: Destination image
        :type dst: dimage
        """

        return super().astype(dtype, dst)

    def resize_padding(self,
                       shape: Tuple[int],
                       dst: 'dimage' = None,
                       padding_value: Union[int, float] = 127
                       ) -> Tuple['dimage', Tuple[int, int]]:
        """
        ### Padded resize the image

        This function resizes the image to a new shape with padding.
        It means that the image is resized to the new shape and
        the remaining pixels are filled with the padding value
        (127 by default).

        If for instance the image is resized from (50, 100) to (200, 200),
        the aspect ratio of the image is preserved and the image is resized
        to (100, 200). The remaining pixels are filled with the padding value.
        In this scenario, the padding would appear on the left and right side
        of the image, with a width of 50 pixels.

        In order to use this function
        in an efficient way, the destination image must be preallocated
        and passed as an argument to the function::

            src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
            dst = dimage(shape=(200, 200, 3), dtype=dolphin.dtype.uint8)
            src.resize_padding((200, 200), dst)

        If aspect ratio does not matter, the function :func:`resize` can be
        used.

        :param shape: The new shape of the image
        :type shape: tuple
        :param dst: The destination image
        :type dst: dimage
        :param padding_value: The padding value
        :type padding_value: int or float
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
                "The destination image must have \
consitent shapes with resize shape.")

        if (width == self.width and
           height == self.height):
            if dst is None:
                return self.copy(), 1.0, (0, 0)
            cuda.memcpy_dtod_async(dst.allocation,
                                   self.allocation,
                                   self._stream)
            return dst, 1.0, (0, 0)

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

        block = (32, 32, 1)
        grid = (
            math.ceil(
                width /
                block[0]),
            math.ceil(
                height /
                block[1]),
            1)

        self.cu_resize_padding(src=self,
                               dst=dst,
                               padding=padding_value,
                               block=block,
                               grid=grid,
                               stream=self._stream)

        r = min(width / self.width, height / self.height)
        new_unpad = (int(self.width * r), int(self.height * r))
        dw, dh = (width - new_unpad[0]) // 2, (height - new_unpad[1]) // 2

        return dst, r, (dw, dh)

    def resize(self,
               shape: Tuple[int],
               dst: 'dimage' = None,
               ) -> 'dimage':
        """
        ### Resize the image

        This function performs a naive resize of the image. The resize
        type is for now only DOLPHIN_NEAREST. In order to use this function
        in an efficient way, the destination image must be preallocated
        and passed as an argument to the function::

            src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
            dst = dimage(shape=(200, 200, 3), dtype=dolphin.dtype.float32)
            src.resize((200, 200), dst)

        The returned resized image aspect ratio might change as the new shape
        is not necessarily a multiple of the original shape.

        If aspect ratio of the orginal image matters to you, use
        :func:`resize_padding` instead::

            src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
            dst = dimage(shape=(200, 200, 3), dtype=dolphin.dtype.float32)
            src.resize_padding((200, 200), dst)

        :param shape: The new shape of the image
        :type shape: tuple
        :param dst: The destination image
        :type dst: dimage
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
                "The destination image must have consitent \
shapes with resize shape.")

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

        block = (32, 32, 1)
        grid = (
            math.ceil(
                width /
                block[0]),
            math.ceil(
                height /
                block[1]),
            1)

        self.cu_resize_linear(src=self,
                              dst=dst,
                              block=block,
                              grid=grid,
                              stream=self._stream)
        return dst

    def normalize(self, normalize_type: dimage_normalize_type =
                  dimage_normalize_type.DOLPHIN_255,
                  mean: List[Union[int, float]] = None,
                  std: List[Union[int, float]] = None,
                  dtype: dolphin.dtype = dolphin.dtype.float32,
                  dst: 'dimage' = None) -> 'dimage':
        """
        ### Normalize the image

        This function is a function to efficiently normalize an image
        in different manners.

        The mean and std values must be passed as a list of values if you want
        to normalize the image using the DOLPHIN_MEAN_STD normalization
        type. To use this function efficiently, the destination image
        must be preallocated and passed as an argument to the function::

            src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
            dst = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.float32)
            src.normalize(dimage_normalize_type.DOLPHIN_MEAN_STD,
                          mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5], dst=dst)

        or::

            src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
            dst = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.float32)
            src.normalize(dimage_normalize_type.DOLPHIN_255, dst=dst)

        :param normalize_type: The type of normalization
        :type normalize_type: dimage_normalize_type
        :param mean: The mean values
        :type mean: List[Union[int, float]]
        :param std: The std values
        :type std: List[Union[int, float]]
        :param dst: The destination image
        :type dst: dimage
        """
        if dst is None:
            dst = self.__class__(shape=self.shape,
                                 dtype=dtype,
                                 stream=self._stream,
                                 channel_format=self._image_channel_format)

        elif dst.shape != self.shape:
            raise ValueError(
                "The destination image must have the same shape as \
                the source image defined in the function arguments.")

        if (normalize_type.value ==
           dimage_normalize_type.DOLPHIN_MEAN_STD.value):
            if mean is None or std is None:
                raise ValueError(
                    "The mean and std values must be \
specified for the mean/std normalization.")

            if len(mean) != self.channel or len(std) != self.channel:
                raise ValueError(
                    "The mean and std values must have \
the same length as the number of channels.")

            # Allocate the mean and std values in the GPU
            # and copy the values to the GPU
            # We do len(mean) * 4 because the mean and std values
            # are float32 (4 bytes)

            mean_allocation = cuda.mem_alloc(len(mean) * 4)
            std_allocation = cuda.mem_alloc(len(std) * 4)

            cuda.memcpy_htod_async(mean_allocation,
                                   numpy.array(mean).astype(numpy.float32),
                                   stream=self._stream)

            cuda.memcpy_htod_async(std_allocation,
                                   numpy.array(std).astype(numpy.float32),
                                   stream=self._stream)

            block = (32, 32, 1)
            grid = (
                math.ceil(
                    self.width /
                    block[0]),
                math.ceil(
                    self.height /
                    block[1]),
                1)

            self.cu_normalize_mean_std(src=self,
                                       dst=dst,
                                       mean=mean_allocation,
                                       std=std_allocation,
                                       block=block,
                                       grid=grid,
                                       stream=self._stream)

        elif normalize_type.value == dimage_normalize_type.DOLPHIN_255.value:
            block = (32, 32, 1)
            grid = (
                math.ceil(
                    self.width /
                    block[0]),
                math.ceil(
                    self.height /
                    block[1]),
                1)

            self.cu_normalize_255(src=self,
                                  dst=dst,
                                  block=block,
                                  grid=grid,
                                  stream=self._stream)

        elif normalize_type.value == dimage_normalize_type.DOLPHIN_TF.value:
            block = (32, 32, 1)
            grid = (
                math.ceil(
                    self.width /
                    block[0]),
                math.ceil(
                    self.height /
                    block[1]),
                1)

            self.cu_normalize_tf(src=self,
                                 dst=dst,
                                 block=block,
                                 grid=grid,
                                 stream=self._stream)

        else:
            raise ValueError(
                "The normalize type is not valid.")

        return dst

    def cvtColor(self, color_format: dimage_channel_format,
                 dst: 'dimage' = None) -> 'dimage':
        """
        ### Transforms the image to the specified color format.

        This function transforms the image to the specified color format. The
        supported color formats are::

            - dolphin.dimage_channel_format.DOLPHIN_RGB
            - dolphin.dimage_channel_format.DOLPHIN_BGR
            - dolphin.dimage_channel_format.DOLPHIN_GRAY_SCALE

        :param color_format: The color format of the output image
        :type color_format: dimage_channel_format
        :param dst: The destination image
        :type dst: dimage
        """

        if (color_format.value ==
           dimage_channel_format.DOLPHIN_GRAY_SCALE.value):

            if dst is None:
                dst = self.__class__(shape=(self.height, self.width),
                                     dtype=self.dtype,
                                     stream=self._stream,
                                     channel_format=color_format)
            else:
                if dst.shape != (self.height,
                                 self.width) or dst.dtype != self.dtype:
                    raise ValueError(
                        "The destination image must have the same shape as \
the source image and the same dtype as defined in the function arguments.")

            if self.channel != 3:
                raise ValueError(
                    "The source image must have 3 channels.")

            block = (32, 32, 1)
            grid = (
                math.ceil(
                    self.width /
                    block[0]),
                math.ceil(
                    self.height /
                    block[1]),
                1)

            if (self.image_channel_format.value ==
               dimage_channel_format.DOLPHIN_RGB.value):
                self.cu_cvtColor_rbg2gray(src=self,
                                          dst=dst,
                                          block=block,
                                          grid=grid,
                                          stream=self._stream)

            elif (self.image_channel_format.value ==
                  dimage_channel_format.DOLPHIN_BGR.value):
                self.cu_cvtColor_bgr2gray(src=self,
                                          dst=dst,
                                          block=block,
                                          grid=grid,
                                          stream=self._stream)

        elif color_format.value == dimage_channel_format.DOLPHIN_RGB.value:

            if (self.image_channel_format.value ==
               dimage_channel_format.DOLPHIN_GRAY_SCALE.value):
                raise ValueError(
                    "The source image must have 3 channels.")

            if dst is None:
                dst = self.__class__(shape=self.shape,
                                     dtype=self.dtype,
                                     stream=self._stream,
                                     channel_format=color_format)

            elif dst.shape != self.shape or dst.dtype != self.dtype:
                raise ValueError(
                    "The destination image must have the same shape as \
the source image and the same dtype as defined in the function arguments.")

            if (self.image_channel_format.value ==
               dimage_channel_format.DOLPHIN_RGB.value):

                cuda.memcpy_dtod_async(dst.allocation,
                                       self.allocation,
                                       self.nbytes,
                                       self._stream)

            else:
                block = (32, 32, 1)
                grid = (
                    math.ceil(
                        self.width /
                        block[0]),
                    math.ceil(
                        self.height /
                        block[1]),
                    1)

                self.cu_cvtColor_bgr2rgb(src=self,
                                         dst=dst,
                                         block=block,
                                         grid=grid,
                                         stream=self._stream)

        elif color_format.value == dimage_channel_format.DOLPHIN_BGR.value:

            if (self.image_channel_format.value ==
               dimage_channel_format.DOLPHIN_GRAY_SCALE.value):
                raise ValueError(
                    "The source image must have 3 channels.")

            if dst is None:
                dst = self.__class__(shape=self.shape,
                                     dtype=self.dtype,
                                     stream=self._stream,
                                     channel_format=color_format)

            elif dst.shape != self.shape or dst.dtype != self.dtype:
                raise ValueError(
                    "The destination image must have the same shape as \
the source image and the same dtype as defined in the function arguments \
arguments.")

            if (self.image_channel_format.value ==
               dimage_channel_format.DOLPHIN_BGR.value):

                cuda.memcpy_dtod_async(dst.allocation,
                                       self.allocation,
                                       self.nbytes,
                                       self._stream)
            else:

                block = (32, 32, 1)
                grid = (
                    math.ceil(
                        self.width /
                        block[0]),
                    math.ceil(
                        self.height /
                        block[1]),
                    1)

                self.cu_cvtColor_rgb2bgr(src=self,
                                         dst=dst,
                                         block=block,
                                         grid=grid,
                                         stream=self._stream)

        else:
            raise ValueError(
                "The color format is not valid.")

        if int(dst.allocation) == int(self.allocation):
            self._image_channel_format = color_format

        return dst

    def transpose(self, *axes: int, dst: 'dimage' = None) -> 'dimage':
        """
        ### Transpose the image

        This function transposes the image. The axes are specified
        as a sequence of axis numbers.

        To be used efficiently,
        the destination image must be provided and
        must have the same shape as the source image::

            src = dimage(shape=(2, 3, 4), dtype=np.float32)
            dst = dimage(shape=(4, 3, 2), dtype=np.float32)
            src.transpose(2, 1, 0, dst=dst)

        :param axes: The permutation of the axes
        :type axes: *int
        :param dst: The destination image
        :type dst: dimage
        :return: The transposed image
        :rtype: dimage
        """
        new_shape = tuple(self._shape[i] for i in axes)
        if dst is not None and (dst.dtype != self.dtype or
                                dst.channel != self.channel):
            raise ValueError(f"The destination image must have the same dtype \
and number of channels than source image. Found dst:{dst.dtype} and \
expected:{self.dtype}, dst:{dst.channel} and expected:{self.channel}")

        if dst is not None and dst.shape != new_shape:
            raise ValueError(
                f"The destination image must have a consistent shape \
regarding axes. Found dst:{dst.shape} and expected:{new_shape}")

        if (dst is not None and
           dst.image_channel_format != self.image_channel_format):
            raise ValueError(
                f"The destination image must have a consistent channel format \
regarding axes. Found dst:{dst.image_channel_format} and \
expected:{self.image_channel_format}")

        if dst is None:
            dst = self.__class__(shape=new_shape,
                                 dtype=self.dtype,
                                 stream=self._stream,
                                 channel_format=self._image_channel_format)

        return super().transpose(*axes, dst=dst)


def resize_padding(src: dimage,
                   shape: tuple,
                   padding_value: Union[int, float] = 127,
                   dst: dimage = None) -> Tuple['dimage',
                                                float,
                                                Tuple[float, float]]:
    """
    ### Padded resize the image

    This function resizes the image to a new shape with padding.
    It means that the image is resized to the new shape and
    the remaining pixels are filled with the padding value (127 by default).

    If for instance the image is resized from (50, 100) to (200, 200),
    the aspect ratio of the image is preserved and the image is resized
    to (100, 200). The remaining pixels are filled with the padding value.
    In this scenario, the padding would appear on the left and right side
    of the image, with a width of 50 pixels.

    In order to use this function
    in an efficient way, the destination image must be preallocated
    and passed as an argument to the function::

        src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
        dst = dimage(shape=(200, 200, 3), dtype=dolphin.dtype.uint8)
        src.resize_padding((200, 200), dst)

    If aspect ratio does not matter, the function :func:`resize` can be
    used.

    :param shape: The new shape of the image
    :type shape: tuple
    :param dst: The destination image
    :type dst: dimage
    :param padding_value: The padding value
    :type padding_value: int or float
    """

    return src.resize_padding(shape=shape,
                              padding_value=padding_value,
                              dst=dst)


def resize(src: dimage,
           shape: tuple,
           dst: dimage = None
           ) -> 'dimage':
    """
    ### Resize the image

    This function performs a naive resize of the image. The resize
    type is for now only DOLPHIN_NEAREST. In order to use this function
    in an efficient way, the destination image must be preallocated
    and passed as an argument to the function::

        src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
        dst = dimage(shape=(200, 200, 3), dtype=dolphin.dtype.float32)
        dolphin.resize(src, (200, 200), dst)

    The returned resized image aspect ratio might change as the new shape
    is not necessarily a multiple of the original shape.

    If aspect ratio of the orginal image matters to you, use
    :func:`resize_padding` instead::

        src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
        dst = dimage(shape=(200, 200, 3), dtype=dolphin.dtype.float32)
        dolphin.resize_padding(src, (200, 200), dst)

    :param shape: The new shape of the image
    :type shape: tuple
    :param dst: The destination image
    :type dst: dimage
    """

    return src.resize(shape=shape, dst=dst)


def normalize(src: dimage,
              normalize_type: dimage_normalize_type,
              mean: List[Union[int, float]] = None,
              std: List[Union[int, float]] = None,
              dtype: dolphin.dtype = None,
              dst: dimage = None) -> None:
    """
    ### Normalize the image

    This function is a wrapper for
    the normalize function of the dimage class. The mean and
    std values must be passed as a list of values if you want
    to normalize the image using the DOLPHIN_MEAN_STD normalization
    type. To use this function efficiently, the destination image
    must be preallocated and passed as an argument to the function::

        src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
        dst = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.float32)
        normalize(src,
                  dimage_normalize_type.DOLPHIN_MEAN_STD,
                  mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5], dst=dst)

    or::

        src = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.uint8)
        dst = dimage(shape=(100, 100, 3), dtype=dolphin.dtype.float32)
        normalize(src, dimage_normalize_type.DOLPHIN_255, dst=dst)

    :param src: Source image
    :type src: dimage
    :param normalize_type: The type of normalization
    :type normalize_type: dimage_normalize_type
    :param mean: The mean values
    :type mean: List[Union[int, float]]
    :param std: The std values
    :type std: List[Union[int, float]]
    :param dst: The destination image
    :type dst: dimage
    """

    return src.normalize(normalize_type=normalize_type,
                         mean=mean,
                         std=std,
                         dtype=dtype,
                         dst=dst)


def cvtColor(src: dimage,
             color_format: dimage_channel_format,
             dst: dimage = None) -> None:
    """
    ### Transforms the image to the specified color format.

    This function is a wrapper for the cvtColor function of the dimage class.

    This function transforms the image to the specified color format. The
    supported color formats are::

        - dolphin.dimage_channel_format.DOLPHIN_RGB
        - dolphin.dimage_channel_format.DOLPHIN_BGR
        - dolphin.dimage_channel_format.DOLPHIN_GRAY_SCALE

    :param color_format: The color format of the output image
    :type color_format: dimage_channel_format
    :param dst: The destination image
    :type dst: dimage
    """
    return src.cvtColor(color_format=color_format, dst=dst)

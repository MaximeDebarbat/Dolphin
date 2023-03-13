"""_summary_
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ImageDimension:
    '''
    ImageDimension defines a generic dataclass in order to
    formally manipulate image dimensions on host-side only.
    Image dimension relies on these information :
        - width dimension
        - height dimension
        - channels dimension
        - dtype of image

    This class will thus be used to define the shape of an image,
    its type.

    It contains the following properties:
      >>> width: np.uint16
      >>> height: np.uint16
      >>> channels: np.uint16
      >>> dtype: np.dtype
      >>> shape_dtype: np.uint16 -> datatype of the shape
      >>> ndarray: np.ndarray -> np.ndarray = dtype(h,w,c)
      >>> bytesize: int -> size in bytes of the object
      >>> ImageDimension: int -> size in bytes of the image
    '''

    width: np.uint16
    height: np.uint16
    channels: np.uint16
    dtype: np.uint8
    shape_dtype: np.dtype = np.uint16

    @property
    def shape(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return (self.height, self.width, self.channels)

    @property
    def shape_byte_size(self) -> int:
        """
        shape_byte_size is useful to know the size of the shape
        in bytes in order to allocate memory for example.

        With CUDA, we need to know the size of the shape
        in bytes in order to allocate memory.

        :return: _description_
        :rtype: int
        """
        return int(self.width*self.height*self.channels *
                   np.dtype(self.shape_dtype).itemsize)

    @property
    def image_byte_size(self) -> int:
        """
        image_byte_size is useful to know the size of the whole image
        itself as it is required to allocate memory for example.

        With CUDA, we need to know the size of the shape
        in bytes in order to allocate memory.

        :return: _description_
        :rtype: int
        """
        return int(self.width*self.height*self.channels *
                   np.dtype(self.dtype).itemsize)

    @property
    def ndarray(self) -> np.ndarray:
        """_summary_

        :return: _description_
        :rtype: np.ndarray
        """

        return np.array([self.height, self.width, self.channels]).flatten(
            order='C').astype(self.shape_dtype)


@dataclass
class BoundingBox:
    '''
    BoundingBox defines a generic dataclass in order to
    formallymanipulate Bounding Boxes, can be relative,
    within [0..1] or absolute, within [0...{WIDTH, HEIGHT}]
    and will only be defined on 16 bytes (float16 or uint16)
    '''

    x_0: object
    y_0: object
    x_1: object
    y_1: object
    relative: bool

    @classmethod
    def relative_bounding_blox(cls, bbox: 'BoundingBox',
                               size: ImageDimension) -> 'BoundingBox':
        """_summary_

        :param bbox: _description_
        :type bbox: BoundingBox
        :param size: _description_
        :type size: ImageDimension
        :return: _description_
        :rtype: BoundingBox
        """

        return BoundingBox(
            x_0=bbox.x_0/size.width,
            y_0=bbox.y_0/size.height,
            x_1=bbox.x_1/size.width,
            y_1=bbox.y_1/size.height,
            relative=True
        )

    def itemsize(self) -> int:
        """_summary_

        :return: _description_
        :rtype: int
        """

        if self.relative:
            return 64*np.dtype(np.float32).itemsize

        return 64*np.dtype(np.uint16).itemsize

    @property
    def ndarray(self) -> np.ndarray:
        """_summary_

        :return: _description_
        :rtype: np.ndarray
        """

        if self.relative:
            dtype = np.dtype(np.float32)
        else:
            dtype = np.dtype(np.uint16)

        return np.array([self.x_0, self.y_0, self.x_1, self.y_1],
                        dtype=dtype, order="C")

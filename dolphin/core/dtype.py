"""_summary_
"""

from enum import Enum
import numpy


class dtype(Enum):  # pylint: disable=invalid-name
    """Dolphin data types
    In order to manage the data types in Dolphin, bind dolphin types
    the numpy data types as well as the CUDA data types.
    To do so, each element from the Enum class is a tuple containing
    the numpy data type (numpy.dtype) and the CUDA data type (str).

    Example::

      uint8 = (numpy.uint8, "uint8_t")
      int8 = (numpy.int8, "int8_t")
      uint16 = (numpy.uint16, "uint16_t")
      int16 = (numpy.int16, "int16_t")
      uint32 = (numpy.uint32, "uint32_t")
      int32 = (numpy.int32, "int32_t")
      float32 = (numpy.float32, "float")
      float64 = (numpy.float64, "double")

    Properties::

        numpy_dtype: numpy.dtype
            >>> mytype = dtype.uint8
            >>> mynptype = mytype.numpy_dtype (numpy.uint8)

        cuda_dtype: numpy.dtype
            >>> mytype = dtype.uint8
            >>> mycudatype = mytype.cuda_dtype ('uint8_t')

    """

    uint8 = (numpy.uint8, "uint8_t")  # pylint: disable=invalid-name
    int8 = (numpy.int8, "int8_t")  # pylint: disable=invalid-name
    uint16 = (numpy.uint16, "uint16_t")  # pylint: disable=invalid-name
    int16 = (numpy.int16, "int16_t")  # pylint: disable=invalid-name
    uint32 = (numpy.uint32, "uint32_t")  # pylint: disable=invalid-name
    int32 = (numpy.int32, "int32_t")  # pylint: disable=invalid-name
    float32 = (numpy.float32, "float")  # pylint: disable=invalid-name
    float64 = (numpy.float64, "double")  # pylint: disable=invalid-name

    @property
    def numpy_dtype(self) -> numpy.dtype:
        """Since Dolphin data types are tuples, we need to access the first
        element which is the numpy data type.

        :return: The equivalent numpy data type of Dolphin data type
        :rtype: numpy.dtype
        """
        return self.value[0]

    @property
    def cuda_dtype(self) -> str:
        """Since Dolphin data types are tuples, we need to access the second
        element which is the CUDA data type. Which as well are standard
        C types.

        :return: The equivalent CUDA data type of Dolphin data type
        :rtype: str
        """
        return self.value[1]


if __name__ == "__main__":

    mydtype = dtype.uint8

    print(mydtype)
    print(mydtype.numpy_dtype)
    print(mydtype.cuda_dtype)

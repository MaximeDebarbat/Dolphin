"""_summary_
"""

from enum import Enum
from typing import Union
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

        @property
        numpy_dtype: numpy.dtype
            >>> mytype = dtype.uint8
            >>> mynptype = mytype.numpy_dtype (numpy.uint8)

        @property
        cuda_dtype: numpy.dtype
            >>> mytype = dtype.uint8
            >>> mycudatype = mytype.cuda_dtype ('uint8_t')

        @property
        itemsize: int
            >>> a = dtype.uint8
            >>> a.itemsize            # 1
            >>> a = dtype.uint16
            >>> a.itemsize            # 2
            >>> a = dtype.uint32
            >>> a.itemsize            # 4

        __getitem__(key: [str, int]): numpy.dtype or str
            >>> a = dtype.uint8
            >>> a[0]                  # numpy.uint8
            >>> a[1]                  # 'uint8_t'
            >>> a["numpy_dtype"]      # numpy.uint8
            >>> a["cuda_dtype"]       # 'uint8_t'

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

    @property
    def itemsize(self) -> int:
        """Returns the size of the data type in bytes.
        Uses the numpy data type to get the size :

          >>> @property
          >>> def itemsize(self) -> int:
          >>>     return self.numpy_dtype.itemsize

        """
        return self.numpy_dtype.itemsize

    def __getitem__(self,
                    key: Union[str, int]
                    ) -> Union[numpy.dtype, str]:
        """In order to dynamically access the numpy and CUDA data types,
        we also need to implement the __getitem__ method.
        if key is an integer, it will return one of the tuple element
        as long as the key is either 0 or 1.
        if key is a string, it will return the numpy or CUDA data type
        as long as the key is either 'numpy_dtype' or 'cuda_dtype'.

        Usage::

          a = dtype.uint8
          a[0]                  # numpy.uint8
          a[1]                  # 'uint8_t'
          a["numpy_dtype"]      # numpy.uint8
          a["cuda_dtype"]       # 'uint8_t'

        :param key: 'numpy_dtype' or 'cuda_dtype' or a int 0 or 1
        :type key: Union[str, int]
        :raises KeyError: If the key is not valid as described above
        :return: The numpy or CUDA data type
        :rtype: Union[numpy.dtype, str]
        """
        if isinstance(key, int):
            if (key != 0 or key != 1):
                raise KeyError("Key must be either 0 or 1")
            return self.value[key]
        elif isinstance(key, str):
            if key == "numpy_dtype":
                return self.numpy_dtype
            elif key == "cuda_dtype":
                return self.cuda_dtype
            else:
                raise KeyError("Key must be either 'numpy_dtype' \
or 'cuda_dtype'")


if __name__ == "__main__":

    mydtype = dtype.uint8

    print(mydtype)
    print(mydtype.numpy_dtype)
    print(mydtype.cuda_dtype)
    print(mydtype[0])
    print(mydtype["cuda_dtype"])

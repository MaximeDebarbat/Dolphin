"""_summary_
"""

from enum import Enum
from typing import Union, Any
import numpy


class dtype(Enum):  # pylint: disable=invalid-name
    """Dolphin data types
    In order to manage the data types in Dolphin, bind dolphin types
    the numpy data types as well as the CUDA data types.
    To do so, each element from the Enum class is a tuple containing
    the numpy data type (numpy.dtype) and the CUDA data type (str).
    """

    uint8 = (numpy.uint8, "uint8_t")  # pylint: disable=invalid-name
    uint16 = (numpy.uint16, "uint16_t")  # pylint: disable=invalid-name
    uint32 = (numpy.uint32, "uint32_t")  # pylint: disable=invalid-name
    int8 = (numpy.int8, "int8_t")  # pylint: disable=invalid-name
    int16 = (numpy.int16, "int16_t")  # pylint: disable=invalid-name
    int32 = (numpy.int32, "int32_t")  # pylint: disable=invalid-name
    float32 = (numpy.float32, "float")  # pylint: disable=invalid-name
    float64 = (numpy.float64, "double")  # pylint: disable=invalid-name

    @classmethod
    def _missing_(cls, value: numpy.dtype) -> 'dtype':
        return cls.from_numpy_dtype(value)

    def __call__(self, value: Any) -> numpy.dtype:
        """In order to use the data type as a function to cast a value into a
        particular type, we need to implement the __call__ method.

        Example::

          a = dtype.uint8
          a(4) -> numpy.uint8(4)

        :param value: The value to cast to the data type
        :type value: Any
        :return: The numpy casted number passed as value
        :rtype: Any
        """
        return self.numpy_dtype(value)

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
        return numpy.dtype(self.numpy_dtype).itemsize

    @staticmethod
    def from_numpy_dtype(numpy_dtype: numpy.dtype) -> "dtype":
        """Returns the equivalent Dolphin data type from the numpy data type.

        :param numpy_dtype: The numpy data type
        :type numpy_dtype: numpy.dtype
        :return: The equivalent Dolphin data type
        :rtype: dtype
        """
        for d in dtype:
            if d.numpy_dtype == numpy_dtype:
                return d
        raise ValueError("Invalid numpy data type")

    def __str__(self) -> str:
        """Triggered when casting into str::
            str(dp.uint8) -> dp.uint8.__str__()

        :return: representation of the data type
        :rtype: str
        """
        return self.cuda_dtype

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
        raise KeyError("Key must be either 'numpy_dtype' \
or 'cuda_dtype'")

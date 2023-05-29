
from typing import Tuple
import math
import pycuda

import numpy as np


class CudaBase:
    """
    This class is mainly used to access device information, such as
    maximum number of threads per block, size of grid etc...
    This is a base class used by many other classes. It has a lot of
    class attributes in order not to load the same things again and
    again in order to speed up execution.
    """

    #: Used device
    device: pycuda._driver.Device = pycuda.driver.Context.get_device()

    #: Maximum number of threads per blocks. Usually, it is 1024
    max_threads_per_block: int = pycuda.autoinit.device.get_attribute(
                pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK)

    #: Maximum number of blocks per grid x on dim
    max_grid_dim_x: int = pycuda.autoinit.device.get_attribute(
            pycuda.driver.device_attribute.MAX_GRID_DIM_X)

    #: Maximum number of blocks per grid y on dim
    max_grid_dim_y: int = pycuda.autoinit.device.get_attribute(
            pycuda.driver.device_attribute.MAX_GRID_DIM_Y)

    #: Maximum number of blocks per grid z on dim
    max_grid_dim_z: int = pycuda.autoinit.device.get_attribute(
            pycuda.driver.device_attribute.MAX_GRID_DIM_Z)

    #: Warp size
    warp_size: int = pycuda.tools.DeviceData(device).warp_size

    #: Number of MP
    multiprocessor_count: int = pycuda.autoinit.device.get_attribute(
            pycuda.driver.device_attribute.MULTIPROCESSOR_COUNT)

    #: Number of threads per MP
    threads_blocks_per_mp: int = pycuda.tools.DeviceData(
        device).thread_blocks_per_mp

    @staticmethod
    def GET_BLOCK_GRID_1D(n: int
                          ) -> Tuple[Tuple[int, int, int],
                                     Tuple[int, int]]:
        """
        In ordert to perform memory coalescing on 1D iterations,
        we need to efficiently compute the block & grid sizes.

        :param n: Number of elements to process
        :type n: int
        :return: block, grid
        :rtype: Tuple[Tuple[int, int, int], Tuple[int, int]]
        """

        min_threads: int = CudaBase.warp_size
        max_threads: int = 4 * CudaBase.warp_size

        max_blocks: int = (4 * CudaBase.threads_blocks_per_mp *
                           CudaBase.multiprocessor_count)

        if n < min_threads:
            return (min_threads, 1, 1), (1, 1)
        elif n < (max_blocks * min_threads):
            return (min_threads, 1, 1), ((n + min_threads - 1) //
                                         min_threads, 1)
        elif n < (max_blocks * max_threads):
            grid = (max_blocks, 1)
            grp = (n + min_threads - 1) // min_threads
            return ((grp + max_blocks - 1) // max_blocks *
                    min_threads), grid

        return (max_threads, 1, 1), (max_blocks, 1)

    @staticmethod
    def GET_BLOCK_X_Y(Z: int) -> tuple:
        """Get the block size for a given Z.
        The block size is calculated using the following formula:
        (max(ceil(sqrt(MAX_THREADS_PER_BLOCKS/Z)),1),
        max(ceil(sqrt(MAX_THREADS_PER_BLOCKS/Z)),1), Z)

        It is useful to quickly compute the block size that suits
        self._max_threads_per_block for
        a given Z which can be channels, depth, batch size, etc.

        :param Z: Size of the third dimension
        :type Z: int
        :return: Optimal block size that ensure
                 block[0]*block[1]*block[2] <= self._max_threads_per_block
        :rtype: tuple
        """

        _s = int(np.sqrt(CudaBase.max_threads_per_block / int(Z)))
        return (_s, _s, Z)

    @staticmethod
    def GET_GRID_SIZE(size: tuple, block: tuple) -> Tuple[int, int]:
        """Get the grid size for a given size and block size.
        The grid size is calculated using the following formula:
        (max(ceil(sqrt(size/block[0])),1), max(ceil(sqrt(size/block[1])),1))

        This function should be used when the width and height of the image
        are the same or can be swapped.

        :param size: Total size of data
        :type size: tuple
        :param block: Current value of the block size
        :type block: tuple
        :return: Grid size
        :rtype: tuple
        """

        size /= block[0] * block[1]
        return (max(math.ceil(np.sqrt(size)), 1),
                max(math.ceil(np.sqrt(size)), 1))

    @staticmethod
    def GET_GRID_SIZE_HW(size: tuple, block: tuple) -> Tuple[int, int]:
        """Get the grid size for a given size and block size.
        The grid size is calculated using the following formula:
        (max(ceil(sqrt(size[0]/block[0])),1),
        max(ceil(sqrt(size[1]/block[1])),1))

        This function should be used when the width and height
        of the image are different and matter.

        :param size: Height and width of the image
        :type size: tuple
        :param block: Current value of the block size
        :type block: tuple
        :return: Grid size
        :rtype: tuple
        """
        return (max(math.ceil(np.sqrt(size[0] / block[0])), 1),
                max(math.ceil(np.sqrt(size[1] / block[1])), 1))

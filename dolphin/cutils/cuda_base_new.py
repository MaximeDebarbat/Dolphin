
from typing import Tuple
import pycuda

import numpy as np
import math


class CudaBaseNew:

    def __init__(self):

        self.DEVICE = pycuda.driver.Context.get_device()

        self.MAX_THREADS_PER_BLOCKS = int(
            pycuda.autoinit.device.get_attribute(
                pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK))

        self.MAX_GRID_DIM_X = int(
            pycuda.autoinit.device.get_attribute(
                pycuda.driver.device_attribute.MAX_GRID_DIM_X))

        self.MAX_GRID_DIM_Y = int(
            pycuda.autoinit.device.get_attribute(
                pycuda.driver.device_attribute.MAX_GRID_DIM_Y))

        self.MAX_GRID_DIM_Z = int(
            pycuda.autoinit.device.get_attribute(
                pycuda.driver.device_attribute.MAX_GRID_DIM_Z))

        self.WARP_SIZE = int(pycuda.tools.DeviceData(self.DEVICE).warp_size)

        self.MULTIPROCESSOR_COUNT = int(
            pycuda.autoinit.device.get_attribute(
                pycuda.driver.device_attribute.MULTIPROCESSOR_COUNT))

        self.THREADS_BLOCKS_PER_MP = int(
            pycuda.tools.DeviceData(self.DEVICE).thread_blocks_per_mp)

        self.MIN_THREADS = self.WARP_SIZE
        self.MAX_THREADS = 4*self.WARP_SIZE

        self.MAX_BLOCKS = 4 * self.THREADS_BLOCKS_PER_MP * self.MULTIPROCESSOR_COUNT


    def GET_BLOCK_GRID_1D(self, n: int) -> Tuple[tuple, tuple]:
        """
        In ordert to perform memory coalescing on 1D iterations,
        we need to efficiently compute the block & grid sizes.

        :param n: Number of elements to process
        :type n: int
        :return: block, grid
        :rtype: Tuple[tuple, tuple]
        """

        if n < self.MIN_THREADS:
            return (self.MIN_THREADS, 1, 1), (1, 1)
        elif n < (self.MAX_BLOCKS * self.MIN_THREADS):
            return (self.MIN_THREADS, 1, 1), ((n + self.MIN_THREADS - 1) // self.MIN_THREADS, 1)
        elif n < (self.MAX_BLOCKS * self.MAX_THREADS):
            grid = (self.MAX_BLOCKS, 1)
            grp = (n + self.MIN_THREADS - 1) // self.MIN_THREADS
            return ((grp + self.MAX_BLOCKS - 1) // self.MAX_BLOCKS * self.MIN_THREADS), grid

        return (self.MAX_THREADS, 1, 1), (self.MAX_BLOCKS, 1)

    def _GET_BLOCK_X_Y(self, Z: int) -> tuple:
        """Get the block size for a given Z.
        The block size is calculated using the following formula:
        (max(ceil(sqrt(MAX_THREADS_PER_BLOCKS/Z)),1), max(ceil(sqrt(MAX_THREADS_PER_BLOCKS/Z)),1), Z)

        It is useful to quickly compute the block size that suits self.MAX_THREADS_PER_BLOCKS for
        a given Z which can be channels, depth, batch size, etc.

        :param Z: Size of the third dimension
        :type Z: int
        :return: Optimal block size that ensure block[0]*block[1]*block[2] <= self.MAX_THREADS_PER_BLOCKS
        :rtype: tuple
        """

        _s = int(np.sqrt(self.MAX_THREADS_PER_BLOCKS / int(Z)))
        return (_s, _s, Z)

    def _GET_GRID_SIZE(self, size: tuple, block: tuple) -> tuple:
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

    def _GET_GRID_SIZE_HW(self, size: tuple, block: tuple) -> tuple:
        """Get the grid size for a given size and block size.
        The grid size is calculated using the following formula:
        (max(ceil(sqrt(size[0]/block[0])),1), max(ceil(sqrt(size[1]/block[1])),1))

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

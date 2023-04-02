import pycuda
import pycuda.autoinit

import numpy as np
import math


class CudaBase:

    def __init__(self):
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

        if round(
            np.sqrt(
                self.MAX_THREADS_PER_BLOCKS)) != np.sqrt(
                self.MAX_THREADS_PER_BLOCKS):
            self.MAX_BLOCK_X = round(np.sqrt(self.MAX_THREADS_PER_BLOCKS))
            self.MAX_BLOCK_Y = int(
                self.MAX_THREADS_PER_BLOCKS /
                self.MAX_BLOCK_X)
        else:
            self.MAX_BLOCK_X = int(np.sqrt(self.MAX_THREADS_PER_BLOCKS))
            self.MAX_BLOCK_Y = int(np.sqrt(self.MAX_THREADS_PER_BLOCKS))

        self.TOTAL_THREADS = self.MAX_BLOCK_X * self.MAX_BLOCK_Y

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

import pycuda
import pycuda.autoinit

import numpy as np

class CUDA_BASE:

    def __init__(self):

        self.MAX_THREADS_PER_BLOCKS = int(pycuda.autoinit.device.get_attribute(pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK))
        self.MAX_GRID_DIM_X = int(pycuda.autoinit.device.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_X))
        self.MAX_GRID_DIM_Y = int(pycuda.autoinit.device.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_Y))
        self.MAX_GRID_DIM_Z = int(pycuda.autoinit.device.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_Z))

        if(round(np.sqrt(self.MAX_THREADS_PER_BLOCKS))!=np.sqrt(self.MAX_THREADS_PER_BLOCKS)):
            self.MAX_BLOCK_X = round(np.sqrt(self.MAX_THREADS_PER_BLOCKS))
            self.MAX_BLOCK_Y = int(self.MAX_THREADS_PER_BLOCKS/self.MAX_BLOCK_X)
        else:
            self.MAX_BLOCK_X =  int(np.sqrt(self.MAX_THREADS_PER_BLOCKS))
            self.MAX_BLOCK_Y = int(np.sqrt(self.MAX_THREADS_PER_BLOCKS))
        
        self.TOTAL_THREADS = self.MAX_BLOCK_X*self.MAX_BLOCK_Y
        
    def _GET_BLOCK_X_Y(self, Z:int) -> tuple:
        """
        """
        
        _s = int(np.sqrt(self.MAX_THREADS_PER_BLOCKS/int(Z)))
        return (_s,_s, Z)
    
    def _GET_GRID_SIZE(self, size, block) -> tuple:
        """
        """
        
        size /= block[0]*block[1]
        return (max(int(np.sqrt(size)),1), max(int(np.sqrt(size)),1))
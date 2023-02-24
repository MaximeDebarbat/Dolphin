import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

from pycuda.compiler import SourceModule

from enum import Enum
import os
import sys
import inspect
import math

from CudaUtils import CUDA_BASE, CUDA_Binding
from Data import ImageSize, BoundingBox

class CuResize(CUDA_BASE):
    
    __CUDA_RESIZE_FILE_NAME = "resize.cu"
    __CUDA_RESIZE_FCT_NAME = "resize"
    
    def __init__(self, out_image_size:ImageSize
                ) -> None:
        
        super().__init__()       
        
        self._out_image_size = out_image_size
        
        self.__RESIZE_cuda_SM = open(os.path.join(os.path.split(os.path.abspath(__file__))[0],"cuda",self.__CUDA_RESIZE_FILE_NAME),"rt")
        self.__RESIZE_cuda_SM = SourceModule(self.__RESIZE_cuda_SM.read())
        self._RESIZE_CUDA_F = self.__RESIZE_cuda_SM.get_function(self.__CUDA_RESIZE_FCT_NAME)
        
        self._out_image_size_binding = CUDA_Binding()
        
        self._out_image_size_binding.allocate(shape=(3,), dtype=np.uint16)
        self._out_image_size_binding.write(data=out_image_size.ndarray)
        self._out_image_size_binding.H2D()
        
        self._BLOCK = (self.MAX_BLOCK_X,self.MAX_BLOCK_Y,1)        
        
    def __call__(self,
                 in_image_binding:CUDA_Binding,
                 in_image_size_binding:CUDA_Binding,
                 out_image_binding:CUDA_Binding,
                 stream:cuda.Stream
                )->None:
        
        _T =  math.ceil(in_image_size_binding.size/self.TOTAL_THREADS)
        _GRID = (_T,_T)
        
        self._RESIZE_CUDA_F(in_image_binding.device,
                            out_image_binding.device,
                            in_image_size_binding.device,
                            self._out_image_size_binding.device,
                            block=self._BLOCK, grid=_GRID,
                            stream=stream
                            )
        
    
        
        
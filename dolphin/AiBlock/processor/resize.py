import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule

import numpy as np

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import math

from CudaUtils import CUDA_BASE, CUDA_Binding
from Data import ImageSize

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
        
        self._BLOCK = (min(self._out_image_size.width,self.MAX_BLOCK_X),min(self._out_image_size.height,self.MAX_BLOCK_Y),1)        
        self._GRID = (max(1,math.ceil(self._out_image_size.width/self._BLOCK[0])),max(1,math.ceil(self._out_image_size.height/self._BLOCK[1])))
        
    def __call__(self,
                 in_image_binding:CUDA_Binding,
                 in_image_size_binding:CUDA_Binding,
                 out_image_binding:CUDA_Binding,
                 stream:cuda.Stream=None
                )->None:
        
        
        
        #print(f"block : {self._BLOCK} {self._GRID}")
        #sys.exit(0)
        
        self._RESIZE_CUDA_F(in_image_binding.device,
                            out_image_binding.device,
                            in_image_size_binding.device,
                            self._out_image_size_binding.device,
                            block=self._BLOCK, grid=self._GRID,
                            stream=stream
                            )
        
    
if __name__ == "__main__":
    
    import time
    import cv2
    
    stream = cuda.Stream()
    
    out_image_size = ImageSize(width=100, height=100, channels=3, dtype=np.uint8)
    resizer = CuResize(out_image_size=out_image_size)
    
    image = cv2.imread("dog.jpg")
        
    in_image_binding = CUDA_Binding()
    in_image_binding.allocate(shape=(image.shape[0],image.shape[1],3), dtype=np.uint8)
    in_image_binding.write(data=image.flatten(order="C")) 
    in_image_binding.H2D(stream=stream)
    
    in_image_size_binding = CUDA_Binding()
    in_image_size_binding.allocate(shape=(3,), dtype=np.uint16)
    in_image_size_binding.write(np.array(image.shape))
    in_image_size_binding.H2D(stream=stream)
    
    out_image_binding = CUDA_Binding()
    out_image_binding.allocate(shape=(out_image_size.height,out_image_size.width,out_image_size.channels), dtype=np.uint8)
    
    N_ITER = int(1e5)
    t1 = time.time()    
    for _ in range(N_ITER):
        resizer(in_image_binding=in_image_binding,
                in_image_size_binding=in_image_size_binding,
                out_image_binding=out_image_binding,
                stream=stream)
    
    print(f"AVG CUDA Time : {1000/N_ITER*(time.time()-t1)}")
    
    out_image_binding.D2H(stream=stream)    
        
    print(f"Original image :\n{in_image_binding.value}")
    print(f"Reshaped image :\n{out_image_binding.value}")
    
    cv2.imwrite("resized.jpg", out_image_binding.value)
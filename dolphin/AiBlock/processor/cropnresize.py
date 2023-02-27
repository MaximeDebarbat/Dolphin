
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule

import numpy as np
import math

import os
import sys
sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CUDA_Binding
from Data import ImageSize

class CuCropNResize(CUDA_BASE):
    
    __CUDA_CROPNRESIZE_FILE_NAME = "cropnresize.cu"
    __CUDA_CROPNRESIZE_FCT_NAME = "cropnresize"
    __CUDA_GETMAXDIM_FCT_NAME = "getmaxdim"
    
    def __init__(self, out_image_size:ImageSize, 
                       n_max_bboxes:int):
        
        super().__init__()
        
        self._out_image_size = out_image_size
        self._n_max_bboxes = n_max_bboxes
        
        # Here, we import and compile self.__CUDA_FILE_NAME
        self.__CNR_cuda_SM = open(os.path.join(os.path.split(os.path.abspath(__file__))[0],"cuda",self.__CUDA_CROPNRESIZE_FILE_NAME),"rt")
        self.__CNR_cuda_SM = SourceModule(self.__CNR_cuda_SM.read())
        self._CNR_CUDA_F = self.__CNR_cuda_SM.get_function(self.__CUDA_CROPNRESIZE_FCT_NAME)
        self._GMD_CUDA_F = self.__CNR_cuda_SM.get_function(self.__CUDA_GETMAXDIM_FCT_NAME)
        
        self._binding_n_max_bboxes = CUDA_Binding()
        self._binding_out_image_size = CUDA_Binding()
        self._binding_max_width = CUDA_Binding()
        self._binding_max_height = CUDA_Binding()
        
        self._binding_n_max_bboxes.allocate(shape=(), dtype=np.uint16)
        self._binding_out_image_size.allocate(shape=(3,), dtype=self._out_image_size.dtype)
        self._binding_max_width.allocate(shape=(), dtype=np.float32)
        self._binding_max_height.allocate(shape=(), dtype=np.float32)
                
        ######## 
        # COPY #
        ########
        
        self._binding_out_image_size.write(data=self._out_image_size.ndarray)
        self._binding_out_image_size.H2D()
        
        self._binding_n_max_bboxes.write(data=self._n_max_bboxes)
        self._binding_n_max_bboxes.H2D()
                
        self._BLOCK = self._GET_BLOCK_X_Y(Z=self._n_max_bboxes)
        self._GRID = (math.ceil(self._out_image_size.width/self._BLOCK[0]),math.ceil(self._out_image_size.height/self._BLOCK[1]))
            
    def __call__(self, binding_in_image:CUDA_Binding, 
                       binding_in_image_size:CUDA_Binding,
                       binding_bounding_boxes:CUDA_Binding,
                       binding_out_image_batch:CUDA_Binding,
                       stream:cuda.Stream=None
                       ) -> None:       
        
        self._CNR_CUDA_F(binding_in_image.device,
                         binding_out_image_batch.device,
                         binding_in_image_size.device,
                         self._binding_out_image_size.device,
                         binding_bounding_boxes.device,
                         block=self._BLOCK,
                         grid=self._GRID,
                         stream=stream)
    
    @property
    def outImageBatch(self) -> CUDA_Binding:
        return self._binding_image_batch
    
    
if __name__ == "__main__":
    
    import cv2
    import time
    
    binding_in_image = CUDA_Binding()
    binding_in_image_size = CUDA_Binding()
    binding_bounding_boxes = CUDA_Binding()
    
    image = np.random.randint(0,255, size=(1080,1920,3), dtype=np.uint8)
    bboxes_list = [[200,200,500,500],[100,100,250,250],[200,200,500,500],[100,100,250,250],[200,200,500,500],[100,100,250,250],[200,200,500,500],[100,100,250,250],[200,200,500,500],[100,100,250,250]]
    N_MAX_BBOXES = len(bboxes_list)
    N_ITER = int(1e3)
    
    stream = cuda.Stream()

    binding_in_image.allocate(shape=image.shape, dtype=np.uint8)
    binding_in_image.write(data=image)
    binding_in_image.H2D(stream=stream)
    
    binding_in_image_size.allocate(shape=(3,), dtype=np.uint16)
    binding_in_image_size.write(np.array(image.shape, dtype=np.uint16))
    binding_in_image_size.H2D(stream=stream)
    
    binding_bounding_boxes.allocate(shape=(N_MAX_BBOXES,4), dtype=np.uint16)
    binding_bounding_boxes.write(bboxes_list)
    binding_bounding_boxes.H2D(stream=stream)
    
    out_image_size = ImageSize(width=224, height=224, channels=3, dtype=np.uint16)
    
    out_image_binding = CUDA_Binding()
    out_image_binding.allocate(shape=(N_MAX_BBOXES, out_image_size.height, out_image_size.width, out_image_size.channels), dtype=np.uint8)
    
    cropnrezise = CuCropNResize(out_image_size=out_image_size, n_max_bboxes=N_MAX_BBOXES)
    
    t1 = time.time()
    for _ in range(N_ITER):
        cropnrezise(binding_in_image=binding_in_image,
                binding_in_image_size=binding_in_image_size,
                binding_bounding_boxes=binding_bounding_boxes,
                binding_out_image_batch=out_image_binding,
                stream=stream)
    cuda_time = 1000/N_ITER*(time.time()-t1)
    print(f"AVG CUDA Time : {cuda_time}ms/iter over {N_ITER} iterations")
    
    t1 = time.time()
    for _ in range(N_ITER):
        for i in range(N_MAX_BBOXES):
            x1, y1, x2, y2 = bboxes_list[i]
            cv2.resize(image[y1:y2, x1:x2], (out_image_size.width, out_image_size.height))
    opencv_time = 1000/N_ITER*(time.time()-t1)
    print(f"OpenCV Time : {opencv_time}ms/iter over {N_ITER} iterations")
    
    print(f"Speedup : {opencv_time/cuda_time}")
    
    


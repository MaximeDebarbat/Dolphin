
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule

import numpy as np

import os
import sys
sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CUDA_BASE, CUDA_Binding
from Data import ImageSize, BoundingBox

class CuRescaleBbox(CUDA_BASE):
    
    __CUDA_RESCALEBBOX_FILE_NAME = "rescalebbox.cu"
    __CUDA_RESCALEBBOX_FCT_NAME = "rescalebbox"
    
    def __init__(self, in_image_size:ImageSize,
                       rescaled_image_size:ImageSize,
                       n_max_bboxes:int):
        
        super().__init__()
        
        if(n_max_bboxes<=1):
            raise AssertionError(f"n_max_bboxes argument should be >=1. Here n_max_bboxes={n_max_bboxes}.")
                
        self._in_image_size = in_image_size
        self._rescaled_image_size = rescaled_image_size
        self._n_max_bboxes = n_max_bboxes
        
        # Here, we import and compile self.__CUDA_FILE_NAME
        self.__RB_cuda_SM = open(os.path.join(os.path.split(os.path.abspath(__file__))[0],"cuda",self.__CUDA_RESCALEBBOX_FILE_NAME),"rt")
        self.__RB_cuda_SM = SourceModule(self.__RB_cuda_SM.read())
        self._RB_CUDA_F = self.__RB_cuda_SM.get_function(self.__CUDA_RESCALEBBOX_FCT_NAME)
        
        self._binding_in_image_size = CUDA_Binding()
        self._binding_rescaled_image_size = CUDA_Binding()

        self._binding_in_image_size.allocate(shape=(3,), dtype=self._in_image_size.dtype)
        self._binding_rescaled_image_size.allocate(shape=(3,), dtype=self._in_image_size.dtype)

        ######## 
        # COPY #
        ########
        
        self._binding_in_image_size.write(data=self._in_image_size.ndarray)
        self._binding_in_image_size.H2D()
                                    
        self._binding_rescaled_image_size.write(data=self._rescaled_image_size.ndarray)
        self._binding_rescaled_image_size.H2D()
                                    
    def __call__(self, binding_bounding_boxes:CUDA_Binding,
                       binding_out_bboxes:CUDA_Binding,
                       stream:cuda.Stream=None
                       ) -> None:
        '''
        We assume the input image is the original image 
        or a processed image that already is on the device
        as well as its size
        '''
        
        self._RB_CUDA_F(binding_bounding_boxes.device,
                        binding_out_bboxes.device,
                        self._binding_in_image_size.device,
                        self._binding_rescaled_image_size.device,
                        block=(self._n_max_bboxes,1,1), 
                        grid=(1,1), stream=stream)
        
        
if __name__ == "__main__":
    #
    # TO FIX OUTPUT IS AN ARGUMENT NOW
    #
    
    
    in_image_size = ImageSize(width=1920, height=1080, channels=3, dtype=np.uint16)
    rescale_image_size = ImageSize(width=640, height=640, channels=3, dtype=np.uint16)
    
    stream = None #
    n_bboxes = 3
    
    bboxes = [BoundingBox(x0=100,y0=100, x1=600, y1=600, relative=False), BoundingBox(x0=0,y0=0, x1=640, y1=640, relative=False)]
    
    print(bboxes)
    
    bboxes_binding = CUDA_Binding()
    bboxes_binding.allocate(shape=(n_bboxes, 4), dtype=np.uint16)
    bboxes_binding.write(data=np.array([e.ndarray for e in bboxes]))
    bboxes_binding.H2D(stream=stream)

    rescaler = CuRescaleBbox(in_image_size=ImageSize(1920,1080, 3, np.uint16), 
                             rescaled_image_size=ImageSize(640,640,3,np.uint16),
                             n_max_bboxes= n_bboxes)
    
    rescaler(binding_bounding_boxes=bboxes_binding, stream=stream)
    
    out = rescaler.outBoundingBoxes
    out.D2H(stream=stream)
    
    print(out.value)


import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule

import numpy as np

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
        
        self._binding_image_batch = CUDA_Binding()
        self._binding_n_max_bboxes = CUDA_Binding()
        self._binding_out_image_size = CUDA_Binding()
        self._binding_max_width = CUDA_Binding()
        self._binding_max_height = CUDA_Binding()
        
        self._binding_image_batch.allocate(shape=(self._n_max_bboxes,)+self._out_image_size.shape, dtype=self._out_image_size.dtype)
        self._binding_n_max_bboxes.allocate(shape=(), dtype=np.uint16)
        self._binding_out_image_size.allocate(shape=(4,), dtype=self._out_image_size.dtype)
        self._binding_max_width.allocate(shape=(), dtype=np.float32)
        self._binding_max_height.allocate(shape=(), dtype=np.float32)
                
        ######## 
        # COPY #
        ########
        
        self._binding_out_image_size.write(data=self._out_image_size.shape)
        self._binding_out_image_size.H2D()
        
        self._binding_n_max_bboxes.write(data=self._n_max_bboxes)
        self._binding_n_max_bboxes.H2D()
                
        self._BLOCK = self._GET_BLOCK_X_Y(Z=self._n_max_bboxes)
            
    def __call__(self, binding_in_image:CUDA_Binding, 
                       binding_in_image_size:CUDA_Binding,
                       binding_bounding_boxes:CUDA_Binding,
                       stream:cuda.Stream=None
                       ) -> None:
        '''
        We assume the input image is the original image 
        or a processed image that already is on the device
        as well as its size
        '''
        
        self._GMD_CUDA_F(binding_bounding_boxes.device, 
                         self._binding_n_max_bboxes.device, 
                         self._binding_max_width.device, 
                         self._binding_max_height.device,
                         block=(self._n_max_bboxes,1,1), 
                         grid=(1,1), stream=stream)
        
        
        self._binding_max_height.D2H(stream=stream)
        self._binding_max_width.D2H(stream=stream)       
        
        print(f"max_width,max_height = ({self._binding_max_width.value},{self._binding_max_height.value})")
        
        _GRID = self._GET_GRID_SIZE(size=int(self._n_max_bboxes*3*self._binding_max_width.value*self._binding_max_height.value),
                                    block=self._BLOCK)
        
        self._CNR_CUDA_F(binding_in_image.device,           #uint16_t *src_image
                         self._binding_image_batch.device,         #uint16_t* dst_images
                         binding_in_image_size.device,      #uint16_t* in_size
                         self._binding_out_image_size.device,      #uint16_t* out_sizes
                         binding_bounding_boxes.device,            #float** bounding_boxes
                         block=self._BLOCK, 
                         grid=_GRID, 
                         stream=stream)
    
    @property
    def outImageBatch(self) -> CUDA_Binding:
        return self._binding_image_batch
    
    
if __name__ == "__main__":
    
    binding_in_image = CUDA_Binding()
    binding_in_image_size = CUDA_Binding()
    binding_bounding_boxes = CUDA_Binding()
    
    stream = None # cuda.Stream()

    binding_in_image.allocate(shape=(3,1920,1080), dtype=np.uint8)
    binding_in_image.write(np.zeros((3,1920,1080)))
    binding_in_image.H2D(stream=stream)
    
    binding_in_image_size.allocate(shape=(4,), dtype=np.uint16)
    binding_in_image_size.write((3,1000,1000))
    binding_in_image_size.H2D(stream=stream)
    
    binding_bounding_boxes.allocate(shape=(3,4), dtype=np.float32)
    binding_bounding_boxes.write([[0,0,0,0],[0,0,10,100],[0,0,100,10]])
    binding_bounding_boxes.H2D(stream=stream)
        
    cropnrezie = CuCropNResize(out_image_size=ImageSize(width=1920, height=1080, channels=3, dtype=np.uint8), n_max_bboxes=3)
    
    cropnrezie(binding_in_image=binding_in_image,
               binding_in_image_size=binding_in_image_size,
               binding_bounding_boxes=binding_bounding_boxes, 
               stream=stream)


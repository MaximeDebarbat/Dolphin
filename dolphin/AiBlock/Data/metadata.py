from dataclasses import dataclass
import numpy as np

@dataclass
class ImageSize:
    '''
    ImageSize defines a generic dataclass in order to
    formally manipulate image sizes  
    '''
    
    width:int
    height:int
    channels:int
    dtype:np.dtype
    
    @property
    def shape(self):
        return (self.width, self.height, self.channels)
        
    @classmethod
    def itemsize(self) -> int:
        '''
        returns datatype(width) + datatype(height) + datatype(channels)
            ->  uint16 + uint16 + uint16
            ->  48
        '''
        
        return 48
    
    @property
    def imagebytesize(self) -> int:
        '''
        returns width x height x channels x size(dtype)
        '''
        return int(self.width*self.height*self.channels*np.dtype(self.dtype).itemsize)
    
    @property
    def ndarray(self) -> np.ndarray:
        '''
        Creates a native numpy array
        '''
        
        return np.array([self.channels, self.width, self.height]).flatten(order='C').astype(self.dtype)
        
        
    @property
    def host_ptr(self) -> int:
        '''
        Returns the pointer of this array
            np.ndarray.__array_interface__['data'] -> (pointer:int, read_only_mode:bool)
        '''
        
        return self.ndarray.__array_interface__['data'][0]
    
@dataclass
class BoundingBox:
    '''
    BoundingBox defines a generic dataclass in order to
    formallymanipulate Bounding Boxes, can be relative, 
    within [0..1] or absolute, within [0...{WIDTH, HEIGHT}]
    and will only be defined on 16 bytes (float16 or uint16)
    '''
    
    x0:object
    y0:object
    x1:object
    y1:object
    relative:bool
    
    @classmethod
    def relativeBoundingBox(bbox:'BoundingBox', size:ImageSize) -> 'BoundingBox':
        '''
        Relative Bounding Box means that we rescale the coordinates betweeen [0..1] 
        in order to adapt this bounding box to any size of image. 
        '''
        
        return BoundingBox(
            x0=bbox.x0/size.width,
            y0=bbox.y0/size.height,
            x1=bbox.x1/size.width,
            y1=bbox.y1/size.height,
            relative=True
        )
        
    @classmethod
    def itemsize(self) -> int:
        '''
        returns datatype(x0) + datatype(y0) + datatype(x1) + datatype(y1) 
            ->  uint16/float16 + uint16/float16 + uint16/float16 + uint16/float16
            ->  64
            
        We assume here that relative coordinates will be float16 
        and absolute coordinates would be uint16
        '''
        
        return int(64)
        
    @property
    def ndarray(self) -> np.ndarray:
        '''
        returns a native numpy array
        '''
        
        if(self.relative):
            dtype = np.dtype(np.float16)
        else:
            dtype = np.dtype(np.uint16)
        return np.array([self.x0, self.y0, self.x1, self.y1]).astype(dtype)
    
    @property
    def host_ptr(self) -> int:
        '''
        Returns the pointer of this array
            np.ndarray.__array_interface__['data'] -> (pointer:int, read_only_mode:bool)
        '''
        
        return self.ndarray.__array_interface__['data'][0]
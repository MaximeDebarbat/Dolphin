
import numpy as np
from functools import reduce

class Data:
    
    def __init__(self, name:str, 
                       shape:tuple,
                       dtype:np.dtype):
        
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._size = reduce((lambda x, y: x.size * y.size), self._shape)*np.dtype(self._dtype)
        
    @property
    def shape(self) -> tuple:
        return self._shape
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def dtype(self) -> np.dtype:
        return self._dtype
    
    @property
    def size(self) -> int:
        return self._size
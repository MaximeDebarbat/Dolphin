
import numpy as np
from functools import reduce
from Data.device import Device

class Data:
    
    def __init__(self, name:str, 
                       shape:tuple,
                       value:object,
                       dtype:np.dtype,
                       device:Device):
        
        self._name = name
        self._shape = shape
        self._value = value
        self._dtype = dtype
        self._device = device
        self._size = reduce((lambda x, y: x.size * y.size), self._shape)*np.dtype(self._dtype)
    
    @property
    def shape(self) -> tuple:
        return self._shape
    
    @property
    def value(self) -> object:
        return self._value
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def dtype(self) -> np.dtype:
        return self._dtype
    
    @property
    def size(self) -> int:
        return self._size
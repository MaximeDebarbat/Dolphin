
from .data import Data
from typing import List
from functools import reduce

class TransitData:
    
    def __init__(self, transitData:List[Data]):
        
        self._transitData = {}
        
        for elt in transitData:
            self._transitData[elt.name] = elt
    
        self._size = reduce((lambda x, y: x.size + y.size), self._transitData.values())
    
    def __getitem__(self, key:str) -> Data:
        
        return self._transitData[key]
    
    @property
    def size(self) -> int:
        return self._size
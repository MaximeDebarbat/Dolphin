
import pycuda.autoinit
import pycuda.driver as cuda

from typing import List, callable
from AiBlock.TrtWrapper import TRTInference
from AiBlock import TransitData

class Block(object):
    
    def __init__(self, 
                 Preprocessors:List[callable],
                 Postprocessors:List[callable],
                 TrtEngine:TRTInference,
                 stream:cuda.Stream | None = None):
        
        self._Preprocessors = Preprocessors
        self._Postprocessors = Postprocessors
        self._TrtEngine = TrtEngine
        if(stream is None):
            self._stream = cuda.Stream()
        else:
            self._stream = stream
    
    def __call__(self, input:object) -> TransitData:
        
        x = input
        for __prp in self._Preprocessors:
            x = __prp(x, stream=self._stream)
            
        self._stream.synchronize()
        
        x = self._TrtEngine(x, stream=self._stream)
        
        self._stream.synchronize()
        
        for __ptp in self._Postprocessors:
            x = __ptp(x, stream=self._stream)
        
        self._stream.synchronize()
            
        return x
    
    @property
    def stream(self):
        return self._stream
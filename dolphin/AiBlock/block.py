
from typing import List, callable
from AiBlock.TrtWrapper import TRTInference 
from AiBlock.CudaUtils import bindings
from AiBlock import TransitData

class Block(object):
    
    def __init__(self, 
                 Preprocessors:List[callable],
                 Postprocessors:List[callable],
                 TrtEngine:TRTInference,
                 cuda_bindings:bindings):
        
        self._Preprocessors = Preprocessors
        self._Postprocessors = Postprocessors
        self._TrtEngine = TrtEngine
        self._cuda_bindings = cuda_bindings
    
    def __call__(self, input:TransitData) -> TransitData:
        
        
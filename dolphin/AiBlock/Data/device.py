
from typing import Union

class Device:
    
    CUDA=0
    CPU=1
    
    def __init__(self, device:Union[str,int]):
        if((isinstance(device, str) and device=="cuda") or (isinstance(device, int) and device==0)):
            self._device = self.CUDA
        elif((isinstance(device, str) and device=="cpu") or (isinstance(device, int) and device==1)):
            self._device = self.CPU
        else:
            raise ValueError(f"Error while creating AiBloc.Data.Device object. {device} is not recognised.")
    
    @property
    def cpu(self) -> bool:
        return self._device==self.CPU

    @property
    def gpu(self) -> bool:
        return self._device==self.CUDA
    
    @property
    def cuda(self) -> bool:
        return self.gpu
        
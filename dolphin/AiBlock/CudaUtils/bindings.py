import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
from functools import reduce


class HostDeviceMem(object):
    """
    This class is used to allocate memory on the host and the device.
    
    :param host_mem: Allocated memory on the host
    :type host_mem: np.ndarray
    :param device_mem: Allocated memory on the device
    :type device_mem: cuda.DeviceAllocation
    """

    def __init__(self, host_mem:np.ndarray, device_mem:cuda.DeviceAllocation):
        self.host = host_mem
        self.device = device_mem

    def __repr__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)


class CUDA_Binding(object):
    """
    This class is used to allocate memory on the host and the device.
    It is also used to copy data from the host to the device and vice versa.
    """
    
    def __init__(self):
        """
        Constructor of the class
        """
        
        self._HDM = None
        self._shape = None
        self._dtype = None
        self._nbytes = 0
        self._size = 0
                
    def allocate(self, shape:tuple, dtype:np.dtype):
        """
        This function allocates the memory on the host and the device.
        
        :param shape: Shape of the memory to be allocated
        :type shape: tuple
        :param dtype: Data type of the memory to be allocated
        :type dtype: np.dtype
        """
        
        self._shape = shape
        self._dtype = dtype
        
        _HM = np.empty(trt.volume(self._shape), self._dtype)
        _DM = cuda.mem_alloc(_HM.nbytes)
        
        self._nbytes=_HM.nbytes
        self._HDM = HostDeviceMem(host_mem = _HM,
                                  device_mem= _DM)
        
        self._size = reduce((lambda x, y: x * y), self._shape)
        
    def write(self, data:object):
        """
        This function copies the data from the host to the device.
        
        :param data: Data to be copied
        :type data: object
        """
        self._HDM.host = np.array(data, dtype=self._dtype, order="C")
    
    def H2D(self, stream:cuda.Stream=None):
        """
        This function copies the data from the host to the device.
        If a stream is provided, the copy will be done asynchronously.
        
        :param stream: Cuda Stream in order to perform asynchronous copy , defaults to None
        :type stream: cuda.Stream, optional
        """
        if(stream is None):
            cuda.memcpy_htod(self._HDM.device,self._HDM.host)
        else:
            cuda.memcpy_htod_async(self._HDM.device,self._HDM.host, stream=stream)

            
    def D2H(self, stream:cuda.Stream=None):
        """
        This function copies the data from the device to the host.
        If a stream is provided, the copy will be done asynchronously.
        
        :param stream: Cuda Stream in order to perform asynchronous copy , defaults to None
        :type stream: cuda.Stream, optional
        """
        
        if(stream is None):
            cuda.memcpy_dtoh(self._HDM.host,self._HDM.device)
        else:
            cuda.memcpy_dtoh_async(self._HDM.host,self._HDM.device, stream=stream)
    
    def __del__(self):
        """
        This function is called when the object is destroyed.
        It is used to free the device memory.
        """
        
        try:
            self._HDM.device.free()
            del self._HDM.device
            del self._HDM.host
        except Exception as e:
            print(f"Encountered Exception while destroying object {self.__class__} : {e}")
    
    @property
    def size(self)->int:
        """
        Returns the number of elements in the buffer
        
        :return: number of elements in the buffer
        :rtype: int
        """
        return self._size
    
    @property
    def nbytes(self)->int:
        """
        Returns the number of bytes in the buffer
        
        :return: number of bytes in the buffer
        :rtype: int
        """
        return self._size
      
    @property
    def host(self)->np.ndarray:
        """
        This is a property that returns the host memory of the buffer as a np.ndarray object.
        The host memory is the value of the buffer on the host memory.
        
        :return: Pointer to host memory
        :rtype: np.ndarray
        """
        return self._HDM.host

    @property
    def device(self)->cuda.DeviceAllocation:
        """
        This is a property that returns the device memory of the buffer as a cuda.DeviceAllocation object.
        The device memory is the value of the buffer on the device memory.

        Returns:
            cuda.DeviceAllocation: Pointer to device memory
        """
        return self._HDM.device
    
    @property
    def shape(self)->tuple:
        """
        Returns the shape of the buffer

        Returns:
            tuple: The shape of the buffer
        """
        return self._shape
    
    @property
    def dtype(self)->np.dtype:
        """
        Returns the dtype of the buffer

        Returns:
            np.dtype: The dtype of the buffer
        """
        return self._dtype
    
    @property
    def value(self)->np.ndarray:
        """
        This is a property that returns the value of the buffer as a numpy array.
        The value is the value of the buffer on the host memory.

        Returns:
            np.ndarray: The value of the buffer on the host memory
        """
        return self._HDM.host.reshape(self._shape).astype(self._dtype)
    
class CUDA_Buffers(object):

    # TO FIX, GENERALIZE THIS SHIT WITH CUDA_Binding

    def __init__(self):

        self._inputs = {}
        self._outputs = {}

        self._output_shapes = {}
        self._input_shapes = {}
        
        self._input_order = []
        self._output_order = []

    def allocate_input(self, name:str, shape:tuple, dtype:object):

        self._input_shapes[name] = shape
        
        host_mem = cuda.pagelocked_empty(trt.volume(shape), trt.nptype(dtype))
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        self._inputs[name] = HostDeviceMem(host_mem, device_mem)
        
        self._input_order.append(name)

    def allocate_output(self, name:str, shape:tuple, dtype:object):

        self._output_shapes[name] = shape
        
        host_mem = cuda.pagelocked_empty(trt.volume(shape), trt.nptype(dtype))
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        self._outputs[name] = HostDeviceMem(host_mem, device_mem)
        
        self._output_order.append(name)

    ### Async copy

    def input_H2D_async(self, stream:cuda.Stream):
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in self._inputs]

    def input_D2H_async(self, stream:cuda.Stream):
        [cuda.memcpy_dtoh_async(inp.host, inp.device, stream) for inp in self._inputs]

    def output_H2D_async(self, stream:cuda.Stream):
        [cuda.memcpy_htod_async(out.device, out.host, stream) for out in self._outputs]

    def output_D2H_async(self, stream:cuda.Stream):
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in self._outputs]

    ### Sync copy

    def input_H2D(self):
        [cuda.memcpy_htod(inp.device, inp.host) for inp in self._inputs]

    def input_D2H(self):
        [cuda.memcpy_dtoh(inp.host, inp.device) for inp in self._inputs]

    def output_H2D(self):
        [cuda.memcpy_htod(out.device, out.host) for out in self._outputs]

    def output_D2H(self):
        [cuda.memcpy_dtoh(out.host, out.device) for out in self._outputs]

    def write_input_host(self,name:str, data:object):
        self._inputs[name].host = np.array(data, dtype=np.float32, order='C')
        return np.asarray(data).shape

    @property
    def input_shape(self):
        return self._input_shapes

    @property
    def output_shape(self):
        return self._output_shapes

    @property
    def output(self):
        return [out.host.reshape(shape) for shape, out in zip(self._output_shapes,self._outputs)]

    @property
    def input_bindings(self):
        return [self._inputs[name].device for name in self._input_order]

    @property
    def output_bindings(self):
        return [self._outputs[name].device for name in self._output_order]

    @property
    def bindings(self):
        return self.input_bindings + self.output_bindings

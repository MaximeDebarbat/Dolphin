import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np

class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __repr__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

class CUDA_Buffers(object):

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

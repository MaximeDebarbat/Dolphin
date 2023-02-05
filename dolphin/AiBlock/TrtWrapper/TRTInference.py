import numpy as np
import tensorrt as trt
import os
import sys


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file, logger):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.TRT_LOGGER = logger
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None
        self.numMaxBatch = 8
        self.__counter=0
        self.calib_shape = None

    def set_image_batcher(self, calib_shape:tuple, calib_dtype:object, numMaxBatch:int):
        """
        """

        self.calib_shape = calib_shape
        self.calib_dtype = calib_dtype
        if(numMaxBatch!=None):
            self.numMaxBatch = numMaxBatch

        size = int(np.dtype(self.calib_dtype).itemsize * np.prod(self.calib_shape))
        
        self.batch_allocation = cuda.mem_alloc(size)

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        
        :return: 1. Since we don't want to create the calibration file. In case, 
        no calibration file is required and we simply want to test the exportability 
        of a certain model, it is still useful.
        """

        if(type(self.calib_shape)==trt.tensorrt.Dims):
            return self.calib_shape[0] 
        return 1 

    def get_batch(self, names:object):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: None since we don't want to create the calibration file. In case, 
        no calibration file is required and we simply want to test the exportability 
        of a certain model, it is still useful
        """
        batch = np.random.rand(*self.calib_shape).astype(self.calib_dtype) 
        cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))

        if(self.__counter>self.numMaxBatch):
            return None
        self.__counter+=1
        return [int(self.batch_allocation)]

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file!=None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.TRT_LOGGER.log(self.TRT_LOGGER.Severity.INFO,"Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        pass


class TrtInference:
    def __init__(self, onnx_file_path:str, 
                       load_engine_path:str=None,
                       save_engine_path:str=None,
                       mode:str="fp16", 
                       optimisation_profile=None, 
                       verbosity:bool=False, 
                       layerProfile:str=None, 
                       profile:str=None, 
                       explicit_batch:bool=False,
                       directIO:bool=False,
                       stdout:object=sys.stdout,
                       stderr:object=sys.stderr,
                       calib_cache:str=None,
                       numMaxBatch:int=8
                       ):

        self.stdout = stdout
        self.stderr = stderr
        self.calib_cache = calib_cache
        self.numMaxBatch = numMaxBatch

        self.layerProfile=layerProfile
        self.profile=profile
        self.explicit_batch = explicit_batch
        self.directIO = directIO

        self.TRT_LOGGER = CustomLogger(verbose_mode=verbosity, stdout=self.stdout, stderr=self.stderr)

        self.runtime = trt.Runtime(self.TRT_LOGGER)

        self.load_engine_path = load_engine_path
        self.save_engine_path = save_engine_path
        
        self.optimisation_profile = optimisation_profile

        if(mode=="int8" and self.calib_cache!=None and not os.path.exists(self.calib_cache)):
            self.TRT_LOGGER.log(self.TRT_LOGGER.Severity.ERROR, f"The calibration cache file '{self.calib_cache}' is unusable.")
            sys.exit(-1)    

        if self.load_engine_path!=None:
            self.engine = self.load_engine(self.load_engine_path)
        else:
            self.engine = self.build_engine(
                onnx_file_path, self.save_engine_path, mode=mode)
        
        if(self.engine==None):
            self.TRT_LOGGER.log(TRT_LOGGER.Severity.ERROR, f"Failed to build or to read a tensorRT Engine. Check the logs for more details.")
            sys.exit(-1)
        
        try:
            if self.engine is not None:
                self.context = self.create_context(self.engine)
                self.buffers = self.allocate_buffers()
        except Exception as e:
            self.TRT_LOGGER.log(TRT_LOGGER.Severity.ERROR, f"Allocation error : {e}")
            sys.exit(-1)

        if(int(trt.__version__[0])==8 and int(trt.__version__[2])>=2 and self.profile!=None):
            inspector = self.engine.create_engine_inspector()
            with open(self.profile,'w') as _f_profile:
                engine_p = inspector.get_engine_information(trt.tensorrt.LayerInformationFormat.JSON)
                _f_profile.write(engine_p)

        if(int(trt.__version__[0])==8 and int(trt.__version__[2])>=2 and self.layerProfile!=None):
            inspector = self.engine.create_engine_inspector()
            with open(self.layerProfile,'w') as _f_layerProfile:
                layer_p = inspector.get_layer_information(0, trt.tensorrt.LayerInformationFormat.JSON)
                _f_layerProfile.write(layer_p)

        self._outputs = {}
        for binding in self.engine:
            if(not self.engine.binding_is_input(binding)):
                self._outputs[binding] = self.engine.get_binding_shape(binding)

    def __del__(self):
        try:
            del self.buffers
            del self.context
        except:
            pass

    @property
    def outputs(self):
        return self._outputs

    def create_context(self, engine):
        pass

    @property
    def shape(self):
        return self.engine.get_binding_shape(0)

    def build_engine(self, onnx_file_path, engine_file_path, mode, max_workspace_size, optimisation_profile):
        pass

    def allocate_buffers(self):
        pass

    def load_engine(self, engine_file_path, runtime=None):
        runtime = runtime or self.runtime
        with open(engine_file_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine

    def run(self, batch:object, asynch:bool=False):
        pass

    def do_inference(self, context):

        self.buffers.H2D()
        context.execute_v2(bindings=self.buffers.bindings)
        self.buffers.D2H()

        return self.buffers.output

    def do_inference_async(self, context):

        self.buffers.H2D_async()
        context.execute_async_v2(
            bindings=self.buffers.bindings, stream_handle=self.buffers.stream.handle)
        self.buffers.D2H_async()
        self.buffers.synchronize()

        return self.buffers.output
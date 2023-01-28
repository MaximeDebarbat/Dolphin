import sys

from trt.TRTInference import CUDA_Buffers, TrtInference, EngineCalibrator

import tensorrt as trt
import os

TRT_LOGGER = None

if(int(trt.__version__[2])==0):
    TRT_LOGGER = trt.Logger
else:
    TRT_LOGGER = trt.ILogger

class EInference(TrtInference):

    def __init__(self, onnx_file_path, 
                       load_engine_path=None,
                       save_engine_path=None,
                       mode="fp16",
                       optimisation_profile=None, 
                       verbosity:bool=False, 
                       layerProfile:str=None, 
                       profile:str=None,
                       directIO:bool=False,
                       stdout:object=sys.stdout,
                       stderr:object=sys.stderr,
                       calib_cache:str=None,
                       numMaxBatch:int=8):

        self.input_binding_index_list = []
        self.output_binding_index_list = []

        self.available_shapes = {}

        super().__init__(onnx_file_path, 
                         load_engine_path=load_engine_path,
                         save_engine_path=save_engine_path,
                         mode=mode,
                         optimisation_profile=optimisation_profile, 
                         verbosity=verbosity, 
                         layerProfile=layerProfile, 
                         profile=profile, 
                         directIO=directIO,
                         stdout=stdout,
                         stderr=stderr,
                         calib_cache=calib_cache,
                         numMaxBatch=numMaxBatch)

    @property
    def shapes(self):
        return self.available_shapes

    def build_engine(self, onnx_file_path, engine_file_path, max_workspace_size=30, mode="fp16"):

        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, self.TRT_LOGGER) as parser:
            
            self.TRT_LOGGER.log(TRT_LOGGER.Severity.INFO, "Loading ONNX file: '{}'".format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                parser.parse(model.read())

                if(parser.num_errors>0):
                    for _error_idx in range(parser.num_errors):
                        self.TRT_LOGGER.log(TRT_LOGGER.Severity.ERROR, f"trt.OnnxParser Error {_error_idx} : {parser.get_error(_error_idx)}")
                    sys.exit(-1)

                input_layer_name = network.get_input(0).name

                last_layer = network.get_layer(network.num_layers - 1)

                if not last_layer.get_output(0):
                    network.mark_output(last_layer.get_output(0))
            self.TRT_LOGGER.log(TRT_LOGGER.Severity.INFO, 'Completed parsing of ONNX file')

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << max_workspace_size
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

            profile = builder.create_optimization_profile()
            profile.set_shape(
                input_layer_name,
                self.optimisation_profile[0],
                self.optimisation_profile[1],
                self.optimisation_profile[2])

            config.add_optimization_profile(profile)

            if(self.directIO):
                config.set_flag(trt.BuilderFlag.DIRECT_IO)

            if mode == "fp16" and builder.platform_has_fast_fp16:
                if(not builder.platform_has_fast_fp16):
                    self.TRT_LOGGER.log(TRT_LOGGER.Severity.WARNING, "--fp16 is not supported. build.platform_has_fast_fp16 check failed.")

                self.TRT_LOGGER.log(TRT_LOGGER.Severity.INFO, "converting to fp16")
                config.set_flag(trt.BuilderFlag.FP16)

            elif mode == "int8":
                if(not builder.platform_has_fast_int8):
                    self.TRT_LOGGER.log(TRT_LOGGER.Severity.WARNING, "--int8 is not supported. build.platform_has_fast_int8 check failed.")

                self.TRT_LOGGER.log(TRT_LOGGER.Severity.INFO, "converting to int8")
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = EngineCalibrator(self.calib_cache, logger=self.TRT_LOGGER)
                if(self.calib_cache==None or engine_file_path==None or not os.path.exists(self.calib_cache) or not os.path.exists(engine_file_path)):
                    self.TRT_LOGGER.log(TRT_LOGGER.Severity.INFO, "--int8 mode but no calibration file has been provided. The accuracy might drop.")
                    inputs = [network.get_input(i) for i in range(network.num_inputs)]

                    config.int8_calibrator.set_image_batcher(calib_shape=inputs[0].shape,
                                                             calib_dtype=trt.nptype(inputs[0].dtype),
                                                             numMaxBatch=self.numMaxBatch)

            self.TRT_LOGGER.log(TRT_LOGGER.Severity.INFO, f'Building an Engine...')
            engine = builder.build_engine(network, config)
            if engine is not None and engine_file_path!=None:
                self.TRT_LOGGER.log(TRT_LOGGER.Severity.INFO, f"Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    def create_context(self, engine):
        _tmp_c = engine.create_execution_context()  
        _tmp_c.active_optimization_profile = 0

        for binding in _tmp_c.engine:
            if(_tmp_c.engine.binding_is_input(binding)):
                b_index = _tmp_c.engine.get_binding_index(binding)
                available_shapes = _tmp_c.engine.get_profile_shape(0,b_index)
                self.available_shapes[binding] = available_shapes

        return _tmp_c

    def allocate_buffers(self):
        
        BUFFER = CUDA_Buffers()

        for b_name in self.context.engine:
            _b_index = self.context.engine.get_binding_index(b_name)
            if(self.context.engine.binding_is_input(b_name)):
                self.input_binding_index_list.append(_b_index)
            else:
                self.output_binding_index_list.append(_b_index)
        
        for b_i_idx in self.input_binding_index_list:
            
            binding_profile_shape = self.context.engine.get_profile_shape(0,b_i_idx)
            self.context.set_binding_shape(b_i_idx, binding_profile_shape[-1])
            
            shape = self.context.get_binding_shape(b_i_idx)
            dtype = self.context.engine.get_binding_dtype(b_i_idx)

            BUFFER.allocate_input(shape, dtype)

        for b_o_idx in self.output_binding_index_list:
            shape = self.context.get_binding_shape(b_o_idx)
            dtype = self.context.engine.get_binding_dtype(b_o_idx)

            BUFFER.allocate_output(shape, dtype)
        
        return BUFFER

    def set_context_profile(self,name, shape):
                
        if(tuple(shape) not in self.available_shapes[name]):
            self.TRT_LOGGER.log(TRT_LOGGER.Severity.ERROR, f"{shape} is not in available optimization profiles : {self.available_shapes[name]}")
            sys.exit(-1)

        self.context.set_binding_shape(self.context.engine.get_binding_index(name), shape)

    def run(self, batch:dict, asynch:bool=False):

        for k in batch.keys():
            named_input = batch[k]
            named_shape = self.buffers.write_input(named_input)
            self.set_context_profile(k, named_shape)

        if(asynch):
            output = self.do_inference_async(self.context)
        else:
            output = self.do_inference(self.context)

        res = []
        for index, output_mem in enumerate(output):
            output_shape = self.context.get_binding_shape(self.output_binding_index_list[index])
            res.append(output_mem[tuple([slice(dim) for dim in output_shape])])

        return res


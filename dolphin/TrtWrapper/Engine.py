"""_summary_
"""

import os
import sys
from typing import Dict, Union
import onnx
import tensorrt as trt
import pycuda.driver as cuda

from dolphin import CudaTrtBuffers, darray
import dolphin

from .utils import TrtLogger
from .utils import EEngine, IEngine


class Engine(EEngine, IEngine):
    """_summary_
    """

    def __init__(self, onnx_file_path: str = None,
                 engine_path: str = None,
                 mode: str = "fp16",
                 optimisation_profile=None,
                 verbosity: bool = False,
                 layer_profile: str = None,
                 profile: str = None,
                 explicit_batch: bool = False,
                 direct_io: bool = False,
                 stdout: object = sys.stdout,
                 stderr: object = sys.stderr,
                 calib_cache: str = None
                 ):
        self.stdout = stdout
        self.stderr = stderr
        self.calib_cache = calib_cache

        self.layer_profile = layer_profile
        self.profile = profile
        self.explicit_batch = explicit_batch
        self.direct_io = direct_io

        self.trt_logger = TrtLogger(
            verbose_mode=verbosity,
            stdout=self.stdout,
            stderr=self.stderr)

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.optimisation_profile = optimisation_profile
        self.runtime = trt.Runtime(self.trt_logger)
        if engine_path is not None and os.path.exists(engine_path):
            self.engine = self.load_engine(engine_path)
            self.dynamic = self.engine.has_implicit_batch_dimension
        else:
            if (onnx_file_path is None or not os.path.exists(onnx_file_path)):
                raise ValueError("onnx_file_path is None and no engine \
has been found at engine_path.")

            self.dynamic = self.is_dynamic(onnx_file_path)
            self.engine = self.build_engine(
                onnx_file_path, engine_path, mode=mode)

        if self.engine is None:
            self.trt_logger.log(
                TrtLogger.Severity.ERROR,
                "Failed to build or to read a tensorRT Engine. \
Check the logs for more details.")
            sys.exit(-1)
        self.trt_logger.log(TrtLogger.Severity.VERBOSE,
                            "Creating Execution Context")
        self.context = self.create_context()

        self.trt_logger.log(TrtLogger.Severity.VERBOSE,
                            "Creating Buffers")
        self.buffers = self.allocate_buffers()

    def allocate_buffers(self) -> CudaTrtBuffers:
        """_summary_

        :return: _description_
        :rtype: CudaTrtBuffers
        """

        buffer = CudaTrtBuffers()
        for binding in self.engine:

            shape = self.engine.get_binding_shape(binding)
            dtype = self.engine.get_binding_dtype(binding)

            self.trt_logger.log(TrtLogger.Severity.VERBOSE,
                                f"Allocating buffer for {binding}, \
shape: {shape}, dtype: {dtype}")

            if self.engine.binding_is_input(binding):
                buffer.allocate_input(name=binding,
                                      shape=tuple(shape[1:]),
                                      buffer_size=shape[0],
                                      dtype=dolphin.dtype.from_numpy_dtype(
                                          trt.nptype(dtype)))
            else:
                buffer.allocate_output(name=binding,
                                       shape=tuple(shape),
                                       dtype=dolphin.dtype.from_numpy_dtype(
                                           trt.nptype(dtype)))
        return buffer

    def load_engine(self, engine_file_path: str,
                    runtime=None):
        """_summary_

        :param engine_file_path: _description_
        :type engine_file_path: str
        :param runtime: _description_, defaults to None
        :type runtime: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        self.trt_logger.log(
            TrtLogger.Severity.INFO,
            f"Loading engine from {engine_file_path}")
        runtime = runtime or self.runtime
        with open(engine_file_path, 'rb') as engine_file:
            engine_data = engine_file.read()
        engine = runtime.deserialize_cuda_engine(engine_data)

        return engine

    def is_dynamic(self, onnx_file_path: str) -> bool:
        """_summary_

        :param onnx_file_path: _description_
        :type onnx_file_path: str
        :return: _description_
        :rtype: bool
        """
        onnx_model = onnx.load(onnx_file_path)
        for inputs in onnx_model.graph.input:
            for dim in inputs.type.tensor_type.shape.dim:
                if len(dim.dim_param) > 0:
                    self.trt_logger.log(
                        TrtLogger.Severity.INFO,
                        "Dynamic model detected.")
                    return True
        self.trt_logger.log(
            TrtLogger.Severity.INFO,
            "Static model detected.")
        return False

    def create_context(self) -> trt.IExecutionContext:
        """_summary_

        :param tensorrt_engine: _description_
        :type tensorrt_engine: trt.ICudaEngine
        :return: _description_
        :rtype: trt.IExecutionContext
        """

        if self.dynamic:
            return EEngine.create_context(self)
        else:
            return IEngine.create_context(self)

    def build_engine(self,
                     onnx_file_path: str = None,
                     engine_file_path: str = None,
                     mode: str = "fp16",
                     max_workspace_size: int = 30) -> trt.ICudaEngine:
        """_summary_

        :param onnx_file_path: _description_, defaults to None
        :type onnx_file_path: str, optional
        :param engine_file_path: _description_, defaults to None
        :type engine_file_path: str, optional
        :param mode: _description_, defaults to "fp16"
        :type mode: str, optional
        :param max_workspace_size: _description_, defaults to 30
        :type max_workspace_size: int, optional
        :return: _description_
        :rtype: trt.ICudaEngine
        """

        if self.dynamic:
            return EEngine.build_engine(self,
                                        onnx_file_path,
                                        engine_file_path,
                                        mode,
                                        max_workspace_size)

        return IEngine.build_engine(self,
                                    onnx_file_path,
                                    engine_file_path,
                                    mode,
                                    max_workspace_size)

    def do_inference(self, stream: cuda.Stream = None) -> None:
        """_summary_

        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        """

        if stream is not None:
            self.context.execute_async_v2(bindings=self.buffers.bindings,
                                          stream_handle=stream.handle)
        else:
            self.context.execute_v2(bindings=self.buffers.bindings)

    def infer(self, inputs: Dict[str, darray],
              batched_input: bool = False,
              force_infer: bool = False,
              stream: cuda.Stream = None) -> Union[darray, None]:
        """_summary_

        :param inputs: _description_
        :type inputs: Dict[str, darray]
        :param batched_input: _description_, defaults to False
        :type batched_input: bool, optional
        :param stream: _description_, defaults to None
        :type stream: cuda.Stream, optional
        :return: _description_
        :rtype: darray
        """

        for name in inputs.keys():
            if batched_input:
                self.buffers.append_multiple_input(name, inputs[name])
            else:
                self.buffers.append_one_input(name, inputs[name])

        if self.buffers.full or force_infer:
            self.do_inference(stream)
            self.buffers.flush()
            return self.output

        return None

    @property
    def output(self) -> dict:
        """_summary_

        :return: _description_
        :rtype: dict
        """
        return self.buffers.output

    @property
    def input_shape(self) -> dict:
        """_summary_

        :return: _description_
        :rtype: dict
        """
        return self.buffers.input_shape

    @property
    def input_dtype(self) -> dict:
        """_summary_

        :return: _description_
        :rtype: dict
        """
        return self.buffers.input_dtype

    @property
    def output_shape(self) -> dict:
        """_summary_

        :return: _description_
        :rtype: dict
        """
        return self.buffers.output_shape

    @property
    def output_dtype(self) -> dict:
        """_summary_

        :return: _description_
        :rtype: dict
        """
        return self.buffers.output_dtype

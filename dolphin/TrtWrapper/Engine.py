"""_summary_
"""

import os
import sys
from typing import Dict, Union, Tuple
import onnx
import tensorrt as trt
import pycuda.driver as cuda

from dolphin import CudaTrtBuffers, darray
import dolphin

from .utils import TrtLogger
from .utils import EEngine, IEngine


class Engine(EEngine, IEngine):
    """
    Class to manage TensorRT engines.
    It is able to read an engine from a file or to create one from an
    onnx file.

    This class is using the :class:`CudaTrtBuffers <dolphin.CudaTrtBuffers>`
    class to manage batched buffers. Find more details on the
    documentation of :func:`~dolphin.Engine.infer`.

    TensorRT Github official Repository: https://github.com/NVIDIA/TensorRT \n
    TensorRT official documentation: https://developer.nvidia.com/tensorrt

    :param onnx_file_path: Path to the onnx file to use, defaults to None
    :type onnx_file_path: str, optional
    :param engine_path: Path to the engine to read or to save, defaults to None
    :type engine_path: str, optional
    :param mode: Can be `fp32`, `fp16` or `int8`, defaults to `fp16`
    :type mode: str, optional
    :param optimisation_profile: Tuple defining the optimisation profile
      to use, defaults to None
    :type optimisation_profile: Tuple[int, ...], optional
    :param verbosity: Boolean to activate verbose mode, defaults to False
    :type verbosity: bool, optional
    :param explicit_batch: Sets `explicit_batch` flag, defaults to False
    :type explicit_batch: bool, optional
    :param direct_io: Sets `direct_io` flag, defaults to False
    :type direct_io: bool, optional
    :param stdout: Out stream to write standard output, defaults to sys.stdout
    :type stdout: object, optional
    :param stderr: Out stream to write errors output, defaults to sys.stderr
    :type stderr: object, optional
    :param calib_cache: Where to write or read calibration cache file,
      defaults to None
    :type calib_cache: str, optional
    :raises ValueError: If no engine nor onnx file is provided
    """

    def __init__(self, onnx_file_path: str = None,
                 engine_path: str = None,
                 mode: str = "fp16",
                 optimisation_profile: Tuple[int, ...] = None,
                 verbosity: bool = False,
                 explicit_batch: bool = False,
                 direct_io: bool = False,
                 stdout: object = sys.stdout,
                 stderr: object = sys.stderr,
                 calib_cache: str = None
                 ):

        self.stdout = stdout
        self.stderr = stderr
        self.calib_cache = calib_cache

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
        """Creates buffers for the engine.
        For now, only implicit dimensions are supported.
        Meaning, that dynamic shapes are not supported yet.

        :return: _description_
        :rtype: dolphin.CudaTrtBuffers
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

    def load_engine(self, engine_file_path: str) -> trt.ICudaEngine:
        """
        Loads a tensorRT engine from a file, using TensorRT.Runtime.

        :param engine_file_path: Path to the engine file
        :type engine_file_path: str
        :return: TensorRT engine
        :rtype: trt.ICudaEngine
        """
        self.trt_logger.log(
            TrtLogger.Severity.INFO,
            f"Loading engine from {engine_file_path}")
        with open(engine_file_path, 'rb') as engine_file:
            engine_data = engine_file.read()
        engine = self.runtime.deserialize_cuda_engine(engine_data)

        return engine

    def is_dynamic(self, onnx_file_path: str) -> bool:
        """
        Returns True if the model is dynamic, False otherwise.
        By `dynamic` we mean that the model has at least one input
        with a dynamic dimension.

        :param onnx_file_path: Path to the onnx file
        :type onnx_file_path: str
        :return: True if the model is dynamic, False otherwise
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
        """
        Creates a tensorRT execution context from the engine.

        :param tensorrt_engine: TensorRT engine
        :type tensorrt_engine: trt.ICudaEngine
        :return: TensorRT execution context
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
        """
        Builds a tensorRT engine from an onnx file. If the onnx model
        is detected as dynamic, then a dynamic engine is built, otherwise
        a static engine is built.

        :param onnx_file_path: Path to an onnx file, defaults to None
        :type onnx_file_path: str, optional
        :param engine_file_path: Path to the engine file to save
          ,defaults to None
        :type engine_file_path: str, optional
        :param mode: Datatype mode `fp32`, `fp16` or `int8`,
          defaults to "fp16"
        :type mode: str, optional
        :param max_workspace_size: maximum workspace size to use,
          defaults to 30
        :type max_workspace_size: int, optional
        :return: TensorRT engine
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
        """
        Executes the inference on the engine. This function assumes
        that the buffers are already filled with the input data.

        :param stream: Cuda Stream, defaults to None
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
              stream: cuda.Stream = None) -> Union[
                  Dict[str, dolphin.Bufferizer],
                  None]:
        """
        Method to call to perform inference on the engine. This method
        will automatically fill the buffers with the input data and
        execute the inference if the buffers are full.
        You can still force the inference by setting `force_infer` to True.

        This expected `inputs` argument expects a dictionary of
        :class:`dolphin.darray` objects or a dict of list of
        :class:`dolphin.darray`.
        The keys of the dictionary must match the names
        of the inputs of the model.

        :param inputs: Dictionary of inputs
        :type inputs: Dict[str, darray]
        :param batched_input: Consider input as batched, defaults to False
        :type batched_input: bool, optional
        :param stream: Cuda stream to use, defaults to None
        :type stream: cuda.Stream, optional
        :return: Output of the model
        :rtype: Union[Dict[str, dolphin.Bufferizer], None]
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
    def output(self) -> Dict[str, dolphin.Bufferizer]:
        """
        Returns the output of the :class:`dolphin.CudaTrtBuffers`
        of the engine.

        :return: Output bufferizer of the engine.
        :rtype: Dict[str, dolphin.Bufferizer]
        """
        return self.buffers.output

    @property
    def input_shape(self) -> Dict[str, tuple]:
        """
        Returns the shape of the inputs of the engine.

        :return: Shape of the inputs
        :rtype: dict
        """
        return self.buffers.input_shape

    @property
    def input_dtype(self) -> Dict[str,
                                  dolphin.dtype]:
        """
        Returns the datatype of the inputs of the engine.

        :return: Datatype of the inputs
        :rtype: Dict[str, dolphin.dtype]
        """
        return self.buffers.input_dtype

    @property
    def output_shape(self) -> Dict[str, tuple]:
        """
        Returns the shape of the outputs of the engine.

        :return: Shape of the outputs
        :rtype: dict
        """
        return self.buffers.output_shape

    @property
    def output_dtype(self) -> Dict[str, dolphin.dtype]:
        """
        Returns the datatype of the outputs of the engine.

        :return: Datatype of the outputs
        :rtype: dict
        """
        return self.buffers.output_dtype

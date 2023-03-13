"""_summary_
"""

import os
import sys
import onnx
import tensorrt as trt

from utils import TrtLogger
from utils import EEngine, IEngine

sys.path.append("..")

from CudaUtils import CUDA_Buffers


class Engine(EEngine, IEngine):
    """_summary_
    """

    def __init__(self, onnx_file_path: str,
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

        self.available_shapes = {}

        self.trt_logger = TrtLogger(
            verbose_mode=verbosity,
            stdout=self.stdout,
            stderr=self.stderr)

        self.optimisation_profile = optimisation_profile
        self.runtime = trt.Runtime(self.trt_logger)

        if engine_path is not None and os.path.exists(engine_path):
            self.engine = self.load_engine(engine_path)
        else:
            if (onnx_file_path is None or not os.path.exists(onnx_file_path)):
                raise ValueError("onnx_file_path is None and no engine \
has been found at engine_path.")

            self.dynamic = self.is_dynamic(onnx_file_path)
            self.engine = self.build_engine(
                onnx_file_path, engine_path, mode=mode)

        self.dynamic = self.engine.has_implicit_batch_dimension

        if self.engine is None:
            self.trt_logger.log(
                TrtLogger.Severity.ERROR,
                "Failed to build or to read a tensorRT Engine. \
Check the logs for more details.")
            sys.exit(-1)
        self.context = self.create_context()
        self.buffers = self.allocate_buffers()

    def allocate_buffers(self) -> CUDA_Buffers:
        """_summary_

        :return: _description_
        :rtype: CUDA_Buffers
        """

        if self.dynamic:
            return EEngine.allocate_buffers(self)

        return IEngine.allocate_buffers(self)

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

    def __del__(self):
        try:
            del self.context
            del self.engine
            del self.buffers
        except RuntimeError as e:
            print(f"Encountered error while deleting Engine: {e}")


if __name__ == "__main__":

    static_model = onnx.load("model_static_shape.onnx")
    dynamic_model = onnx.load("model_dynamic_shape.onnx")

    static_engine = Engine(onnx_file_path="model_static_shape.onnx",
                           engine_path="model_static_shape.trt",
                           mode="fp16")

    dynamic_engine = Engine(onnx_file_path="model_dynamic_shape.onnx",
                            engine_path="model_dynamic_shape.trt",
                            mode="fp16",
                            optimisation_profile=[(1, 3, 224, 224),
                                                  (1, 3, 224, 224),
                                                  (1, 3, 224, 224)])


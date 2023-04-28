"""_summary_
"""
import sys
import tensorrt as trt

from .logger import TrtLogger


class EEngine:
    """_summary_
    """

    def build_engine(self,
                     onnx_file_path: str,
                     engine_file_path: str,
                     mode: str = "fp16",
                     max_workspace_size: int = 30,) -> trt.ICudaEngine:
        """_summary_

        :param onnx_file_path: _description_
        :type onnx_file_path: _type_
        :param engine_file_path: _description_
        :type engine_file_path: _type_
        :param max_workspace_size: _description_, defaults to 30
        :type max_workspace_size: int, optional
        :param mode: _description_, defaults to "fp16"
        :type mode: str, optional
        :return: _description_
        :rtype: _type_
        """

        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.
                                    EXPLICIT_BATCH)

        with trt.Builder(self.trt_logger) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, self.trt_logger) as parser:

            self.trt_logger.log(TrtLogger.Severity.INFO,
                                f"Loading ONNX file: {onnx_file_path}")
            with open(onnx_file_path, 'rb') as model:
                parser.parse(model.read())

                if parser.num_errors > 0:
                    for _error_idx in range(parser.num_errors):
                        self.trt_logger.log(
                            TrtLogger.Severity.ERROR,
                            f"trt.OnnxParser Error {_error_idx} : \
{parser.get_error(_error_idx)}")

                    sys.exit(1)

                if not network.get_layer(network.num_layers - 1).get_output(0):
                    network.mark_output(
                        network.get_layer(
                            network.num_layers -
                            1).get_output(0))
            self.trt_logger.log(
                TrtLogger.Severity.INFO,
                'Completed parsing of ONNX file')

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << max_workspace_size
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

            profile = builder.create_optimization_profile()

            if self.optimisation_profile is None:
                raise RuntimeError("optimisation_profile is None")

            profile.set_shape(
                network.get_input(0).name,
                self.optimisation_profile,
                self.optimisation_profile,
                self.optimisation_profile)

            config.add_optimization_profile(profile)

            if self.direct_io:
                config.set_flag(trt.BuilderFlag.DIRECT_IO)

            if mode == "fp16" and builder.platform_has_fast_fp16:
                if not builder.platform_has_fast_fp16:
                    self.trt_logger.log(
                        TrtLogger.Severity.WARNING,
                        "fast fp16 is not supported. \
build.platform_has_fast_fp16 check failed.")

                self.trt_logger.log(
                    TrtLogger.Severity.INFO,
                    "converting to fp16")
                config.set_flag(trt.BuilderFlag.FP16)

            #         INT8          #
            # Needs to be implemented

            # elif mode == "int8":
            #     if not builder.platform_has_fast_int8:
            #         self.trt_logger.log(
            #             TrtLogger.Severity.WARNING,
            #             "--int8 is not supported. \
            # build.platform_has_fast_int8 check failed.")

            #     self.trt_logger.log(
            #         TrtLogger.Severity.INFO,
            #         "converting to int8")
            #     config.set_flag(trt.BuilderFlag.INT8)
            #     config.int8_calibrator = EngineCalibrator(
            #         self.calib_cache, logger=self.trt_logger)
            #     if (self.calib_cache is None or engine_file_path
            # is None or not os.path.exists(
            #             self.calib_cache) or
            # not os.path.exists(engine_file_path)):
            #         self.trt_logger.log(
            #             TrtLogger.Severity.INFO,
            #             "--int8 mode but no calibration
            # file has been provided. The accuracy might drop.")
            #         inputs = [
            #             network.get_input(i) for i in range(
            #                 network.num_inputs)]

            #         config.int8_calibrator.set_image_batcher(
            #             calib_shape=inputs[0].shape, calib_dtype=trt.nptype(
            #                 inputs[0].dtype), numMaxBatch=self.numMaxBatch)

            self.trt_logger.log(
                TrtLogger.Severity.INFO,
                'Building an Engine...')
            engine = builder.build_engine(network, config)
            if engine is not None and engine_file_path is not None:
                self.trt_logger.log(
                    TrtLogger.Severity.INFO,
                    "Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    def create_context(self) -> trt.IExecutionContext:
        """_summary_

        :param engine: _description_, defaults to None
        :type engine: trt.ICudaEngine, optional
        :return: _description_
        :rtype: trt.IExecutionContext
        """

        _tmp_c = self.engine.create_execution_context()
        _tmp_c.active_optimization_profile = 0

        return _tmp_c

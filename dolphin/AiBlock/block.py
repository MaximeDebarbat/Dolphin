
from typing import List, Callable
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda

from AiBlock.TrtWrapper import TRTInference
from AiBlock.processor import CuLetterBox, CuNormalize, \
                              CuHWC2CHW, CuBGR2RGB, NormalizeMode, \
                              CuRescaleBbox
from AiBlock.Data import ImageDimension


class Block(object):
    """_summary_

    :param preprocessors: _description_
    :type preprocessors: List[callable]
    :param postprocessors: _description_
    :type postprocessors: List[callable]
    :param trt_engine: _description_
    :type trt_engine: TRTInference
    :param stream: _description_, defaults to None
    :type stream: cuda.Stream, optional
    """

    def __init__(self,
                 preprocessors: List[Callable],
                 postprocessors: List[Callable],
                 trt_engine: TRTInference,
                 stream: cuda.Stream = None):
        """_summary_
        """

        self._preprocessors = preprocessors
        self._postprocessors = postprocessors
        self._trtengine = trt_engine

        if stream is None:
            self._stream = cuda.Stream()
        else:
            self._stream = stream

    def __call__(self, inp: object) -> None:

        x = inp
        for __prp in self._preprocessors:
            x = __prp(x, stream=self._stream)

        self._stream.synchronize()

        x = self._trtengine(x, stream=self._stream)

        self._stream.synchronize()

        for __ptp in self._postprocessors:
            x = __ptp(x, stream=self._stream)

        self._stream.synchronize()

        return x

    @property
    def stream(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return self._stream


if __name__ == "__main__":

    preprocessors = [
        CuLetterBox(out_image_size=ImageDimension(640, 640, 3, np.float32)),
        CuBGR2RGB(),
        CuHWC2CHW(),
        CuNormalize(norm_type=NormalizeMode._255)
    ]

    block = Block(preprocessors=[preprocessors],
                  postprocessors=[CuRescaleBbox()],
                  trt_engine=TRTInference(engine_path="temp_data/detection.onnx"))
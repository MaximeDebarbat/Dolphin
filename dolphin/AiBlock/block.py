
from typing import List, Callable, Dict
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda

from processor import CuLetterBox, CuNormalize, \
    CuHWC2CHW, CuBGR2RGB, NormalizeMode, \
    CuRescaleBbox
from Data import ImageDimension
from CudaUtils import CudaBinding
from TrtWrapper import Engine

import cv2

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
                 preprocessors: Dict[str, List[Callable]],
                 postprocessors: Dict[str, List[Callable]],
                 trt_engine: Engine,
                 stream: cuda.Stream = None):
        """_summary_
        """

        self._preprocessors = preprocessors
        self._postprocessors = postprocessors
        self._trtengine = trt_engine
        self._engine_input_buffer = {}
        self._engine_input_size_buffer = {}

        if (sorted(self._trtengine.input_shape.keys()) !=
           sorted(self._preprocessors.keys())):
            raise ValueError(f"Mismatch in input shape and preprocessors. \
Found on engine side : {list(self._trtengine.input_shape.keys())} and on \
preprocessors side : {list(self._preprocessors.keys())}")

        for __key in self._trtengine.input_shape.keys():

            self._engine_input_buffer[__key] = CudaBinding()
            self._engine_input_buffer[__key].allocate(
                self._trtengine.input_shape[__key],
                self._trtengine.input_dtype[__key])

            self._engine_input_size_buffer[__key] = CudaBinding()
            self._engine_input_size_buffer[__key].allocate(
                                        (3,),
                                        self._trtengine.input_dtype[__key])
            self._engine_input_size_buffer[__key].write(self._trtengine.input_shape[__key])
            self._engine_input_size_buffer[__key].h2d()

        if stream is None:
            self._stream = cuda.Stream()
        else:
            self._stream = stream

    def __call__(self, inp: Dict[str, CudaBinding]) -> None:
        """_summary_
        """

        inp_bindings = {}
        for __key in inp.keys():
            dim = ImageDimension(width=inp[__key].shape[1],
                                 height=inp[__key].shape[0],
                                 channels=inp[__key].shape[2],
                                 dtype=inp[__key].dtype)
            inp_bindings[__key] = CudaBinding()
            inp_bindings[__key].allocate((3,),
                                         inp[__key].dtype)
            inp_bindings[__key].write(dim.ndarray)

        for __key in self._engine_input_buffer.keys():  # pylint: disable=consider-using-dict-items
            for index, __ptp in enumerate(self._preprocessors[__key]):
                print(index)
                if index == 0:
                    inp_size_binding = inp_bindings[__key]
                else:
                    inp_size_binding = self._engine_input_size_buffer[__key]

                __ptp(in_image_binding=inp[__key],
                      in_image_size_binding=inp_size_binding,
                      out_image_binding=self._engine_input_buffer[__key],
                      stream=self._stream)

        self._engine_input_buffer["images"].d2h(stream=self._stream)
        temp_image = self._engine_input_buffer["images"].value
        print(temp_image.astype(np.uint8).shape)
        cv2.imwrite("temp.jpg", temp_image.transpose(2,1,0).astype(np.uint8))
        print("lol")
        exit(0)
        self._trtengine.infer(inputs=self._engine_input_buffer,
                              stream=self._stream)

        print(self._trtengine.output)

    @property
    def stream(self):
        """_summary_

        :return: _description_
        :rtype: _type_
        """
        return self._stream


if __name__ == "__main__":

    preprocessors = [
        CuLetterBox(out_image_dimension=ImageDimension(640, 640, 3, np.float32)),
        CuBGR2RGB(),
        CuHWC2CHW(),
        CuNormalize(norm_type=NormalizeMode._255)
    ]

    block = Block(preprocessors={"images": preprocessors},
                  postprocessors=[],
                  trt_engine=Engine(onnx_file_path="detection.onnx",
                                    engine_path="detection.engine",
                                    mode="fp16"))

    image = cv2.imread("traffic.jpg")
    image_binding = Dolphin.Image(image)

    image_binding.resize((640, 640), method=Dolphin.ResizeMethod.LETTERBOX) \
                 .channelSwap(Dolphin.ChannelSwap.BGR2RGB) \
                 .transpose(0,1,2) \
                 .normalize(Dolphin.NormalizeMode._255)

    trt_engine=Engine(onnx_file_path="detection.onnx",
                                    engine_path="detection.engine",
                                    mode="fp16")

    trt_engine.infer(inputs={"images": image_binding}, stream=block.stream)

    CuRescaleBBox(trt_engine.output)




    image_binding.allocate(shape=image.shape, dtype=np.uint8)
    image_binding.write(image)
    image_binding.h2d(stream=block.stream)

    block(inp={"images": image_binding})





import os
import sys

import pycuda.driver as cuda  # pylint: disable=import-error
from pycuda.compiler import SourceModule  # pylint: disable=import-error

sys.path.append("..")
sys.path.append("../..")

from CudaUtils import CudaBase, CudaBinding  # pylint: disable=import-error
from Data import ImageDimension  # pylint: disable=import-error


class ImageProcessor(CudaBase):

    _CUDA_FILE_NAME: str = None

    def __init__(self):

        super().__init__()

        self._cuda_sm = open(os.path.join(os.path.split(
            os.path.abspath(__file__))[0], "cuda",
            self._CUDA_FILE_NAME), "rt", encoding="utf-8")

    def forward(self, in_image_binding: CudaBinding,
                in_image_size_binding: CudaBinding,
                out_image_binding: CudaBinding,
                stream: cuda.Stream = None):
        """_summary_

        :raises NotImplementedError: _description_
        """
        raise NotImplementedError

    def __call__(self,
                 in_image_binding: CudaBinding,
                 in_image_size_binding: CudaBinding,
                 out_image_binding: CudaBinding,
                 stream: cuda.Stream = None
                 ) -> None:
        """_summary_
        """

        self.forward(in_image_binding=in_image_binding,
                     in_image_size_binding=in_image_size_binding,
                     out_image_binding=out_image_binding,
                     stream=stream)

import os
from typing import Union
from abc import abstractmethod
from pycuda.driver import module_from_file
from pycuda.compiler import compile


class CompilerBase:
    """
    Base Class for compiler classes.
    This class simply reads the CUDA source code from the file and
    stores it in the _cuda_source attribute.
    """

    __UTILS = [
        "index_transform.cu"
    ]

    def __init__(self,
                 filename: str):

        self._filename: str = filename
        self._cuda_source: str = ""
        self._func: dict = {}

        with open(os.path.join(os.path.split(
                            os.path.abspath(__file__))[0], "cuda",
                            filename),
                  "rt",
                  encoding="utf-8")as f_s:
            self._cuda_source = f_s.read()

    @abstractmethod
    def generate_source(self) -> str:
        """
        Abstract method that has to be implemented by the child class.
        """
        raise NotImplementedError("generate_source method has \
to be implemented")

    def append_utils(self, source: str):
        for util_file in self.__UTILS:
            with open(self.get_source_path(util_file),
                      "rt",
                      encoding="utf-8")as f_s:
                source = f_s.read() + source
        return source

    def try_load_cubin(self) -> Union[None, object]:

        source = self.generate_source()

        cubin_path = self.get_cubin_path(
            self._filename.replace(".cu", ".cubin"))

        if not os.path.exists(cubin_path):
            print(f"Couldn't find cubin file at {cubin_path}. \
Starting compiling... cubin file should be saved \
and loaded next time.")
            cubin = compile(source)
            if not os.path.exists(self.get_cubin_path("")):
                os.mkdir(self.get_cubin_path(""))

            with open(cubin_path, "wb") as cubin_file:
                cubin_file.write(cubin)

        return module_from_file(cubin_path)

    @staticmethod
    def get_source_path(filename: str):
        return os.path.join(os.path.split(
                            os.path.abspath(__file__))[0], "cuda",
                            filename)

    @staticmethod
    def get_cubin_path(filename: str):
        return os.path.join(os.path.split(
                            os.path.abspath(__file__))[0], "cubin",
                            filename)

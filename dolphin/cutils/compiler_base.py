
import os
from pycuda.compiler import DynamicModule


class CompilerBase:
    """
    Base Class for compiler classes.
    This class simply reads the CUDA source code from the file and
    stores it in the _cuda_source attribute.
    """

    __UTILS = [
        "index_transform.cu"
    ]

    def __init__(self, filename: str):

        print(f"Compiling : {filename}")

        with open(os.path.join(os.path.split(
                            os.path.abspath(__file__))[0], "cuda",
                            filename),
                  "rt",
                  encoding="utf-8")as f_s:
            self._cuda_source: str = f_s.read()
        self._func: dict = {}

    def append_utils(self, source: str):
        for util_file in self.__UTILS:
            with open(os.path.join(os.path.split(
                            os.path.abspath(__file__))[0], "cuda",
                            util_file),
                      "rt",
                      encoding="utf-8")as f_s:
                source = f_s.read() + source
        return source


class CompilerBase_Dynamic:
    """
    Base Class for compiler classes.
    This class simply reads the CUDA source code from the file and
    stores it in the _cuda_source attribute.
    """

    __UTILS = [
        "index_transform.cu"
    ]

    def __init__(self, filename: str):

        print(f"Compiling : {filename}")

        with open(os.path.join(os.path.split(
                            os.path.abspath(__file__))[0], "cuda",
                            filename),
                  "rt",
                  encoding="utf-8")as f_s:
            self._cuda_source: str = f_s.read()

        self._func: dict = {}
        self._compiler = DynamicModule()
        self._compiler.add_source(self._cuda_source,
                                  name=filename.replace("cu", "ptx"))
        for file in self.__UTILS:
            with open(os.path.join(os.path.split(
                                os.path.abspath(__file__))[0], "cuda",
                                file),
                      "rt",
                      encoding="utf-8")as f_s:

                _cuda_source: str = f_s.read()

            self._compiler.add_source(_cuda_source, name=file.replace("cu",
                                                                      "ptx"))

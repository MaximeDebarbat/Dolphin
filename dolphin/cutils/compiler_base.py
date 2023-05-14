
import os
import sys
import shutil
from pycuda.driver import module_from_buffer
from pycuda.driver import Context


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

    @staticmethod
    def compile(source: str,
                destination: str = None):

        if destination is None:
            import tempfile
            destination = tempfile.gettempdir()

        file_name: str = source.split("/")[-1].split(".")[0]

        arch: str = "sm_%d%d" % Context.get_device().compute_capability()
        sys_size: int = 64 if sys.maxsize > 2**32 else 32
        options = ["-Xptxas -O3,-v",
                   "-use_fast_math",
                   f"-arch={arch}",
                   "--cubin",
                   f"-m{sys_size}"]

        os.system(f"nvcc {options} {source}")

        shutil.copyfile(file_name + ".cubin",
                        os.path.join(destination,
                                     file_name + ".cubin"))

        return module_from_buffer(source)



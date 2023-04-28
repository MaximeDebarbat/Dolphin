
import os


class CompilerBase:
    """
    Base Class for compiler classes.
    This class simply reads the CUDA source code from the file and
    stores it in the _cuda_source attribute.
    """

    def __init__(self, filename: str):

        print(f"Compiling : {filename}")

        with open(os.path.join(os.path.split(
                            os.path.abspath(__file__))[0], "cuda",
                            filename),
                  "rt",
                  encoding="utf-8")as f_s:
            self._cuda_source: str = f_s.read()

        self._func: dict = {}

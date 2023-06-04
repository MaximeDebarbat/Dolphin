# Dolphin

![Banner](misc/banner.png)

General python package for CUDA accelerated deep learning inference.

- **Documentation** : [ReadTheDoc](https://dolphin-python.readthedocs.io/en/latest/index.html)
- **Source code** : [https://github.com/MaximeDebarbat/Dolphin](https://github.com/MaximeDebarbat/Dolphin)
- **Bug reports** : [https://github.com/MaximeDebarbat/Dolphin/issues](https://github.com/MaximeDebarbat/Dolphin/issues)
- **Getting Starterd** : <a href="https://colab.research.google.com/drive/1RTZI9hJ6a33NtVUYM0esvSg8nzh2MlP2?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

It provides :

- A set of common image processing functions
- A TensorRT wrapper for easy inference
- Speeds up the inference with CUDA and TensorRT
- An easy to use API with Numpy
- A fast N-Dimensional array

Testing :

In order to test the package, you will need the library `pytest` which you can run from the root of the project :
```
pytest
```

## Install

```
pip install dolphin-python
```

## Build

Dolphin can be installed with Pypi (coming soon) or built with Docker which is the recommended way to use it :

```
docker build  -f Dockerfile \
              --rm \
              -t dolphin:latest \
              .
```

## Docker run

Ensure that you have the `nvidia-docker` package installed and run the following command :

```
docker run \
        -it \
        --rm \
        --gpus all \
        -v "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )":"/app" \
        dolphin:latest \
        bash
```

Please note that Dolphin might not work without the `--gpus all` flag or `--runtime nvidia`.

## Acknowledgements

This project could not have been possible without [PyCuda](https://github.com/inducer/pycuda):

> Andreas Klöckner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul Ivanov, Ahmed Fasih, PyCUDA and PyOpenCL: A scripting-based approach to GPU run-time code generation, > Parallel Computing, Volume 38, Issue 3, March 2012, Pages 157-174.

## TODOs

- [ ] Improve `Engine` class in order to support *int8*
- [ ] Use Cython to speed up the code
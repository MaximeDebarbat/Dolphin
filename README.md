# Dolphin

![Banner](misc/banner.png)

General python package for CUDA accelerated deep learning inference.

- **Documentation** : [coming soon]()
- **Source code** : [https://github.com/MaximeDebarbat/Dolphin](https://github.com/MaximeDebarbat/Dolphin)
- **Bug reports** : [https://github.com/MaximeDebarbat/Dolphin/issues](https://github.com/MaximeDebarbat/Dolphin/issues)
- **Contributing** : [coming soon]()

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

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

Please note that Dolphin might not work without the `--gpus all` flag.

## Usage

As Dolphin is built on top of TensorRT, you will need to have a TensorRT installed in your environment to use it. You can find a tutorial on how to create one [here](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#tutorial).

### `dtype`

Dolphin supports the following data types::
- uint8
- int8
- uint16
- int16
- uint32
- int32
- float32
- float64

These dtypes are accessible from the object `dolphin.dtype` or directly from the module `dolphin` :

```python
import dolphin as dp

# Accessing the dtype from the module
type1 = dp.dtype.uint8

# Accessing the dtype from the package
type2 = dp.uint8
```

Also, Dolphin dtypes are perfectly compatible with Numpy dtypes and Cuda dtypes :

```python

import dolphin as dp
import numpy as np

# Create a dolphin dtype from a numpy dtype
type1 = dp.dtype(np.uint8)

# Create a dolphin dtype from the module
type2 = dp.dtype.uint8

# Create a dolphin dtype from the package
type3 = dp.uint8

# Convert a dolphin dtype to a numpy dtype
type4 = type3.numpy_dtype # -> np.uint8 | np.dtype

# Convert a dolphin dtype to a cuda dtype
type5 = type3.cuda_dtype # -> 'uint8_t' | str

# Access the size of a dtype, in bytes
size_type5 = type5.itemsize # -> 1 | int
```

### `darray`

Dolphin implements a few objects and functions to make the inference easier. Here is a quick example of how to use it :

```python

import numpy as np
import dolphin as dp

# Create a darray from zeros
darray = dp.zeros((1, 3, 224, 224))

# Create a darray from ones
darray = dp.ones((1, 3, 224, 224))

# Create a darray from a numpy array
darray = dp.from_numpy(np.ones((1, 3, 224, 224)))

# Create an empty darray
darray = dp.empty((1, 3, 224, 224))

# Create a darray from shape and dtype
darray = dp.darray(shape=(1, 3, 224, 224), dtype=dp.uint8)
```

`darray` are Numpy-like objects, can be used as such and are compatible with Numpy `ndarray` :

```python

import numpy as np
import dolphin as dp

# Create a darray from zeros
darray = dp.zeros((1, 3, 224, 224), dtype=dp.uint8)

# Create a numpy array from the darray, this will copy the data to the host
numpy_array = darray.ndarray

# Copy the data from the host to the device
darray.ndarray = numpy_array
```

`darray` also implements a few CUDA-accelerated functions :

```python

import dolphin as dp

# Create a darray from zeros
darray = dp.zeros((1, 3, 224, 224), dtype=dp.uint8)

# Fill the darray with a value
res = darray.fill(10)

# Divide the darray by a value
res = darray.divide(10)
res = darray / 10
darray /= 10

# Multiply the darray by a value
res = darray.multiply(10)
res = darray * 10
darray *= 10

# Transpose the darray
res = darray.transpose((0, 2, 3, 1))

# Cast the darray to another dtype
res = darray.cast(dp.float32)

```

Note that `darray` does not support dynamic casting in order keep values in the range of the `dtype` defined. For example, if you try to divide a `darray` of `dtype=dp.uint8` by 10, the result can be unexpected, as well as negative values of `dtype=dp.uint8`.

You have 2 ways to copy a `darray`. The first one is to use the `copy` method which will create a new `darray` with the same data as the original one but on a different memory location :

```python

import dolphin as dp

# Create a darray from zeros
darray = dp.zeros((1, 3, 224, 224), dtype=dp.uint8)

# Copy the darray
darray_copy = darray.copy()
```

Or you can use the argument `allocation` of the `darray` constructor to specify the memory location of the new `darray` :

```python

import dolphin as dp

# Create a darray from zeros
darray = dp.zeros((1, 3, 224, 224), dtype=dp.uint8)

# Copy the darray
darray_copy = dp.darray(shape=(1, 3, 224, 224), dtype=dp.uint8, allocation=darray.allocation)
```

In this case `darray` and `darray_copy` will share the same memory location. If you modify the data of one of them, the other one will be modified as well.

### `dimage`

Dolphin implements a particular type of `darray` called `dimage` which is a 2D or 3D array with a specific layout. It is used to store images, properties and image processing functions.

```python

import dolphin as dp

# Create a dimage from zeros
dimage = dp.dimage(shape=(3, 224, 224), dtype=dp.uint8)

# Create a dimage from a numpy array
dimage = dp.dimage.(np.ones((3, 224, 224)))

```

`dimage` contains status dynamically processed about the image during usage :

- `dimage.shape` : The shape of the image
- `dimage.dtype` : The dtype of the image
- `dimage.size` : The size of the image
- `dimage.width` : The width of the image
- `dimage.height` : The height of the image
- `dimage.channels` : The number of channels of the image
- `dimage.allocation` : The memory location of the image
- `dimage.ndarray` : The numpy array of the image
- `dimage.image_channel_format`: The image channel format of the image
- `dimage.image_dim_format`: The image dimension format of the image

It also contains image specific functions :
- `dimage.resize` : Resize the image
- `dimage.resize_padding` : Resize the image with padding in order to keep aspect ratio
- `dimage.cvtColor` : Convert the image to another color space
- `dimage.normalize` : Normalize the image

These functions can be use as such :

```python

import dolphin as dp
import cv2

np_image = cv2.imread('image.jpg')
dimage = dp.dimage(np_image, channel_format=dp.DOLPHIN_GRAY_SCALE)

# Resize the image
dimage = dimage.resize((224, 224))

# Transpose the image
dimage = dimage.transpose((1, 0))

# Convert the image to BGR
dimage = dimage.cvtColor(dp.DOLPHIN_RGB)

# Normalize the image
dimage = dimage.normalize(dp.DOLPHIN_255)
```

Transposing an image will also automatically update the `dimage.image_dim_format` status.

There also exists status Enum classes used by `dimage` :

```python

class dimage_normalize_type(Enum):
    DOLPHIN_MEAN_STD = 0
    DOLPHIN_255 = 1
    DOLPHIN_TF = 2

class dimage_resize_type(Enum):
    # Currently not used
    DOLPHIN_NEAREST = 0
    DOLPHIN_PADDED = 1

class dimage_channel_format(Enum):
    DOLPHIN_RGB = 0
    DOLPHIN_BGR = 1
    DOLPHIN_GRAY_SCALE = 2

class dimage_dim_format(Enum):
    DOLPHIN_CHW: int = 0
    DOLPHIN_HWC: int = 1
    DOLPHIN_HW: int = 2
```

### `Engine`

The `Engine` class is the main class of Dolphin. It is used to create and manage the execution context of the network. It is also used to load and execute the network with TensorRT.

`Engine` objects supports dynamic and static shapes. If you want to use dynamic shapes models, you will have to specify the shape you want to target while creating the object. [coming soon]

Also, `Engine` assumes that data is batched. Thus, an object described later called `bufferizer` is used to efficiently manage incoming data in case of batched which doesn't fit the batch size of the network.

You can still force inference althoug the `bufferizer` object is not full
by using the `force_infer` argument of the `Engine.infer()` method.

`Engine` can take an onnx model path and a TensorRT engine path as input.
If the TensorRT engine does not exist, it will be created from the onnx model,
if it exists, it will be loaded.

You can refer to the [Yolov7-tiny example using dolphin](https://github.com/MaximeDebarbat/Dolphin/example/yolov7-tiny/run_dolphin.py) for a better understanding of how to use `Engine`.

## Optimization

There is a concept of optimization in Dolphin which has to do with memory allocation. If, while doing `darray` or `dimage` operations, the result is not written directly in a pre-allocated memory location, it will be allocated on the fly. This can be problematic if you want to do a lot of operations on the same `darray` or `dimage` as it will create a lot of memory allocations which can slow down the overall performances. To avoid this, you can specify the argument `dst` while using some operations :

```python

import dolphin as dp

# Unefficient way
darray = dp.zeros((1, 3, 224, 224), dtype=dp.uint8)
darray = darray + 1 # Creates a copy on the fly

# Efficient way
darray = dp.zeros((1, 3, 224, 224), dtype=dp.uint8)
darray_res = darray.copy()

darray_res = darray_res.add(1, dst=darray_res) # Reuses the memory location of darray_res
```

This works for all operations that return a `darray` or `dimage` object.

Getting Started with Dolphin
============================

Manipulating :py:class:`dolphin.dtype` :
-----------------------------------------

:py:class:`dolphin.dtype` is a class that represents the data type of a
:py:class:`dolphin.darray` object. It is similar to numpy's dtype. It is used
to create a gate between numpy types and cuda types. It currently supports
the following operations :
    * :py:attr:`dolphin.dtype.uint8`
    * :py:attr:`dolphin.dtype.uint16`
    * :py:attr:`dolphin.dtype.uint32`
    * :py:attr:`dolphin.dtype.int8`
    * :py:attr:`dolphin.dtype.int16`
    * :py:attr:`dolphin.dtype.int32`
    * :py:attr:`dolphin.dtype.float32`
    * :py:attr:`dolphin.dtype.float64`

Creating :py:class:`dolphin.dtype` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to create a :py:class:`dolphin.dtype` object :

.. code-block:: python

    import dolphin as dp
    import numpy as np

    d = dp.dtype.float32
    print(d)  # float

    # Create a dtype from a numpy dtype
    d = dp.dtype.from_numpy_dtype(np.float32)


Manipulating :py:class:`dolphin.darray` :
-----------------------------------------

Creating :py:class:`dolphin.darray` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to create a :py:class:`dolphin.darray` object :

.. code-block:: python

    import dolphin as dp
    import numpy as np

    # Create a darray from a numpy array
    a = np.arange(10).astype(np.float32)
    d = dp.darray(array=a)

    # Create a zero-filled darray
    d = dp.zeros(shape=(10,), dtype=dp.float32)

    # Create an empty darray
    d = dp.empty(shape=(10,), dtype=dp.float32)

    # or
    d = dp.darray(shape=(10,), dtype=dp.float32)

    # Create a zeros darray like another
    d = dp.zeros_like(d)

    # Create an empty darray like another
    d = dp.empty_like(d)

    # Create a ones darray
    d = dp.ones(shape=(10,), dtype=dp.float32)

    # Create a ones darray like another
    d = dp.ones_like(d)


Numpy-Dolphin interoperability :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can convert a :py:class:`dolphin.darray` object to a numpy array using
the method :py:meth:`dolphin.darray.to_numpy`. You can also convert a numpy
array to a :py:class:`dolphin.darray` object using the function :py:func:`dolphin.from_numpy`.

.. code-block:: python

    import dolphin as dp
    import numpy as np

    # numpy to darray using dolphin constructor
    a = np.arange(10).astype(np.float32)
    d = dp.darray(array=a)

    # Convert a darray to a numpy array
    a = d.to_numpy()

    # Convert a numpy array to a darray
    # numpy array and darray need to
    # have the same dtype and shape.
    d = dp.from_numpy(a)

Transpose :py:class:`dolphin.darray` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transpose a :py:class:`dolphin.darray` object is easy and works like numpy.
You can use the method :py:meth:`dolphin.darray.transpose`, the shortcut
:py:attr:`dolphin.darray.T` or the function :py:func:`dolphin.transpose`.

.. code-block:: python

    import dolphin as dp

    d = dp.darray(shape=(4, 3, 2), dtype=dp.float32)
    print(d.shape)  # (4, 3, 2)

    t = d.transpose(1, 0, 2)
    print(d.shape)  # (3, 4, 2)

    # You can also use the shortcut
    t = d.T
    print(d.shape)  # (2, 4, 3)

    # Or dp.transpose
    t = dp.transpose(src=d, axes=(2, 1, 0))

Cast :py:class:`dolphin.darray` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As numpy implements `astype` operation, Dolphin also implements it. You can
use the method :py:meth:`dolphin.darray.astype`. Also, take a look
at :py:class:`dolphin.dtype` to see the supported types.

.. code-block:: python

    import dolphin as dp

    d = dp.darray(shape=(4, 3, 2), dtype=dp.float32)
    print(d.dtype)  # float32

    d = d.astype(dp.int32)
    print(d.dtype)  # int32

Indexing :py:class:`dolphin.darray` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indexing a :py:class:`dolphin.darray` object is easy and works like numpy.

.. code-block:: python

    import dolphin as dp
    import numpy as np

    n = np.random.rand(10, 10).astype(np.float32)
    d = dp.darray(array=n)

    d_1 = d[0:5, 0:5]
    d_2 = d[5:10, 5:10]

Indexing works in read and write mode :

.. code-block:: python

    import dolphin as dp
    import numpy as np

    d = dp.zeros((4, 4))

    d[0:2, 0:2] = 10
    d[2:4, 2:4] = 20

    print(d)
    #  array([[10., 10.,  0.,  0.],
    #         [10., 10.,  0.,  0.],
    #         [ 0.,  0., 20., 20.],
    #         [ 0.,  0., 20., 20.]])

Operations with :py:class:`dolphin.darray` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dolphin implements several operations with :py:class:`dolphin.darray` objects :

.. code-block:: python

    import dolphin as dp

    d = dp.zeros((4, 4))
    z = dp.ones((4, 4))

    # Addition
    d = d + z
    d += 5

    # Subtraction
    d = d - z
    d -= 5

    # Multiplication
    d = d * z
    d *= 5

    # Division
    d = d / z
    d /= 5

Manipulating :py:class:`dolphin.dimage` :
-----------------------------------------

As :py:class:`dolphin.dimage` is a subclass of :py:class:`dolphin.darray`,
you can use all the methods and functions of :py:class:`dolphin.darray`.
On top of that, :py:class:`dolphin.dimage` implements several methods and
functions to manipulate images as well as image specific attributes.

Creating :py:class:`dolphin.dimage` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a :py:class:`dolphin.dimage` object is easy and works like
:py:class:`dolphin.darray`. The difference comes from is the argument
dimage_channel_format. This argument is used to specify the channel format
of the image. It has to be :py:attr:`dolphin.dimage_channel_format`, by default :
py:attr:`dolphin.dimage_channel_format.DOLPHIN_BGR`.

.. code-block:: python

    import dolphin as dp
    import cv2

    image = cv2.imread("your_image.jpg")
    d = dp.dimage(array=image)

    # or
    d = dp.dimage(array=image, channel_format=dp.dimage_channel_format.DOLPHIN_BGR)

Resizing :py:class:`dolphin.dimage` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With **Dolphin**, you can resize a :py:class:`dolphin.dimage` object using
2 methods :py:meth:`dolphin.dimage.resize` and :py:meth:`dolphin.dimage.resize_padding`.
The first one resizes the image without padding. The second one resizes the image
with padding. The padding is computed to keep the aspect ratio of the image.

.. code-block:: python

    import dolphin as dp
    import cv2

    image = cv2.imread("your_image.jpg")
    d = dp.dimage(array=image)

    # Resize without padding
    a = d.resize((100, 100))
    print(a.shape)  # (100, 100, 3)

    # Resize with padding
    b = d.resize_padding((100, 100), padding_value=0)
    print(b.shape)  # (100, 100, 3)


Normalization :py:class:`dolphin.dimage` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With **Dolphin**, you can normalize a :py:class:`dolphin.dimage` object using
the method :py:meth:`dolphin.dimage.normalize`. You have Normalization modes defined
by the `Enum` class :py:class:`dolphin.dimage_normalize_type`. By default, the mode
is :py:attr:`dolphin.dimage_normalize_type.DOLPHIN_255`. This method is optimized.

.. code-block:: python

    import dolphin as dp
    import cv2

    image = cv2.imread("your_image.jpg")
    d = dp.dimage(array=image)

    # image/255
    a = d.normalize(dp.DOLPHIN_255)

    # image/127.5 - 1
    b = d.normalize(dp.DOLPHIN_TF)

    # image - mean/std
    c = d.normalize(dp.DOLPHIN_MEAN_STD, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

Change channel format :py:class:`dolphin.dimage` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The equivalent of `cv2.cvtColor` is :py:meth:`dolphin.dimage.cvtColor` which
converts a :py:class:`dolphin.dimage` object from one channel format to another.
The channel formats are defined by the `Enum` class :py:class:`dolphin.dimage_channel_format`.

.. code-block:: python

    import dolphin as dp
    import cv2

    image = cv2.imread("your_image.jpg")
    d = dp.dimage(array=image)

    a = dp.dimage.cvtColor(d, dp.dimage_channel_format.DOLPHIN_GRAY_SCALE) # BGR to GRAY
    b = dp.dimage.cvtColor(d, dp.dimage_channel_format.DOLPHIN_RGB) # BGR to RGB

Manipulating :py:class:`dolphin.Engine` :
-----------------------------------------

Creating :py:class:`dolphin.Engine` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`dolphin.Engine` is a TensorRT based object. It is used to create, manage
and run TensorRT engines. To create an :py:class:`dolphin.Engine` object, you need
to specify the path to an onnx model or a TensorRT engine. You can also specify
different other arguments in order to customize the engine built.

.. code-block:: python

    import dolphin as dp

    # Create an engine from an onnx model
    engine = dp.Engine(onnx_file_path="your_model.onnx")

    # Create an engine from a TensorRT engine
    engine = dp.Engine(engine_path="your_engine.trt")

    # Create an engine from an onnx model and specify different arguments
    engine = dp.Engine(onnx_file_path="your_model.onnx",
                       engine_path="your_engine.trt",
                       mode="fp16",
                       explicit_batch=True,
                       direct_io=False)


Running a :py:class:`dolphin.Engine` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a :py:class:`dolphin.Engine` is created, you can run it using the method
:py:meth:`dolphin.Engine.infer`. This method takes a dictionary as argument, this dictionary
defines the inputs of the engine. The keys of the dictionary are the names of the inputs of
the engine. The values of the dictionary are :py:class:`dolphin.darray` objects. The method returns
a dictionary with the outputs of the engine or `None` (see below). The keys of the dictionary are
the names of the outputs of the engine. The values of the dictionary are :py:class:`dolphin.darray`.

:py:class:`dolphin.Engine` implements internally :py:class:`dolphin.CudaTrtBuffers` which are used
to efficiently bufferize the inputs of the engine. The purpose is to memory copy between host and device
and to rather do device to device copies which is faster. By default, calling :py:meth:`dolphin.Engine.infer`
will be batch-blocking, meaning that the method will not infer the engine if the buffer is not full, it allows
the user to fill the buffer automatically. You can still force infer with the argument `force_infer=True`.

Here are some use cases of :py:meth:`dolphin.Engine.infer`.

.. code-block:: python

    import dolphin as dp

    engine = dp.Engine(engine_path="your_engine.trt") # batch size = 1

    input_dict = {
        "image": dp.zeros(shape=(224,224,3), dtype=dp.float32)
    }

    output = engine.infer(inputs=input_dict) # The buffer is full, the engine is inferred

    print(output) # {"output": darray(shape=(1000,), dtype=float32)}


In case you want to use a batch size greater than 1.

.. code-block:: python

    import dolphin as dp

    engine = dp.Engine(engine_path="your_engine.trt") # batch size = 2


    input_dict = {
        "image": dp.zeros(shape=(224,224,3), dtype=dp.float32)
    }

    output = engine.infer(inputs=input_dict) # batch-blocking

    print(output) # None

    output = engine.infer(inputs=input_dict) # The buffer is full, the engine is inferred

    print(output) # {"output": darray(shape=(2,1000), dtype=float32)}

    # or you can force infer

    import dolphin as dp

    engine = dp.Engine(engine_path="your_engine.trt") # batch size = 2


    input_dict = {
        "image": dp.zeros(shape=(224,224,3), dtype=dp.float32)
    }

    output = engine.infer(inputs=input_dict, force_infer=True) # batch-blocking

    print(output) # {"output": darray(shape=(2,1000), dtype=float32)}

You can also use batched inferences.

.. code-block:: python

    import dolphin as dp

    engine = dp.Engine(engine_path="your_engine.trt") # batch size = 16


    input_dict = {
        "image": dp.zeros(shape=(16, 224, 224, 3), dtype=dp.float32)
    }

    output = engine.infer(inputs=input_dict) # The buffer is full, the engine is inferred

    print(output) # {"output": darray(shape=(16,1000), dtype=float32)}

Full example
------------

You can go to the `examples` folder to see a full example of how to use the library.
Here, we will go step by step through Yolov7 inference using `Dolphin`.

1. Preprocessing
~~~~~~~~~~~~~~~~

Most of the time, we underestimate the latency of preprocessing and try to find ways to accelerate the inference
part which would make a lot of sense if the bottleneck was indeed the inference time. In reality, in real-time applications,
it often happens that your fps are drastically decreased compared to your expectations due to pre/post processing.
In this example, Yolov7 needs images to be resized using :py:meth:`dp.dimage.resize_padding` method in order to keep the
orginal aspect ratio of the image as well as it needs to be normalized.
A good practice would be to resize your image first before doing any further processings in order to limit
the amount of data processed at a time.

Keep in mind that it is much better to pre-allocate the :py:class:`dp.darray` and :py:class:`dp.dimage` in order not
to perform memory allocation during the core of your application. This is what we will be doing here.

.. code-block:: python

    import cv2
    import dolphin as dp

    stream = cv2.VideoCapture("your_video.mp4")

    # We need to know the size of the frame
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # As OpenCV reads HWC uint8_t images, we allocate the
    # corresponding dp.dimage
    d_frame = dp.dimage(shape=(height, width, 3), dtype=dp.uint8)

    # Yolov7 is processing directly CHW images, we thus have to
    # transpose the array, meaning, pre-allocate where we will
    # store the transposed reordered data
    transposed_frame = dp.dimage(shape=(3, height, width),
                                dtype=dp.uint8,
                                stream=stream)

    # We also pre-allocate the image once resized in order
    # (640, 640) is the size Yolov7 works with
    resized_frame = dp.dimage(shape=(3, 640, 640),
                              dtype=dp.uint8,
                              stream=stream)

    # Once the image is correctly formatted, meaning :
    # 3x640x640 uint8, we need to normalize the image
    # between 0<=image<=1. To do so, we need to use
    # dp.DOLPHIN_255 flag which will write float32
    # data
    inference_frame = dp.dimage(shape=(3, 640, 640),
                                dtype=dp.float32,
                                stream=stream)


2. Inference
~~~~~~~~~~~~~~~~

We thus have pre-allocated `18MB` to speed up the preprocessing by avoiding
on-the-fly allocations. Shall we now go through the inference part of all of this.

.. code-block:: python


    # We now instanciate our AI model as a TensorRT engine
    engine = dp.Engine("your_model.onnx",
                       "your_model.engine",
                       mode="fp16",
                       verbosity=True)

    while(True):
        # We copy the OpenCV frame onto the GPU
        d_frame.from_numpy(frame)

        # We process the frame
        # 1. We transpose the frame and call 'flatten' in order
        # to rearrange the data in memory as expected
        d_frame.transpose(2, 0, 1).flatten(dst=transposed_frame)

        # 2. We perform padding resize
        _, r, dwdh = dp.resize_padding(src=transposed_frame,
                                       shape=(640, 640),
                                       dst=resized_frame)

        # 3. We do channel swapping in order to transform
        # our BGR image into RGB
        dp.cvtColor(src=resized_frame,
                    color_format=dp.DOLPHIN_RGB,
                    dst=resized_frame)

        # 4. We normalize the frame as described just above
        dp.normalize(src=resized_frame,
                     dst=inference_frame,
                     normalize_type=dp.DOLPHIN_255)

        # 5. We finally infer our model
        output = engine.infer({"images": inference_frame})

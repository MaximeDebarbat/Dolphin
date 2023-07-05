Advanced Utilisation of Dolphin
===============================

Dolphin is based on CUDA and therefore it has some limitations that you should be aware of.
For instance, memory allocation, memory copy, kernel launch, etc. are all consuming time that
is perhaps shorter than the time needed by CPU only operations depending on the complexicity
of computation required and the amount of data you want to process.

There is a general rule that you should always keep in mind when using Dolphin: **The more data,
the more complex the computation, the more Dolphin is efficient**.
There are still ways to speed up the execution of Dolphin functions. We will go through a few
of them in this section.

Memory Management
-----------------

You will have noticed that some Dolphin functions have a parameter usually called `dst` which
is optionnal. This parameter is used to specify the destination of the result of the function.
As mentionned in the introduction of this section, memory allocation and memory copy are consuming time,
if you already have allocated a memory space to store a result, you can use it as a destination
and save more time. Memory allocation can represent up to 95% of the total execution time of a function,
it is not negligible.

Let's take an example, we want to perform a addition between two matrices `A` and `B`.
In this first code snippet, we run a naive addition, in the second one we use the `dst` parameter.

Naive approach:

.. code-block:: python

    import dolphin as dp
    import time

    N_ITER = int(1e3)

    a = dp.zeros((100, 100))
    b = dp.ones((100, 100))

    t1 = time.time()
    for i in range(N_ITER)
        c = a + b
    print(f"Naive approach: {1000*(time.time() - t1)/N_ITER}ms/iter")

Optimized approach:

.. code-block:: python

    import dolphin as dp
    import time

    N_ITER = int(1e3)

    a = dp.zeros((100, 100))
    b = dp.ones((100, 100))
    c = dp.zeros((100, 100))

    t1 = time.time()
    for i in range(N_ITER)
        dp.add(src=a, other=b, dst=c)
    print(f"Optimized approach: {1000*(time.time() - t1)/N_ITER}ms/iter")

When your application is based on a loop or the consecutive execution of several functions,
you should always try to use the `dst` parameter to save time. It can really be a game changer in
some cases.

Usage of allocations
--------------------

Dolphin by default allocates CUDA memory and operates on it. Also, any modification made on
a view of this array will imact the array itself. For instance, :py:meth:`dolphin.darray.__getitem__`
returns a view of the current array, any *in-place modification of the values* on it will modify
the array, exactly like Numpy does:

.. code-block:: python

    import dolphin as dp

    a = dp.zeros(shape=(2, 2), dtype=dp.float32)
    a[:,0].fill(1)

    print(a)
    # array([[1., 0.],
    #        [1., 0.]], dtype=float32)

Usage of Cuda Stream
--------------------

Dolphin is based on CUDA and therefore it is possible to use CUDA streams.
To create a cuda stream, you can use the :py:func:`dolphin.Stream` function.
I recommend to read this `pdf <https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf>`_
about CUDA streams and concurrency.

To use CUDA streams with Dolphin, you can use the `stream` parameter of functions and constructors.

.. code-block:: python

    import dolphin as dp

    stream = dp.Stream()

    a = dp.zeros((100, 100), stream=stream)
    b = dp.ones((100, 100), stream=stream)

    c = dp.add(src=a, other=b)

    # Wait for the stream to finish
    stream.synchronize()

    # Do something with c

With :py:meth:`dolphin.Engine.infer` you can provide a stream as an argument.

.. code-block:: python

    import dolphin as dp

    stream = dp.Stream()

    a = dp.zeros((100, 100), stream=stream)
    b = dp.ones((100, 100), stream=stream)

    c = dp.add(src=a, other=b)

    # Wait for the stream to finish
    stream.synchronize()

    # Do something with c

    # Run the inference on the stream
    output = engine.infer(inputs={"input":c}, stream=stream)

    # Wait for the stream to finish
    stream.synchronize()

    # Do something with the output
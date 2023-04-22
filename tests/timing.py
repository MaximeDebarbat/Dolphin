
import dolphin as dp
import time
import numpy

N_ITER = int(1e3)

def darray_timing():

    stream = dp.Stream()

    darray = dp.darray(shape=(3, 640, 640),
                       dtype=dp.uint8,
                       stream=stream)

    nparray = numpy.random.randint(0,
                                   255,
                                   size=(3, 640, 640),
                                   dtype=numpy.uint8)

    darray_res = dp.darray(shape=(640, 640, 3),
                           dtype=dp.uint8,
                           stream=stream)

    t1 = time.time()
    for _ in range(N_ITER):
        darray.ndarray = nparray
        dp.multiply(src=darray, other=10, dst=darray)
        dp.add(src=darray, other=10, dst=darray)
        dp.divide(src=darray, other=15, dst=darray)
        dp.transpose(axes=(1, 2, 0), src=darray, dst=darray_res)

    t2 = time.time() - t1

    print(f"Timing : darray.ndarray = darray.ndarray: \
{1000*(t2/N_ITER)} ms/iter (total: {t2} s)")

    darray = dp.darray(shape=(640, 640, 640),
                       dtype=dp.uint8,
                       stream=stream)

    t1 = time.time()
    for _ in range(N_ITER):
        darray.transpose(2, 0, 1, dst=darray)
    t2 = time.time() - t1

    print(f"Timing : darray.transpose(2, 0, 1): \
{1000*(t2/N_ITER)} ms/iter (total: {t2} s)")

def dimage_timing():
    pass

if __name__ == "__main__":

    darray_timing()
    dimage_timing()
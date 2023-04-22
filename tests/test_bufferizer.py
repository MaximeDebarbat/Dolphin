import pytest
import numpy
import cv2
import dolphin as dp

@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_bufferizer_general:

    @pytest.mark.parametrize("shape", [
                            (10, 10),
                            (10, 10, 3),
                            (3, 10, 10),
                            (10, 10, 2, 2)])
    @pytest.mark.parametrize("buffer_size", [
                            1,
                            10,
                            3
                            ])
    def test_bufferizer_creation(self, dtype, shape, buffer_size):

        bufferizer = dp.Bufferizer(shape=shape,
                                   dtype=dtype,
                                   buffer_size=buffer_size)

        diff = numpy.linalg.norm(bufferizer.darray.ndarray - numpy.zeros(shape=bufferizer.shape,
                                                                        dtype=dtype.numpy_dtype))

        assert bufferizer.element_shape == shape
        assert bufferizer.shape == (buffer_size,)+shape
        assert bufferizer.dtype == dtype
        assert diff < 1e-5

    @pytest.mark.parametrize("shape", [
                            (10, 10),
                            (10, 10, 3),
                            (3, 10, 10),
                            (10, 10, 2, 2)])
    @pytest.mark.parametrize("buffer_size", [
                            1,
                            10,
                            3
                            ])
    def test_bufferizer_flush(self, dtype, shape, buffer_size):

        bufferizer = dp.Bufferizer(shape=shape,
                                   dtype=dtype,
                                   buffer_size=buffer_size)

        value = 1

        bufferizer.flush(value=value)

        np_array = numpy.zeros(shape=bufferizer.shape,
                               dtype=dtype.numpy_dtype)+value
        diff = numpy.linalg.norm(bufferizer.darray.ndarray - np_array)

        assert bufferizer.element_shape == shape
        assert bufferizer.shape == (buffer_size,)+shape
        assert bufferizer.dtype == dtype
        assert diff < 1e-5

    @pytest.mark.parametrize("shape", [
                            (10, 10),
                            (10, 10, 3),
                            (3, 10, 10),
                            (10, 10, 2, 2)])
    @pytest.mark.parametrize("buffer_size", [
                            10,
                            3
                            ])
    def test_bufferizer_append_one_not_full(self, dtype, shape, buffer_size):

        bufferizer = dp.Bufferizer(shape=shape,
                                   dtype=dtype,
                                   buffer_size=buffer_size)

        randomarray = numpy.random.rand(*shape).astype(dtype.numpy_dtype)*10
        bufferizer.append_one(element=dp.darray(array=randomarray))

        diff = numpy.linalg.norm(bufferizer.darray.ndarray[0] - randomarray)

        assert not bufferizer.full
        assert bufferizer.element_shape == shape
        assert bufferizer.shape == (buffer_size,)+shape
        assert bufferizer.dtype == dtype
        assert diff < 1e-5

    @pytest.mark.parametrize("shape", [
                            (10, 10),
                            (10, 10, 3),
                            (3, 10, 10),
                            (10, 10, 2, 2)])
    @pytest.mark.parametrize("buffer_size", [
                            1
                            ])
    def test_bufferizer_append_one_full(self, dtype, shape, buffer_size):

        bufferizer = dp.Bufferizer(shape=shape,
                                   dtype=dtype,
                                   buffer_size=buffer_size)

        randomarray = numpy.random.rand(*shape).astype(dtype.numpy_dtype)*10
        bufferizer.append_one(element=dp.darray(array=randomarray))

        diff = numpy.linalg.norm(bufferizer.darray.ndarray[0] - randomarray)

        assert bufferizer.full
        assert bufferizer.element_shape == shape
        assert bufferizer.shape == (buffer_size,)+shape
        assert bufferizer.dtype == dtype
        assert diff < 1e-5

    @pytest.mark.parametrize("shape", [
                            (10, 10),
                            (10, 10, 3),
                            (3, 10, 10),
                            (10, 10, 2, 2)])
    @pytest.mark.parametrize("buffer_size", [
                            10,
                            5
                            ])
    @pytest.mark.parametrize("batch_size", [
                            4,
                            2
                            ])
    def test_bufferizer_append_multiple_not_full(self, dtype, shape, buffer_size, batch_size):

        bufferizer = dp.Bufferizer(shape=shape,
                                   dtype=dtype,
                                   buffer_size=buffer_size)

        r_shape = (batch_size,)+shape
        randomarray = numpy.random.rand(*r_shape).astype(dtype.numpy_dtype)*10
        bufferizer.append_multiple(element=dp.darray(array=randomarray))

        diff = numpy.linalg.norm(bufferizer.darray.ndarray[:batch_size] - randomarray)

        assert not bufferizer.full
        assert bufferizer.element_shape == shape
        assert bufferizer.shape == (buffer_size,)+shape
        assert bufferizer.dtype == dtype
        assert diff < 1e-5

    @pytest.mark.parametrize("shape", [
                            (10, 10),
                            (10, 10, 3),
                            (3, 10, 10),
                            (10, 10, 2, 2)])
    @pytest.mark.parametrize("buffer_size", [
                            10,
                            5
                            ])
    def test_bufferizer_append_multiple_full(self, dtype, shape, buffer_size):

        bufferizer = dp.Bufferizer(shape=shape,
                                   dtype=dtype,
                                   buffer_size=buffer_size)

        r_shape = (buffer_size,)+shape
        randomarray = numpy.random.rand(*r_shape).astype(dtype.numpy_dtype)*10
        bufferizer.append_multiple(element=dp.darray(array=randomarray))

        diff = numpy.linalg.norm(bufferizer.darray.ndarray - randomarray)

        assert bufferizer.full
        assert bufferizer.element_shape == shape
        assert bufferizer.shape == (buffer_size,)+shape
        assert bufferizer.dtype == dtype
        assert diff < 1e-5

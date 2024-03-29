import numpy
import pytest
import dolphin as dp


@pytest.mark.parametrize("shape", [(50, 50),
                                   (30, 5, 30),
                                   (4, 3, 2),
                                   (5, 4, 3, 2, 1),
                                   (1, 2, 3)])
@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_darray:
    """
    Set of tests for the darray class
    """

    def test_raw_creation(self, shape, dtype):
        """
        Test for raw creation of darray directly from
        shape and dtype
        """

        arr = dp.darray(shape=shape, dtype=dtype)
        assert arr.shape == shape
        assert arr.dtype == dtype

    def test_creation(self, shape, dtype):
        """
        Test for creation of darray from numpy array
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)
        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy)

        assert diff < 1e-5

    def test_zeros(self, shape, dtype):
        """
        Test for creation of darray from numpy array
        """
        dummy = numpy.zeros(shape, dtype=dtype.numpy_dtype)
        cuda_array = dp.zeros(shape, dtype)
        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy)

        assert diff < 1e-5

    def test_zeros_like(self, shape, dtype):
        """
        Test for creation of darray from numpy array
        """
        dummy = numpy.zeros(shape, dtype=dtype.numpy_dtype)
        cuda_array = dp.zeros_like(dummy)
        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy)

        assert diff < 1e-5

    def test_ones(self, shape, dtype):
        """
        Test for creation of darray from numpy array
        """
        dummy = numpy.ones(shape, dtype=dtype.numpy_dtype)
        cuda_array = dp.ones(shape, dtype)
        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy)

        assert diff < 1e-5

    def test_ones_like(self, shape, dtype):
        """
        Test for creation of darray from numpy array
        """
        dummy = numpy.ones(shape, dtype=dtype.numpy_dtype)
        cuda_array = dp.ones_like(dummy)
        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy)

        assert diff < 1e-5

    def test_empty(self, shape, dtype):
        """
        Test for creation of darray from numpy array
        """
        cuda_array = dp.empty(shape, dtype)

        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_empty_like(self, shape, dtype):
        """
        Test for creation of darray from numpy array
        """
        dummy = numpy.empty(shape, dtype=dtype.numpy_dtype)
        cuda_array = dp.empty_like(dummy)

        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_ndarray_copy(self, shape, dtype):
        """
        Test for copying ndarray from host to gpu
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(shape=shape, dtype=dtype)
        cuda_array.np = dummy

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy)

        assert diff < 1e-5

    def test_transpose(self, shape, dtype):
        """
        Test for transpose of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        perms = [i for i in range(len(shape))[::-1]]
        new_shape = tuple([shape[i] for i in perms])

        cuda_array_result = dp.zeros(shape=new_shape, dtype=dtype)

        cuda_array_result = dp.transpose(cuda_array, perms)
        dummy_result = numpy.transpose(dummy, perms)

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == new_shape
        assert cuda_array_result.dtype == dtype

    def test_flatten(self, shape, dtype):
        """
        Test for flatten of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        perms = [i for i in range(len(shape))[::-1]]
        new_shape = tuple([shape[i] for i in perms])

        cuda_array_result = dp.zeros(shape=new_shape, dtype=dtype)

        cuda_array_result = dp.transpose(cuda_array, perms).flatten()
        dummy_result = numpy.transpose(dummy, perms).flatten(order="C")

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == dummy_result.shape
        assert cuda_array_result.dtype == dtype

    def test_astype(self, shape, dtype):
        """
        Test for astype of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        for new_dtype in dp.dtype:

            cuda_array_result = dp.zeros(shape=shape, dtype=new_dtype)

            cuda_array.astype(new_dtype, cuda_array_result)
            dummy_result = dummy.astype(new_dtype.numpy_dtype)

            diff = numpy.linalg.norm(cuda_array_result.to_numpy()
                                     - dummy_result)

            assert diff < 1e-5, f"new_dtype: {new_dtype}, dtype: {dtype}"
            assert cuda_array_result.shape == shape
            assert cuda_array_result.dtype == new_dtype

    def test_astype_inplace(self, shape, dtype):
        """
        Test for astype of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        for new_dtype in dp.dtype:

            res = cuda_array.astype(new_dtype)
            dummy_result = dummy.astype(new_dtype.numpy_dtype)

            diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

            assert diff < 1e-5, f"new_dtype: {new_dtype}, dtype: {dtype}"
            assert res.shape == shape
            assert res.dtype == new_dtype

    def test_copy(self, shape, dtype):
        """
        Test for copy of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        cuda_array_result = cuda_array.copy()
        dummy_result = dummy.copy()

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_abs(self, shape, dtype):
        """
        Test for abs of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        dp.absolute(cuda_array, cuda_array_result)

        dummy_result = numpy.abs(dummy)

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_abs_inplace(self, shape, dtype):
        """
        Test for abs of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        res = dp.absolute(cuda_array)

        dummy_result = numpy.abs(dummy)

        diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert res.shape == shape
        assert res.dtype == dtype

    def test_add(self, shape, dtype):
        """
        Test for add of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        dp.add(cuda_array, cuda_array2, cuda_array_result)

        dummy_result = dummy + dummy2

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_add_inplace(self, shape, dtype):
        """
        Test for add of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        res = dp.add(cuda_array, cuda_array2)

        dummy_result = dummy + dummy2

        diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert res.shape == shape
        assert res.dtype == dtype

    def test_add_scalar(self, shape, dtype):
        """
        Test for add of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        dp.add(cuda_array, scalar, cuda_array_result)

        dummy_result = dummy + scalar

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_add_scalar_inplace(self, shape, dtype):
        """
        Test for add of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        res = dp.add(cuda_array, scalar)

        dummy_result = dummy + scalar

        diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert res.shape == shape
        assert res.dtype == dtype

    def test_iadd(self, shape, dtype):
        """
        Test for iadd of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array += cuda_array2

        dummy_result = dummy + dummy2

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_iadd_scalar(self, shape, dtype):
        """
        Test for iadd of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        cuda_array += scalar

        dummy_result = dummy + scalar

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_reversed_scalar_add(self, shape, dtype):
        """
        Test for add of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = 15

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array_result = scalar + cuda_array

        dummy_result = scalar + dummy

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_sub(self, shape, dtype):
        """
        Test for sub of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array.sub(cuda_array2, cuda_array_result)

        dummy_result = dummy - dummy2

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_sub_inplace(self, shape, dtype):
        """
        Test for sub of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        res = cuda_array - cuda_array2

        dummy_result = dummy - dummy2

        diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert res.shape == shape
        assert res.dtype == dtype

    def test_sub_scalar(self, shape, dtype):
        """
        Test for sub of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array.sub(scalar, cuda_array_result)

        dummy_result = dummy - scalar

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_sub_scalar_inplace(self, shape, dtype):
        """
        Test for sub of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        res = cuda_array - scalar

        dummy_result = dummy - scalar

        diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert res.shape == shape
        assert res.dtype == dtype

    def test_isub(self, shape, dtype):
        """
        Test for isub of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array -= cuda_array2

        dummy_result = dummy - dummy2

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_isub_scalar(self, shape, dtype):
        """
        Test for isub of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        cuda_array -= scalar

        dummy_result = dummy - scalar

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_rsub_array(self, shape, dtype):
        """
        Test for rsub of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array2.sub(cuda_array, cuda_array_result)

        dummy_result = dummy2 - dummy

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_rsub_scalar(self, shape, dtype):
        """
        Test for rsub of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = 15

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array_result = scalar - cuda_array

        dummy_result = scalar - dummy

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_mul(self, shape, dtype):
        """
        Test for mul of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array.multiply(cuda_array2, cuda_array_result)

        dummy_result = dummy * dummy2

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_mul_inplace(self, shape, dtype):
        """
        Test for mul of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        res = cuda_array * cuda_array2

        dummy_result = dummy * dummy2

        diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert res.shape == shape
        assert res.dtype == dtype

    def test_mul_scalar(self, shape, dtype):
        """
        Test for mul of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array.multiply(scalar, cuda_array_result)

        dummy_result = dummy * scalar

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_mul_scalar_inplace(self, shape, dtype):
        """
        Test for mul of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        res = cuda_array * scalar

        dummy_result = dummy * scalar

        diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert res.shape == shape
        assert res.dtype == dtype

    def test_imul(self, shape, dtype):
        """
        Test for imul of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array *= cuda_array2

        dummy_result = dummy * dummy2

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_imul_scalar(self, shape, dtype):
        """
        Test for imul of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)

        cuda_array *= scalar

        dummy_result = dummy * scalar

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_rmul_array(self, shape, dtype):
        """
        Test for rmul of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array2.multiply(cuda_array, cuda_array_result)

        dummy_result = dummy2 * dummy

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_rmul_scalar(self, shape, dtype):
        """
        Test for rmul of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = 3

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array_result = scalar * cuda_array

        dummy_result = scalar * dummy

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_div(self, shape, dtype):
        """
        Test for div of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array.divide(cuda_array2, cuda_array_result)

        dummy_result = dummy / dummy2

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_div_inplace(self, shape, dtype):
        """
        Test for div of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array2 = dp.darray(array=dummy2)

        res = cuda_array / cuda_array2

        dummy_result = dummy / dummy2

        diff = numpy.linalg.norm(res.to_numpy() - dummy_result)

        assert diff < 1e-5
        assert res.shape == shape
        assert res.dtype == dtype

    def test_div_scalar(self, shape, dtype):
        """
        Test for div of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)+1

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array.divide(scalar, cuda_array_result)

        dummy_result = dummy / scalar

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-4
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_div_scalar_inplace(self, shape, dtype):
        """
        Test for div of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)+1

        res = cuda_array / scalar

        dummy_result = dummy / scalar

        diff = numpy.linalg.norm(res.to_numpy() -
                                 dummy_result.astype(dtype.numpy_dtype))

        assert diff < 1e-4
        assert res.shape == shape
        assert res.dtype == dtype

    def test_idiv(self, shape, dtype):
        """
        Test for idiv of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array /= cuda_array2

        dummy_result = dummy / dummy2

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy_result)

        assert diff < 1e-4
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_idiv_scalar(self, shape, dtype):
        """
        Test for idiv of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array = dp.darray(array=dummy)

        scalar = dtype.numpy_dtype(numpy.random.rand()*10)+1

        cuda_array /= scalar

        dummy_result = dummy / scalar

        diff = numpy.linalg.norm(cuda_array.to_numpy() -
                                 dummy_result.astype(dtype.numpy_dtype))

        assert diff < 1e-4
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype

    def test_rdiv_array(self, shape, dtype):
        """
        Test for rdiv of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array = dp.darray(array=dummy)

        dummy2 = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array2 = dp.darray(array=dummy2)

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array2.divide(cuda_array, cuda_array_result)

        dummy_result = dummy2 / dummy

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-4
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    def test_rdiv_scalar(self, shape, dtype):
        """
        Test for rdiv of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)+1
        cuda_array = dp.darray(array=dummy)

        scalar = 2

        cuda_array_result = dp.zeros(shape=shape, dtype=dtype)
        cuda_array_result = scalar / cuda_array

        dummy_result = scalar / dummy

        diff = numpy.linalg.norm(cuda_array_result.to_numpy() - dummy_result)

        assert diff < 1e-4
        assert cuda_array_result.shape == shape
        assert cuda_array_result.dtype == dtype

    @pytest.mark.parametrize("value", [1, 10, 64])
    def test_darray_fill(self, shape, dtype, value):
        """
        Test for fill of darray
        """
        cuda_array = dp.darray(shape=shape, dtype=dtype)
        cuda_array.fill(value)

        dummy = numpy.zeros(shape, dtype=dtype.numpy_dtype)
        dummy.fill(value)

        diff = numpy.linalg.norm(cuda_array.to_numpy() - dummy)

        assert diff < 1e-4
        assert cuda_array.shape == shape
        assert cuda_array.dtype == dtype


@pytest.mark.parametrize("shape", [(50, 50),
                                   (200, 200, 200),
                                   (4, 3, 2),
                                   (5, 4, 3, 2, 1),
                                   (1, 2, 3)])
@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_darray_getitem:

    def darray_test_getitem_creation(self, shape, dtype):
        """
        Test for getitem of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index]
        dummy_result = dummy[index]

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_result.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_result)

    def darray_test_getitem_copy(self, shape, dtype):
        """
        Test for getitem of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index].copy()
        dummy_result = dummy[index]

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_result.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_result)

    def darray_test_getitem_fill(self, shape, dtype):
        """
        Test for getitem of darray
        """
        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index].fill(10)
        dummy_res = dummy[index]
        dummy_res.fill(10)

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_multiply_scalar(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index] * 10
        dummy_res = dummy[index] * 10

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_multiply_scalar_2(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index].multiply(10)
        dummy_res = dummy[index] * 10

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_multiply_darray(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape)*10
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])
        new_shape = tuple([max(s//2, 1) for s in shape])

        dummy_2 = numpy.random.rand(*new_shape).astype(dtype.numpy_dtype)
        cuda_array_2 = dp.darray(array=dummy_2)

        cuda_array_result = cuda_array[index].multiply(cuda_array_2)
        dummy_res = dummy[index] * dummy_2

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_div_scalar(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape) * 10
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index] / 3
        dummy_res = dummy[index] / 3
        dummy_res = dummy_res.astype(dtype.numpy_dtype)

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_div_scalar_2(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape) * 10
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index].divide(3)
        dummy_res = dummy[index] / 3
        dummy_res = dummy_res.astype(dtype.numpy_dtype)

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_div_darray(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape)*10
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])
        new_shape = tuple([max(s//2, 1) for s in shape])

        dummy_2 = numpy.random.rand(*new_shape)*3 + 1
        dummy_2 = dummy_2.astype(dtype.numpy_dtype)
        cuda_array_2 = dp.darray(array=dummy_2)

        cuda_array_result = cuda_array[index].divide(cuda_array_2)
        dummy_res = dummy[index] / dummy_2
        dummy_res = dummy_res.astype(dtype.numpy_dtype)

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_rdiv_scalar(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape) * 10 + 1
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = 3 / cuda_array[index]
        dummy_res = 3 / dummy[index]
        dummy_res = dummy_res.astype(dtype.numpy_dtype)

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_rdiv_scalar_2(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape) * 10 + 1
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index].reversed_divide(3)
        dummy_res = 3 / dummy[index]
        dummy_res = dummy_res.astype(dtype.numpy_dtype)

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_rdiv_darray(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape)*10+1
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])
        new_shape = tuple([max(s//2, 1) for s in shape])

        dummy_2 = numpy.random.rand(*new_shape)*3 + 1
        dummy_2 = dummy_2.astype(dtype.numpy_dtype)
        cuda_array_2 = dp.darray(array=dummy_2)

        cuda_array_result = cuda_array[index].reversed_divide(cuda_array_2)
        dummy_res = dummy_2 / dummy[index]
        dummy_res = dummy_res.astype(dtype.numpy_dtype)

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_cast(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape)*10
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        for dt in dp.dtype:
            dp_res = cuda_array[index].astype(dt)
            np_res = dummy[index].astype(dt.numpy_dtype)

            assert dp_res.shape == np_res.shape
            assert dp_res.dtype == dt
            assert numpy.allclose(dp_res.to_numpy(), np_res), f"{dtype} to {dt}"

    def darray_test_getitem_add_scalar(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index] + 10
        dummy_res = dummy[index] + 10

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_add_scalar_2(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index].add(10)
        dummy_res = dummy[index] + 10

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_add_darray(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape)*10
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])
        new_shape = tuple([max(s//2, 1) for s in shape])

        dummy_2 = numpy.random.rand(*new_shape).astype(dtype.numpy_dtype)
        cuda_array_2 = dp.darray(array=dummy_2)

        cuda_array_result = cuda_array[index].add(cuda_array_2)
        dummy_res = dummy[index] + dummy_2

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_sub_scalar(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index] - 10
        dummy_res = dummy[index] - 10

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_sub_scalar_2(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = cuda_array[index].substract(10)
        dummy_res = dummy[index] - 10

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_sub_darray(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape)*10
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])
        new_shape = tuple([max(s//2, 1) for s in shape])

        dummy_2 = numpy.random.rand(*new_shape).astype(dtype.numpy_dtype)
        cuda_array_2 = dp.darray(array=dummy_2)

        cuda_array_result = cuda_array[index].substract(cuda_array_2)
        dummy_res = dummy[index] - dummy_2

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def darray_test_getitem_abs(self, shape, dtype):
        """
        Test for getitem of darray
        """

        dummy = numpy.random.rand(*shape)*-1
        dummy = dummy.astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array_result = abs(cuda_array[index])
        dummy_res = abs(dummy[index])

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert cuda_array_result.shape == dummy_res.shape
        assert cuda_array_result.dtype == dtype
        assert numpy.allclose(cuda_array_result.to_numpy(), dummy_res)
        assert numpy.allclose(cuda_array.to_numpy(), dummy)


@pytest.mark.parametrize("shape", [(50, 50),
                                   (200, 200, 200),
                                   (4, 3, 2),
                                   (5, 4, 3, 2, 1),
                                   (1, 2, 3)])
@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_darray_setitem:

    def test_darray_setitem_scalar(self, shape, dtype):
        """
        """

        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])

        cuda_array[index] = 10
        dummy[index] = 10

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

    def test_darray_setitem_darray(self, shape, dtype):
        """
        """

        dummy = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        cuda_array = dp.darray(array=dummy)

        index = tuple([slice(0, max(s//2, 1), 1) for s in shape])
        new_shape = tuple([max(s//2, 1) for s in shape])
        new_array = numpy.random.rand(*new_shape).astype(dtype.numpy_dtype)

        cuda_array[index] = 10
        dummy[index] = 10

        for i in range(len(shape)):
            assert cuda_array.shape[i] == shape[i]

        assert cuda_array.dtype == dtype
        assert numpy.allclose(cuda_array.to_numpy(), dummy)

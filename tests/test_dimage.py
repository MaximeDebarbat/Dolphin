
import pytest
import numpy
import cv2
import dolphin as dp

@pytest.mark.parametrize("shape_format", [((50, 50), dp.dimage_dim_format.DOLPHIN_HW),
                                          ((50, 50, 1), dp.dimage_dim_format.DOLPHIN_HWC),
                                          ((1, 50, 50), dp.dimage_dim_format.DOLPHIN_CHW),
                                          ((50, 50, 3), dp.dimage_dim_format.DOLPHIN_HWC),
                                          ((3, 50, 50), dp.dimage_dim_format.DOLPHIN_CHW)])
@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_dimage:
    """
    Set of tests for the darray class
    """

    def test_dimage_creation(self, shape_format: tuple, dtype: dp.dtype):
        """
        Test the creation of a dimage
        """
        shape, format = shape_format
        dimage = dp.dimage(shape=shape, dtype=dtype)

        assert isinstance(dimage, dp.dimage)
        assert dimage.image_dim_format == format
        assert dimage.dtype == dtype

    def test_dimage_creation_from_numpy(self, shape_format: tuple, dtype: dp.dtype):
        """
        Test the creation of a dimage from a numpy array
        """
        shape, format = shape_format
        numpy_array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=numpy_array)

        assert isinstance(dimage, dp.dimage)
        assert dimage.image_dim_format == format
        assert dimage.dtype == dtype

    def test_dimage_creation_channel_format(self, shape_format: tuple, dtype: dp.dtype):
        """
        Test the creation of a dimage from a numpy array
        """
        shape, format = shape_format
        numpy_array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=numpy_array)

        assert isinstance(dimage, dp.dimage)
        assert dimage.image_dim_format == format
        assert dimage.dtype == dtype
        if format == dp.dimage_dim_format.DOLPHIN_HW:
            assert dimage.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        elif format == dp.dimage_dim_format.DOLPHIN_HWC and shape[2] == 1:
            assert dimage.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        elif format == dp.dimage_dim_format.DOLPHIN_CHW and shape[0] == 1:
            assert dimage.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        else:
            assert dimage.image_channel_format == dp.DOLPHIN_RGB

    def test_dimage_copy(self, shape_format: tuple, dtype: dp.dtype):
        """
        Test the creation of a dimage from a numpy array
        """
        shape, format = shape_format
        numpy_array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=numpy_array)
        dimage_copy = dimage.copy()

        diff = numpy.linalg.norm(dimage.to_numpy() - dimage_copy.to_numpy())

        assert isinstance(dimage_copy, dp.dimage)
        assert dimage_copy.image_dim_format == format
        assert dimage_copy.dtype == dtype
        assert diff < 1e-5

    def test_dimage_copy_channel_format(self, shape_format: tuple, dtype: dp.dtype):
        """
        Test the creation of a dimage from a numpy array
        """
        shape, format = shape_format
        numpy_array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=numpy_array)
        dimage_copy = dimage.copy()

        diff = numpy.linalg.norm(dimage.to_numpy() - dimage_copy.to_numpy())

        assert isinstance(dimage_copy, dp.dimage)
        assert dimage_copy.image_dim_format == format
        assert dimage_copy.dtype == dtype
        assert diff < 1e-5
        if format == dp.dimage_dim_format.DOLPHIN_HW:
            assert dimage_copy.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        elif format == dp.dimage_dim_format.DOLPHIN_HWC and shape[2] == 1:
            assert dimage_copy.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        elif format == dp.dimage_dim_format.DOLPHIN_CHW and shape[0] == 1:
            assert dimage_copy.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        else:
            assert dimage_copy.image_channel_format == dp.DOLPHIN_RGB

    def test_dimage_transpose(self, shape_format: tuple, dtype: dp.dtype):
        """
        Test the transpose of a dimage
        """
        shape, format = shape_format
        numpy_array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=numpy_array)

        perms_dp = tuple([i for i in range(len(dimage.shape))[::-1]])

        dimage_transpose = dimage.transpose(*perms_dp)

        diff = numpy.linalg.norm(dimage.to_numpy().transpose(*perms_dp) - dimage_transpose.to_numpy())

        assert isinstance(dimage_transpose, dp.dimage)
        assert dimage_transpose.dtype == dtype
        assert diff < 1e-5


@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_dimage_transpose:

    @pytest.mark.parametrize("shape", [(40, 50),
                                       (40, 500)])
    def test_dimage_HW(self, dtype, shape):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array)
        shape = dimage.shape

        perms = tuple([i for i in range(len(shape))[::-1]])
        new_shape = tuple([shape[i] for i in perms])
        dimage_transpose = dimage.transpose(*perms)

        assert dimage_transpose.shape == new_shape
        assert dimage_transpose.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_transpose.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        assert dimage_transpose.dtype == dtype


    @pytest.mark.parametrize("shape_format", [((40, 50, 3), dp.DOLPHIN_RGB),
                                       ((10, 10, 3), dp.DOLPHIN_RGB),
                                       ((500, 500, 1), dp.DOLPHIN_GRAY_SCALE)])
    def test_dimage_HWC(self, dtype, shape_format):
        """
        Test the transpose of a dimage
        """
        shape, format = shape_format
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        perms = tuple([i for i in range(len(shape))[::-1]])
        new_shape = tuple([shape[i] for i in perms])
        dimage_transpose = dimage.transpose(*perms)

        assert dimage_transpose.shape == new_shape
        assert dimage_transpose.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_transpose.image_channel_format == format
        assert dimage_transpose.dtype == dtype


    @pytest.mark.parametrize("shape_format", [((3, 40, 50), dp.DOLPHIN_RGB),
                                              ((3, 10, 10), dp.DOLPHIN_RGB),
                                              ((1, 500, 500), dp.DOLPHIN_GRAY_SCALE)])
    def test_dimage_CHW(self, dtype, shape_format):
        """
        Test the transpose of a dimage
        """
        shape, format = shape_format
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        perms = tuple([i for i in range(len(shape))[::-1]])
        new_shape = tuple([shape[i] for i in perms])
        dimage_transpose = dimage.transpose(*perms)

        assert dimage_transpose.shape == new_shape
        assert dimage_transpose.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_transpose.image_channel_format == format
        assert dimage_transpose.dtype == dtype


def letterbox(im: numpy.ndarray,
              new_shape: tuple = (640, 640),
              padding_value: int = 127):

    import cv2

    new_shape = new_shape[::-1]
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_NEAREST)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(
        padding_value,) * len(im.shape))  # add border
    return im

@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_dimage_resize:

    @pytest.mark.parametrize("shape", [(40, 50),
                                       (400, 500)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    def test_dimage_HW(self, dtype, shape, new_shape):
        """
        Test resize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape) * 100
        array = array.astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array)
        shape = dimage.shape

        dimage_resize = dimage.resize(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = cv2.resize(dimage.np.astype(s_dtype), new_shape, interpolation=cv2.INTER_NEAREST)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize)

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_resize.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(40, 50),
                                       (400, 500)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    def test_dimage_HW_transpose(self, dtype, shape, new_shape):
        """
        Test resize of a HW (grayscale) dimage but transposed
        """
        array = numpy.random.rand(*shape) * 100
        array = array.astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array).transpose(1, 0)
        shape = dimage.shape

        dimage_resize = dimage.resize(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = cv2.resize(dimage.np.astype(s_dtype), new_shape, interpolation=cv2.INTER_NEAREST)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize)

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_resize.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(40, 50),
                                       (400, 500)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    def test_dimage_HW_resize_padding(self, dtype, shape, new_shape):
        """
        Test resize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array)
        shape = dimage.shape

        dimage_resize, _, _ = dimage.resize_padding(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = letterbox(dimage.to_numpy().astype(s_dtype), new_shape)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize)

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_resize.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(40, 50),
                                       (400, 500)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    def test_dimage_HW_resize_padding_transpose(self, dtype, shape, new_shape):
        """
        Test resize of a HW (grayscale) dimage transpose
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array).transpose(1, 0)
        shape = dimage.shape

        dimage_resize, _, _ = dimage.resize_padding(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = letterbox(dimage.to_numpy().astype(s_dtype), new_shape)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize)

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_resize.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        assert dimage_resize.dtype == dtype


    @pytest.mark.parametrize("shape", [(400, 500, 3),
                                       (40, 50, 3),
                                       (200, 200, 3)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_HWC(self, dtype, shape, new_shape, format):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        dimage_resize = dimage.resize(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = cv2.resize(dimage.to_numpy().astype(s_dtype), new_shape, interpolation=cv2.INTER_NEAREST)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize)

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_resize.image_channel_format == format
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(3, 400, 500),
                                       (3, 40, 50),
                                       (3, 200, 200)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_HWC_transpose(self, dtype, shape, new_shape, format):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format).transpose(2, 1, 0)
        shape = dimage.shape

        dimage_resize = dimage.resize(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = cv2.resize(dimage.to_numpy().astype(s_dtype), new_shape, interpolation=cv2.INTER_NEAREST)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize)

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_resize.image_channel_format == format
        assert dimage_resize.dtype == dtype


    @pytest.mark.parametrize("shape", [(400, 500, 3),
                                       (40, 50, 3),
                                       (200, 200, 3)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_HWC_resize_padding(self, dtype, shape, new_shape, format):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape
        dimage_resize, _, _ = dimage.resize_padding(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = letterbox(dimage.to_numpy().astype(s_dtype), new_shape)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize)

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_resize.image_channel_format == format
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(3, 400, 500),
                                       (3, 40, 50),
                                       (3, 200, 200)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_HWC_resize_padding_transpose(self, dtype, shape, new_shape, format):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format).transpose(2, 1, 0)
        shape = dimage.shape
        dimage_resize, _, _ = dimage.resize_padding(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = letterbox(dimage.to_numpy().astype(s_dtype), new_shape)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize)

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_resize.image_channel_format == format
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(3, 400, 500),
                                       (3, 40, 50),
                                       (3, 200, 200)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_CHW(self, dtype, shape, new_shape, format):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        dimage_resize = dimage.resize(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = cv2.resize(dimage.np.transpose(1, 2, 0).astype(s_dtype), new_shape, interpolation=cv2.INTER_NEAREST)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize.transpose(2, 0, 1))

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_resize.image_channel_format == format
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(400, 500, 3),
                                       (40, 50, 3),
                                       (200, 200, 3)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_CHW_transpose(self, dtype, shape, new_shape, format):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format).transpose(2,1,0)
        shape = dimage.shape

        dimage_resize = dimage.resize(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = cv2.resize(dimage.np.transpose(1, 2, 0).astype(s_dtype), new_shape, interpolation=cv2.INTER_NEAREST)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize.transpose(2, 0, 1))

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_resize.image_channel_format == format
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(3, 400, 500),
                                       (3, 40, 50),
                                       (3, 200, 200)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_CHW_resize_padding(self, dtype, shape, new_shape, format):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        dimage_resize, _, _ = dimage.resize_padding(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = letterbox(dimage.to_numpy().transpose(1, 2, 0).astype(s_dtype), new_shape)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize.transpose(2, 0, 1))

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_resize.image_channel_format == format
        assert dimage_resize.dtype == dtype

    @pytest.mark.parametrize("shape", [(400, 500, 3),
                                       (40, 50, 3),
                                       (200, 200, 3)])
    @pytest.mark.parametrize("new_shape", [(200, 200),
                                           (400, 100),
                                           (100, 400)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_CHW_resize_padding_transpose(self, dtype, shape, new_shape, format):
        """
        Test the transpose of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format).transpose(2, 1, 0)
        shape = dimage.shape

        dimage_resize, _, _ = dimage.resize_padding(new_shape)

        s_dtype = dtype.numpy_dtype
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            s_dtype = numpy.float32

        cv_resize = letterbox(dimage.to_numpy().transpose(1, 2, 0).astype(s_dtype), new_shape)

        diff = numpy.linalg.norm(dimage_resize.to_numpy() - cv_resize.transpose(2, 0, 1))

        assert isinstance(dimage_resize, dp.dimage)
        assert dimage_resize.width == new_shape[0]
        assert dimage_resize.height == new_shape[1]
        assert diff < 1e-5
        assert dimage_resize.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_resize.image_channel_format == format
        assert dimage_resize.dtype == dtype


@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_dimage_normalization:

    @pytest.mark.parametrize("shape", [(40, 50),
                                       (400, 500)])
    def test_dimage_HW_255(self, dtype, shape):
        """
        Test resize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_255, dtype=dtype)
        result = (dimage.to_numpy()/255).astype(dimage_normalized.dtype.numpy_dtype)

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result)

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-5
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_normalized.image_channel_format == dp.DOLPHIN_GRAY_SCALE
        assert dimage_normalized.dtype == dtype

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 400, 500),
                                       (3, 200, 200)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_CHW_255(self, dtype, shape, format):
        """
        Test resize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_255, dtype=dtype)
        result = (dimage.to_numpy()/255).astype(dimage_normalized.dtype.numpy_dtype)

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result)

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-5
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_normalized.image_channel_format == format
        assert dimage_normalized.dtype == dtype

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (400, 500, 3),
                                       (200, 200, 3)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_HWC_255(self, dtype, shape, format):
        """
        Test resize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_255, dtype=dtype)
        result = (dimage.to_numpy()/255).astype(dimage_normalized.dtype.numpy_dtype)

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result)

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-5
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_normalized.image_channel_format == format
        assert dimage_normalized.dtype == dtype


    @pytest.mark.parametrize("shape", [(40, 50),
                                       (400, 500)])
    def test_dimage_HW_TF(self, dtype, shape):
        """
        Test resize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_TF)
        result = (dimage.to_numpy() / 127.5 - 1.0)

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-5, f"diff: {diff} | {dimage_normalized.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_normalized.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 400, 500),
                                       (3, 200, 200)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_CHW_TF(self, dtype, shape, format):
        """
        Test resize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_TF)
        result = (dimage.to_numpy() / 127.5 - 1.0)

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-5
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_normalized.image_channel_format == format

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (400, 500, 3),
                                       (200, 200, 3)])
    @pytest.mark.parametrize("format", [dp.DOLPHIN_RGB,
                                        dp.DOLPHIN_BGR])
    def test_dimage_HWC_TF(self, dtype, shape, format):
        """
        Test resize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=format)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_TF)
        result = (dimage.to_numpy() / 127.5 - 1.0)

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-5
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_normalized.image_channel_format == format

    @pytest.mark.parametrize("shape", [(40, 50),
                                       (400, 500)])
    def test_dimage_HW_mean_std(self, dtype, shape):
        """
        Test normalize of a HW (grayscale) dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        mean = numpy.random.rand(1).astype(dtype.numpy_dtype)+1
        std = numpy.random.rand(1).astype(dtype.numpy_dtype)+1
        dimage = dp.dimage(array=array)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_MEAN_STD, mean=mean, std=std)
        result = ((dimage.to_numpy()/255) - mean) / std

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_normalized.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_normalized.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_CHW_mean_std(self, dtype, shape):
        """
        Test normalize of a CHW dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        mean = numpy.random.rand(3).astype(dtype.numpy_dtype)+1
        std = numpy.random.rand(3).astype(dtype.numpy_dtype)+1
        dimage = dp.dimage(array=array)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_MEAN_STD, mean=mean, std=std, dtype=dp.dtype.float64)
        result = dimage.to_numpy()/255
        result[0, :, :] = (result[0, :, :] - mean[0]) / std[0]
        result[1, :, :] = (result[1, :, :] - mean[1]) / std[1]
        result[2, :, :] = (result[2, :, :] - mean[2]) / std[2]

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result.astype(dtype=numpy.double))

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-3, f"diff: {diff} | {dimage_normalized.to_numpy()[0][0][0]} | {result[0][0][0]}"
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_normalized.image_channel_format == dp.DOLPHIN_RGB

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    def test_dimage_HWC_mean_std(self, dtype, shape):
        """
        Test normalize of a HWC dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        mean = numpy.random.rand(3).astype(dtype.numpy_dtype)+1
        std = numpy.random.rand(3).astype(dtype.numpy_dtype)+1
        dimage = dp.dimage(array=array)
        shape = dimage.shape

        dimage_normalized = dimage.normalize(normalize_type=dp.dimage_normalize_type.DOLPHIN_MEAN_STD, mean=mean, std=std, dtype=dp.dtype.float64)
        result = dimage.to_numpy()/255
        result[:, :, 0] = (result[:, :, 0] - mean[0]) / std[0]
        result[:, :, 1] = (result[:, :, 1] - mean[1]) / std[1]
        result[:, :, 2] = (result[:, :, 2] - mean[2]) / std[2]

        diff = numpy.linalg.norm(dimage_normalized.to_numpy() - result.astype(dtype=numpy.double))

        assert isinstance(dimage_normalized, dp.dimage)
        assert diff < 1e-3, f"diff: {diff} | {dimage_normalized.to_numpy()[0][0][0]} | {result[0][0][0]}"
        assert dimage_normalized.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_normalized.image_channel_format == dp.DOLPHIN_RGB

@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_dimage_cvtcolor:

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    def test_dimage_cvtcolor_hwc_bgr2gray(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_GRAY_SCALE)

        result = dimage.to_numpy()[:, :, 0]*0.114 + dimage.to_numpy()[:, :, 1]*0.587 + dimage.to_numpy()[:, :, 2]*0.299

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_cvtcolor_hwc_bgr2gray_transpose(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR).transpose(2, 1, 0)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_GRAY_SCALE)

        result = dimage.to_numpy()[:, :, 0]*0.114 + dimage.to_numpy()[:, :, 1]*0.587 + dimage.to_numpy()[:, :, 2]*0.299

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_cvtcolor_chw_bgr2gray(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR)
        result = numpy.zeros((dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_GRAY_SCALE)

        result = dimage.to_numpy()[0, :, :]*0.114 + dimage.to_numpy()[1, :, :]*0.587 + dimage.to_numpy()[2, :, :]*0.299

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    def test_dimage_cvtcolor_chw_bgr2gray_transpose(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR).transpose(2, 1, 0)
        result = numpy.zeros((dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_GRAY_SCALE)

        result = dimage.to_numpy()[0, :, :]*0.114 + dimage.to_numpy()[1, :, :]*0.587 + dimage.to_numpy()[2, :, :]*0.299

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    def test_dimage_cvtcolor_hwc_rgb2gray(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_GRAY_SCALE)

        result = dimage.to_numpy()[:, :, 0]*0.299 + dimage.to_numpy()[:, :, 1]*0.587 + dimage.to_numpy()[:, :, 2]*0.114

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_cvtcolor_hwc_rgb2gray(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB).transpose(2, 1, 0)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_GRAY_SCALE)

        result = dimage.to_numpy()[:, :, 0]*0.299 + dimage.to_numpy()[:, :, 1]*0.587 + dimage.to_numpy()[:, :, 2]*0.114

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_cvtcolor_chw_rgb2gray(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB)
        result = numpy.zeros((dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_GRAY_SCALE)

        result = dimage.to_numpy()[0, :, :]*0.299 + dimage.to_numpy()[1, :, :]*0.587 + dimage.to_numpy()[2, :, :]*0.114

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_GRAY_SCALE

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    def test_dimage_cvtcolor_hwc_bgr2rgb(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_RGB)

        result[:, :, 0] = dimage.to_numpy()[:, :, 2]
        result[:, :, 1] = dimage.to_numpy()[:, :, 1]
        result[:, :, 2] = dimage.to_numpy()[:, :, 0]

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_RGB

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_cvtcolor_hwc_bgr2rgb_transpose(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR).transpose(2, 1, 0)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_RGB)

        result[:, :, 0] = dimage.to_numpy()[:, :, 2]
        result[:, :, 1] = dimage.to_numpy()[:, :, 1]
        result[:, :, 2] = dimage.to_numpy()[:, :, 0]

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_RGB

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_cvtcolor_chw_bgr2rgb(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_RGB)

        result[0, :, :] = dimage.to_numpy()[2, :, :]
        result[1, :, :] = dimage.to_numpy()[1, :, :]
        result[2, :, :] = dimage.to_numpy()[0, :, :]

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_RGB

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    def test_dimage_cvtcolor_chw_bgr2rgb_transpose(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR).transpose(2, 1, 0)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_RGB)

        result[0, :, :] = dimage.to_numpy()[2, :, :]
        result[1, :, :] = dimage.to_numpy()[1, :, :]
        result[2, :, :] = dimage.to_numpy()[0, :, :]

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_RGB

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    def test_dimage_cvtcolor_hwc_rgb2bgr(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_BGR)

        result[:, :, 0] = dimage.to_numpy()[:, :, 2]
        result[:, :, 1] = dimage.to_numpy()[:, :, 1]
        result[:, :, 2] = dimage.to_numpy()[:, :, 0]

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_BGR

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_cvtcolor_hwc_rgb2bgr(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB).transpose(2, 1, 0)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_BGR)

        result[:, :, 0] = dimage.to_numpy()[:, :, 2]
        result[:, :, 1] = dimage.to_numpy()[:, :, 1]
        result[:, :, 2] = dimage.to_numpy()[:, :, 0]

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_HWC
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_BGR

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    def test_dimage_cvtcolor_chw_rgb2bgr(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape).astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        dimage_cvt = dimage.cvtColor(dp.DOLPHIN_BGR)

        result[0, :, :] = dimage.to_numpy()[2, :, :]
        result[1, :, :] = dimage.to_numpy()[1, :, :]
        result[2, :, :] = dimage.to_numpy()[0, :, :]

        diff = numpy.linalg.norm(dimage_cvt.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage_cvt, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | {dimage_cvt.to_numpy()[0][0]} | {result[0][0]}"
        assert dimage_cvt.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage_cvt.image_channel_format == dp.DOLPHIN_BGR

    @pytest.mark.parametrize("shape", [(3, 2, 2),
                                       ])
    def test_dimage_cvtcolor_srcdst_bgrrgb(self, dtype, shape):
        """
        Test conversion of a dimage
        """
        array = numpy.random.rand(*shape)*10
        array = array.astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_BGR)
        result = numpy.zeros((dimage.shape[0], dimage.shape[1], dimage.shape[2]), dtype=dtype.numpy_dtype)

        result[0, :, :] = array[2, :, :]
        result[1, :, :] = array[1, :, :]
        result[2, :, :] = array[0, :, :]

        dp.cvtColor(src=dimage, color_format=dp.DOLPHIN_RGB, dst=dimage)

        diff = numpy.linalg.norm(dimage.to_numpy() - result.astype(dtype=numpy.float32))

        assert isinstance(dimage, dp.dimage)
        assert diff < 1e-4, f"diff: {diff} | \n {array} \n {dimage.to_numpy()} \n {result}"
        assert dimage.image_dim_format == dp.dimage_dim_format.DOLPHIN_CHW
        assert dimage.image_channel_format == dp.DOLPHIN_RGB


@pytest.mark.parametrize("dtype", [dp.dtype.float32,
                                   dp.dtype.float64,
                                   dp.dtype.int32,
                                   dp.dtype.int16,
                                   dp.dtype.int8,
                                   dp.dtype.uint32,
                                   dp.dtype.uint16,
                                   dp.dtype.uint8])
class test_dimage_crop_n_resize:

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    @pytest.mark.parametrize("size", [(20, 20),
                                      (100, 100),
                                      (40, 60),
                                      (222, 333)])
    @pytest.mark.parametrize("crop", [[[0, 0, 20, 20]],
                                      [[10, 10, 12, 15],
                                      [5, 5, 30, 25]],
                                      [[20, 20, 30, 40]]])
    def test_dimage_hwc_crop_n_resize(self, dtype, shape, size, crop):
        """
        Test conversion of a dimage
        """
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            dtype = dp.float32

        array = numpy.random.rand(*shape)*100
        array = array.astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB)
        coords = dp.darray(array=numpy.array(crop).astype(numpy.uint32))

        dimage_crop = dimage.crop_and_resize(coordinates=coords, size=size)

        image_crop = []
        for coords in crop:
            image_crop.append(
                cv2.resize(array[coords[1]:coords[3], coords[0]:coords[2]],
                           size,
                           interpolation=cv2.INTER_NEAREST)
            )

        diff = numpy.linalg.norm(dimage_crop.np - numpy.array(image_crop).astype(dtype=dtype.numpy_dtype))

        assert diff < 1e-5
        assert dimage_crop.shape == (len(crop),)+size[::-1]+(3,)
        assert dimage_crop.size == numpy.array(image_crop, dtype=dtype.numpy_dtype).size
        assert dimage_crop.shape == numpy.array(image_crop, dtype=dtype.numpy_dtype).shape

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    @pytest.mark.parametrize("size", [(20, 20),
                                      (100, 100),
                                      (40, 60),
                                      (222, 333)])
    @pytest.mark.parametrize("crop", [[[0, 0, 20, 20]],
                                      [[10, 10, 12, 15],
                                      [5, 5, 30, 25]],
                                      [[20, 20, 30, 40]]])
    def test_dimage_chw_crop_n_resize(self, dtype, shape, size, crop):
        """
        Test conversion of a dimage
        """
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            dtype = dp.float32

        array = numpy.random.rand(*shape)*100
        array = array.astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB)
        coords = dp.darray(array=numpy.array(crop).astype(numpy.uint32))
        array = array.transpose(1, 2, 0)

        dimage_crop = dimage.crop_and_resize(coordinates=coords, size=size)

        image_crop = []
        for coords in crop:
            image_crop.append(
                cv2.resize(array[coords[1]:coords[3], coords[0]:coords[2]],
                           size,
                           interpolation=cv2.INTER_NEAREST)
            )

        diff = numpy.linalg.norm(dimage_crop.to_numpy() - numpy.array(image_crop).astype(dtype=dtype.numpy_dtype).transpose(0,3,1,2))

        assert diff < 1e-5
        assert dimage_crop.shape == (len(crop), 3)+size[::-1]
        assert dimage_crop.size == numpy.array(image_crop, dtype=dtype.numpy_dtype).size

    @pytest.mark.parametrize("shape", [(40, 50, 3),
                                       (240, 250, 3)])
    @pytest.mark.parametrize("size", [(20, 20),
                                      (100, 100),
                                      (40, 60),
                                      (200, 300)])
    @pytest.mark.parametrize("crop", [[[0, 0, 20, 20]],
                                      [[10, 10, 12, 15],
                                      [5, 5, 30, 25]],
                                      [[20, 20, 30, 40]]])
    def test_dimage_hwc_crop_n_resize_padding(self, dtype, shape, size, crop):
        """
        Test conversion of a dimage
        """
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            dtype = dp.float32

        array = numpy.random.rand(*shape)*100
        array = array.astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB)
        coords = dp.darray(array=numpy.array(crop).astype(numpy.uint32))

        dimage_crop = dimage.crop_and_resize_padding(coordinates=coords,
                                                     size=size)

        image_crop = []
        for coords in crop:
            image_crop.append(
                letterbox(array[coords[1]:coords[3], coords[0]:coords[2]],
                          size)
            )

        diff = numpy.linalg.norm(dimage_crop.to_numpy() - numpy.array(image_crop).astype(dtype=dtype.numpy_dtype))

        assert diff < 1e-5
        assert dimage_crop.shape == (len(crop),)+size[::-1]+(3,)
        assert dimage_crop.shape == numpy.array(image_crop, dtype=dtype.numpy_dtype).shape
        assert dimage_crop.size == numpy.array(image_crop, dtype=dtype.numpy_dtype).size

    @pytest.mark.parametrize("shape", [(3, 40, 50),
                                       (3, 240, 250)])
    @pytest.mark.parametrize("size", [(20, 20),
                                      (100, 100),
                                      (40, 60),
                                      (200, 300)])
    @pytest.mark.parametrize("crop", [[[0, 0, 20, 20]],
                                      [[10, 10, 12, 15],
                                      [5, 5, 30, 25]],
                                      [[20, 20, 30, 40]]])
    def test_dimage_chw_crop_n_resize_padding(self, dtype, shape, size, crop):
        """
        Test conversion of a dimage
        """
        if dtype.numpy_dtype not in [
                numpy.uint8,
                numpy.int8,
                numpy.uint16,
                numpy.int16,
                numpy.int32,
                numpy.float32,
                numpy.float64]:
            dtype = dp.float32

        array = numpy.random.rand(*shape)*100
        array = array.astype(dtype.numpy_dtype)
        dimage = dp.dimage(array=array, channel_format=dp.DOLPHIN_RGB)
        coords = dp.darray(array=numpy.array(crop).astype(numpy.uint32))
        array = array.transpose(1, 2, 0)

        dimage_crop = dimage.crop_and_resize_padding(coordinates=coords,
                                                     size=size)

        image_crop = []
        for coords in crop:
            image_crop.append(
                letterbox(array[coords[1]:coords[3], coords[0]:coords[2]],
                          size)
            )

        diff = numpy.linalg.norm(dimage_crop.to_numpy() - numpy.array(image_crop).astype(dtype=dtype.numpy_dtype).transpose(0,3,1,2))

        assert diff < 1e-5
        assert dimage_crop.shape == (len(crop), 3)+size[::-1]
        assert dimage_crop.size == numpy.array(image_crop, dtype=dtype.numpy_dtype).size
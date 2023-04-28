"""_summary_

:return: _description_
:rtype: _type_
"""

import os

import tensorrt as trt
import pycuda.driver as cuda

import numpy as np

from .image_batcher import ImageBatcher  # pylint: disable=import-error
from .logger import TrtLogger  # pylint: disable=import-error


class CalibratorBase:
    """_summary_

    :param cache_file: _description_
    :type cache_file: str
    :param logger: _description_
    :type logger: object
    :param quantile: _description_, defaults to 0.5
    :type quantile: float, optional
    :param regression_cutoff: _description_, defaults to 0.5
    :type regression_cutoff: float, optional
    """

    def __init__(
            self,
            cache_file: str,
            logger: object,
            quantile: float = 0.5,
            regression_cutoff: float = 0.5):
        """_summary_
        """

        self.quantile = quantile
        self.regression_cutoff = regression_cutoff

        self.trt_logger = logger
        self.cache_file = cache_file
        self.image_batcher = None
        self._batch_generator = None
        self._batch_allocation = None

    def get_regression_cutoff(self):
        """
        __summary__
        """
        return self.regression_cutoff

    def get_quantile(self):
        """
        __summary__
        """
        return self.quantile

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        __summary__
        """

        self.image_batcher = image_batcher
        size = int(
            np.dtype(
                self.image_batcher.dtype).itemsize *
            np.prod(
                self.image_batcher.shape))
        self._batch_allocation = cuda.mem_alloc(
            size)  # pylint: disable=no-member
        self._batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        __summary__
        """

        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names: object):
        # pylint: disable=unused-argument
        """
        __summary__
        """
        if not self.image_batcher:
            return None
        try:
            batch, _ = next(self._batch_generator)
            self.trt_logger.log(
                self.trt_logger.Severity.INFO,
                f"Calibrating image {self.image_batcher.image_index} \
/ {self.image_batcher.num_images}")
            cuda.memcpy_htod(  # pylint: disable=no-member
                self._batch_allocation,
                np.ascontiguousarray(batch))
            return [int(self._batch_allocation)]

        except StopIteration:
            self.trt_logger.log(
                self.trt_logger.Severity.INFO,
                "Finished calibration batches")
            return None

    def read_histogram_cache(self, *args):
        # pylint: disable=unused-argument
        """
        Overrides from trt.IInt8Calibrator.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as histogram_file:
                self.trt_logger.log(
                    self.trt_logger.Severity.INFO,
                    f"Using histogram calibration cache file: \
{self.cache_file}")
                return histogram_file.read()
        else:
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8Calibrator.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as read_file:
                self.trt_logger.log(
                    self.trt_logger.Severity.INFO,
                    f"Using calibration cache file: {self.cache_file}")
                return read_file.read()
        else:
            return None

    def write_calibration_cache(self, cache, **kwargs):
        # pylint: disable=unused-argument
        """
        Overrides from trt.IInt8Calibrator.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is not None:
            with open(self.cache_file, "wb") as calib_cache:
                self.trt_logger.log(
                    self.trt_logger.Severity.INFO,
                    f"Writing calibration cache data to: {self.cache_file}")
                calib_cache.write(cache)

    def write_histogram_cache(self, cache: object):
        """
        Overrides from trt.IInt8Calibrator.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is not None:
            with open(self.cache_file, "wb") as histogram_cache:
                self.trt_logger.log(
                    self.trt_logger.Severity.INFO,
                    f"Writing histogram calibration cache data to: \
{self.cache_file}")
                histogram_cache.write(cache)


class IInt8LegacyCalibrator(CalibratorBase, trt.IInt8LegacyCalibrator):
    # pylint: disable=no-member
    """_summary_

    :param cache_file: _description_
    :type cache_file: str
    :param logger: _description_
    :type logger: object
    :param quantile: _description_, defaults to 0.5
    :type quantile: float, optional
    :param regression_cutoff: _description_, defaults to 0.5
    :type regression_cutoff: float, optional
    """

    def __init__(self, cache_file: str,
                 logger: TrtLogger,
                 quantile: float = 0.5,
                 regression_cutoff: float = 0.5):
        """_summary_
        """

        trt.IInt8LegacyCalibrator.__init__(self)
        CalibratorBase.__init__(self, cache_file,
                                logger, quantile,
                                regression_cutoff)


class IInt8EntropyCalibrator(CalibratorBase, trt.IInt8EntropyCalibrator):
    # pylint: disable=no-member
    """_summary_

    :param CalibratorBase: _description_
    :type CalibratorBase: _type_
    :param trt: _description_
    :type trt: _type_
    """

    def __init__(self, cache_file: str,
                 logger: TrtLogger):
        trt.IInt8EntropyCalibrator.__init__(self)
        CalibratorBase.__init__(self, cache_file, logger)


class IInt8EntropyCalibrator2(CalibratorBase, trt.IInt8EntropyCalibrator2):
    # pylint: disable=no-member
    """_summary_

    :param CalibratorBase: _description_
    :type CalibratorBase: _type_
    :param trt: _description_
    :type trt: _type_
    """

    def __init__(self, cache_file: str, logger: TrtLogger):
        trt.IInt8EntropyCalibrator2.__init__(self)
        CalibratorBase.__init__(self, cache_file, logger)


class IInt8MinMaxCalibrator(CalibratorBase, trt.IInt8MinMaxCalibrator):
    # pylint: disable=no-member
    """_summary_

    :param CalibratorBase: _description_
    :type CalibratorBase: _type_
    :param trt: _description_
    :type trt: _type_
    """

    def __init__(self, cache_file: str, logger: TrtLogger):
        trt.IInt8MinMaxCalibrator.__init__(self)
        CalibratorBase.__init__(self, cache_file, logger)

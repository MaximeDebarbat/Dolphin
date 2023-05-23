
import os
import sys

import numpy as np
from PIL import Image  # pylint: disable=import-error


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self,
                 num_images: int,
                 preprocessor: str,
                 calib_images: str,
                 dtype: object,
                 shape: tuple,
                 logger: object):

        self.dtype = dtype
        self.shape = shape
        self.batch_size = self.shape[0]
        self.max_num_images = num_images
        self.preprocessor = preprocessor
        self.logger = logger

        if calib_images is None:
            self.init_calib_empty()
        else:
            self.init_calib_image_path(calib_images)

        self.batch_index = 0
        self.image_index = 0

    def init_calib_empty(self):
        """_summary_
        """

        self.num_batches = 1 + int((self.max_num_images - 1) / self.batch_size)
        self.num_images = self.max_num_images
        self.batches = []
        self.height = self.shape[2]
        self.width = self.shape[3]
        self.format = "NCHW"

        if self.shape[1] == 1:
            self.convert_mode = "L"
        else:
            self.convert_mode = "RGB"

        for _ in range(self.num_batches):
            self.batches.append([np.random.randint(low=0, high=255, size=(
                500, 500, 3)).astype(np.uint8)] * self.batch_size)

    def init_calib_image_path(self, path):
        """_summary_

        :param path: _description_
        :type path: _type_
        :return: _description_
        :rtype: _type_
        """

        inp = os.path.realpath(path)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[
                1].lower() in extensions

        if os.path.isdir(inp):
            self.images = [
                os.path.join(
                    inp,
                    f) for f in os.listdir(inp) if is_image(
                    os.path.join(
                        inp,
                        f))]
            self.images.sort()
        elif os.path.isfile(inp):
            if is_image(inp):
                self.images.append(inp)
        self.num_images = len(self.images)
        if self.max_num_images < 1:
            print(
                f"No valid {'/'.join(extensions)} images found in {inp}")
            sys.exit(1)

        assert len(self.shape) == 4
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3 or self.shape[1] == 1:
            if self.shape[1] == 3:
                self.convert_mode = "RGB"
            else:
                self.convert_mode = "L"

            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3 or self.shape[3] == 1:
            if self.shape[3] == 3:
                self.convert_mode = "RGB"
            else:
                self.convert_mode = "L"

            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if self.max_num_images and 0 < self.max_num_images < len(self.images):
            self.num_images = self.max_num_images
        if self.num_images < 1:
            self.logger.log(
                self.logger.Severity.ERROR,
                "Not enough images to create batches")
            sys.exit(1)

        self.images = self.images[0: self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

    def preprocess_image(self, inp):

        if isinstance(inp, str):
            image = Image.open(inp)
        else:
            image = Image.fromarray(inp)

        image = image.convert(mode=self.convert_mode)

        if self.preprocessor == "128":
            image = image.resize(
                (self.width, self.height), resample=Image.BILINEAR)
            image = np.asarray(image, dtype=self.dtype)
            image = (image - 128.0) / 128.0
        elif self.preprocessor == "255":
            image = image.resize(
                (self.width, self.height), resample=Image.BICUBIC)
            image = np.asarray(image, dtype=self.dtype)
            image = image / 255.0
        elif self.preprocessor == "imagenet":
            if self.convert_mode == "L":
                self.logger.log(
                    self.logger.Severity.ERROR,
                    "Can't use --preprocessor=imagenet if the model \
uses grayscale images")

            image = image.resize(
                (self.width, self.height), resample=Image.BICUBIC)
            image = np.asarray(image, dtype=self.dtype)
            image = image - np.asarray([123.68, 116.28, 103.53])
            image = image / np.asarray([58.395, 57.120, 57.375])
        else:
            print(
                f"Preprocessing method {self.preprocessor} not supported.")
            sys.exit(1)
        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image

    def get_batch(self):

        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            for i, image in enumerate(batch_images):
                batch_data[i] = self.preprocess_image(image)
                self.image_index += 1
            self.batch_index += 1
            yield batch_data, batch_images

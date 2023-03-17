"""
Dolphin setup file.
If you want to install the package from source, run:
  >>> pip install -e .
Otherwise, run:
  >>> pip install dolphin-python

"""

import io
import os

from setuptools import setup, find_packages

# Package meta-data.
NAME = 'dolphin'
DESCRIPTION = 'Cuda based library for fast and seamless deep learning inference.'
URL = 'https://github.com/MaximeDebarbat/Dolphin'
EMAIL = 'debarbat.maxime@gmail.com'
AUTHOR = 'Maxime'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = '0.0.1'

# What packages are required for this module to be executed?
REQUIRED = [
    "nvidia-pyindex",
    "appdirs==1.4.4",
    "Jinja2==3.1.2",
    "Mako==1.2.4",
    "MarkupSafe==2.1.2",
    "numpy==1.23.5",
    "onnx==1.13.1",
    "opencv-python==4.5.5.64",
    "Pillow==9.4.0",
    "pip-chill==1.0.1",
    "platformdirs==2.6.2",
    "polygraphy==0.45.3",
    "protobuf==3.20.3",
    "pycodestyle==2.10.0",
    "pycuda==2022.2.2",
    "pytools==2022.1.14",
    "tomli==2.0.1",
    "tqdm==4.65.0",
    "typing_extensions==4.4.0"
]

try:
    with io.open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                              'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license='Apache',
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)

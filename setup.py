"""
Dolphin setup file.
If you want to install the package from source, run:
  >>> pip install -e .
Otherwise, run:
  >>> pip install dolphin-python

"""

import io
import os
from glob import glob

from setuptools import setup, find_packages

NAME = 'dolphin-python'
DESCRIPTION = 'Cuda based library for fast \
and seamless deep learning inference.'
URL = 'https://github.com/MaximeDebarbat/Dolphin'
EMAIL = 'debarbat.maxime@gmail.com'
AUTHOR = 'Maxime'
REQUIRES_PYTHON = '>=3.5.0'

# What packages are required for this module to be executed?
REQUIRED = [
    "Jinja2==3.1.2",
    "numpy==1.23.5",
    "onnx",
    "opencv-python==4.5.5.64",
    "Pillow==9.4.0",
    "pycuda==2022.2.2",
    "tqdm==4.65.0",
]

here = os.path.abspath(os.path.dirname(__file__))
about = {}

try:
    with io.open(os.path.join(here, 'README.md'),
                 encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

with open(os.path.join(here, "dolphin", 'version.py'),
          encoding='utf-8') as f:
    exec(f.read(), about)

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests",
                                    "*.tests",
                                    "*.tests.*",
                                    "tests.*"]),
    data_files=[('dolphin', glob('dolphin/cutils/cuda/*.cu'))],
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
    options={'bdist_wheel': {
                             'universal': True
                            }
             }
)

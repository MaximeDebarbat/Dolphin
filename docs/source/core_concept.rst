Core Concepts
=============

Dolphin is a tool for CUDA-accelerated computing in a deep learning context. It
is designed to be used as a library. Its purpose is to gather the most common
optimisation techniques for deep learning inference and make them available
in a simple and easy to use interface.

Overview
--------

Dolphin provides a set of classes implementing CUDA-accelerated operations.
The base class :py:class:`dolphin.darray` is an object manipulating a CUDA
array. It provides a set of numpy like methods to perform operations on the
array.

The :py:class:`dolphin.dimage` part is the part of the library dedicated to
image processing. It provides a set of methods to manipulate images which are
the ones providing the most speed up compared to common CPU implementations.

Simplicity
----------

Dolphin is meant to provide a simple and easy to use interface for deep learning inference application,
centralizing the most common optimisation techniques in a single library in order to have an easy-to-use
optimized library.

Disclaimer
----------

This library is currently under development. The API might not be stable yet. Some features might be missing,
some might be broken, some might be optimized. You are vert welcome to contribute to this project.
Be kind, be constructive, be open.

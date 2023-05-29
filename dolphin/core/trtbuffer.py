"""
_summary_
"""

from typing import Any, Dict, List
import pycuda.driver as cuda  # pylint: disable=import-error

import dolphin


class CudaTrtBuffers:
    """
    To be used with the :func:`darray <dolphin.engine>` class.
    This class actually manages the :func:`darray <dolphin.Bufferizer>`
    used by the engine, both for inputs and outputs.

    To ease the use of the :func:`darray <dolphin.Bufferizer>`,
    this class can be understood as a dict in order to
    name inputs and outputs.

    Note that the names of inputs and outputs have to match the names
    of the inputs and outputs of the engine.

    The constructor of this class takes an optional `cuda.Stream`
    as an argument.
    """

    def __init__(self, stream: cuda.Stream = None):
        """
        Constructor of the class.
        """

        self._inputs: Dict[str, dolphin.Bufferizer] = {}
        self._outputs: Dict[str, dolphin.Bufferizer] = {}

        self._input_order = []
        self._output_order = []

        self._stream = stream

    def allocate_input(self, name: str,
                       shape: tuple,
                       buffer_size: int,
                       dtype: object,
                       buffer_full_hook: callable = None,
                       flush_hook: callable = None,
                       allocate_hook: callable = None,
                       append_one_hook: callable = None,
                       append_multiple_hook: callable = None) -> None:
        """
        Method to allocate an input buffer.
        This methods creates a `dolphin.Bufferizer` and adds it to the inputs
        dict with the given name.

        :param name: Name of the input.
        :type name: str
        :param shape: Shape of a single element in the buffer.
        :type shape: tuple
        :param buffer_size: Size of the buffer.
        :type buffer_size: int
        :param dtype: Dtype of the buffer.
        :type dtype: object
        :param buffer_full_hook: callable function called each time the
                                    buffer is full.
        :type buffer_full_hook: callable
        :param flush_hook: callable function called each time the buffer
                            is flushed.
        :type flush_hook: callable
        :param allocate_hook: callable function called each time the
                                `allocate` method is called.
        :type allocate_hook: callable
        :param append_one_hook: callable function called each time the
                                `append_one` method is called.
        :type append_one_hook: callable
        :param append_multiple_hook: callable function called each time the
                                        `append_multiple` method is called.
        :type append_multiple_hook: callable
        """

        self._inputs[name] = dolphin.Bufferizer(
            shape=shape,
            buffer_size=buffer_size,
            dtype=dtype,
            stream=self._stream,
            buffer_full_hook=buffer_full_hook,
            flush_hook=flush_hook,
            allocate_hook=allocate_hook,
            append_one_hook=append_one_hook,
            append_multiple_hook=append_multiple_hook)
        self._inputs[name].allocate()

        self._input_order.append(name)

    def allocate_output(self, name: str,
                        shape: tuple,
                        dtype: dolphin.dtype) -> None:
        """
        Method to allocate an output buffer.
        Oppositely to the inputs, the outputs are not `dolphin.Bufferizer`.
        They are `darray`.

        :param name: Name of the output.
        :type name: str
        :param shape: Shape of a single element in the buffer.
        :type shape: tuple
        :param dtype: Dtype of the buffer.
        :type dtype: dolphin.dtype
        """

        self._outputs[name] = dolphin.darray(shape=shape,
                                             dtype=dtype,
                                             stream=self._stream)

        self._output_order.append(name)

    def flush(self, value: Any = 0) -> None:
        """
        Method to flush all the input buffers.
        Note that this method will trigger the `flush_hook` of each
        `dolphin.Bufferizer`.

        :param value: value to initialize the inputs with, defaults to 0
        :type value: int, optional
        """

        for inp in self._inputs.values():
            inp.flush(value)

    def append_one_input(self, name: str,
                         data: dolphin.darray):
        """
        Method to append a single element to the input buffer.
        Note that this method will trigger the `append_one_hook` of the
        `dolphin.Bufferizer`. It will also trigger the `buffer_full_hook`
        if the buffer is full.

        :param name: Name of the input.
        :type name: str
        :param data: Data to append.
        :type data: dolphin.darray
        """

        self._inputs[name].append_one(data)

    def append_multiple_input(self, name: str,
                              data: dolphin.darray):
        """
        Method to append multiple elements to the input buffer.
        Note that this method will trigger the `append_multiple_hook` of the
        `dolphin.Bufferizer`. It will also trigger the `buffer_full_hook`
        if the buffer is full.

        :param name: Name of the input.
        :type name: str
        :param data: Data to append.
        :type data: dolphin.darray
        """

        self._inputs[name].append_multiple_input(data)

    @property
    def input_shape(self) -> Dict[str, tuple]:
        """
        Property to get the shape of the inputs.
        Returns a dict with the name of the input as key and the shape as
        value.

        :return: Shape of the inputs.
        :rtype: Dict[str, tuple]
        """
        return {key: inp.shape for key, inp in self._inputs.items()}

    @property
    def input_dtype(self) -> Dict[str, dolphin.dtype]:
        """
        Property to get the dtype of the inputs.
        Returns a dict with the name of the input as key and the dtype as
        value.

        :return: Dtype of the inputs.
        :rtype: Dict[str, dolphin.dtype]
        """
        return {key: inp.dtype for key, inp in self._inputs.items()}

    @property
    def output_shape(self) -> Dict[str, tuple]:
        """
        Property to get the shape of the outputs.
        Returns a dict with the name of the output as key and the shape as
        value.

        :return: Shape of the outputs.
        :rtype: Dict[str, tuple]
        """
        return {key: out.shape for key, out in self._outputs.items()}

    @property
    def output_dtype(self):
        """
        Property to get the dtype of the outputs.
        Returns a dict with the name of the output as key and the dtype as
        value.

        :return: Dtype of the outputs.
        :rtype: Dict[str, dolphin.dtype]
        """
        return {key: out.dtype for key, out in self._outputs.items()}

    @property
    def full(self) -> bool:
        """
        Property to check if the buffer is full.
        Returns True if at least one of the input buffer is full.

        :return: True if at least one of the input buffer is full.
        :rtype: bool
        """

        for key, buffer in self._inputs.items():
            if buffer.full:
                return True

        return False

    @property
    def output(self) -> Dict[str, dolphin.darray]:
        """
        Property to get the output of the buffer.
        Returns a dict with the name of the output as key and the output as
        value.

        :return: Output of the buffer.
        :rtype: Dict[str, dolphin.darray]
        """

        return self._outputs

    @property
    def input_bindings(self) -> List[dolphin.darray]:
        """
        Property to get the input bindings.
        'input bindings' refers here to the list
        of input allocations

        :return: Input bindings.
        :rtype: List[dolphin.darray]
        """

        return [self._inputs[name].allocation for name in self._input_order]

    @property
    def output_bindings(self) -> List[int]:
        """
        Property to get the output bindings.
        'output bindings' refers here to the list
        of output allocations

        :return: Output bindings.
        :rtype: List[int]
        """

        return [self._outputs[name].allocation for name in self._output_order]

    @property
    def bindings(self):
        """
        Property to get the bindings.

        :return: Bindings.
        :rtype: List[int]
        """
        return self.input_bindings + self.output_bindings

"""_summary_
"""

import sys
from datetime import datetime

import tensorrt as trt  # noqa

TRT_LOGGER = None

if int(trt.__version__[2]) == 0:
    TRT_LOGGER = trt.Logger  # noqa
else:
    TRT_LOGGER = trt.ILogger  # noqa


class TrtLogger(TRT_LOGGER):
    """
    TensorRT Logger, used to print TensorRT's internal log messages to stdout.

    It is recommended to use this logger to print TensorRT's internal log.
    """

    def __init__(self,
                 verbose_mode: bool = False,
                 stdout: object = sys.stdout,
                 stderr: object = sys.stderr):

        TRT_LOGGER.__init__(self)  # pylint: disable=non-parent-init-called
        if verbose_mode:
            self.min_severity = TRT_LOGGER.Severity.VERBOSE
        else:
            self.min_severity = TRT_LOGGER.Severity.INFO

        self.stdout = stdout
        self.stderr = stderr

    def now(self) -> str:
        """Helping function that returns the current date
        in format %d/%m/%Y-%H:%M:%S

        :return: Current time
        :rtype: str
        """
        return f'[{datetime.now().strftime("%d/%m/%Y-%H:%M:%S")}]'

    def log(self, severity: TRT_LOGGER.Severity,
            msg: str) -> None:
        """Main logging function.

        :param severity: Severity of the message.
        :type severity: trt.tensorrt.ILogger.Severity
        :param msg: Message to be printed.
        :type msg: str
        """

        if severity == TRT_LOGGER.Severity.ERROR:
            print(self.now() + "[E] " + msg, file=self.stderr)

        if severity == TRT_LOGGER.Severity.INTERNAL_ERROR:
            print(self.now() + "[INTERNAL ERROR] " + msg, file=self.stderr)

        if severity == TRT_LOGGER.Severity.WARNING:
            print(self.now() + "[W] " + msg, file=self.stdout)

        if (self.min_severity == TRT_LOGGER.Severity.VERBOSE
           and severity == TRT_LOGGER.Severity.VERBOSE):
            print(self.now() + "[V] " + msg, file=self.stdout)

        if severity == TRT_LOGGER.Severity.INFO:
            print(self.now() + "[I] " + msg, file=self.stdout)

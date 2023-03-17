"""_summary_
"""

import sys
from datetime import datetime

import tensorrt as trt  # pylint: disable=import-error

TRT_LOGGER = None

if int(trt.__version__[2]) == 0:
    TRT_LOGGER = trt.Logger  # pylint: disable=no-member
else:
    TRT_LOGGER = trt.ILogger  # pylint: disable=no-member


class TrtLogger(TRT_LOGGER):
    """_summary_

    :param TRT_LOGGER: _description_
    :type TRT_LOGGER: _type_
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

    def log(self, severity, msg):
        """_summary_

        :param severity: _description_
        :type severity: _type_
        :param msg: _description_
        :type msg: _type_
        """

        current_date = f'[{datetime.now().strftime("%d/%m/%Y-%H:%M:%S")}]'

        if severity == TRT_LOGGER.Severity.ERROR:
            print(current_date + "[E] " + msg, file=self.stderr)

        if severity == TRT_LOGGER.Severity.INTERNAL_ERROR:
            print(current_date + "[INTERNAL ERROR] " + msg, file=self.stderr)

        if severity == TRT_LOGGER.Severity.WARNING:
            print(current_date + "[W] " + msg, file=self.stdout)

        if (self.min_severity == TRT_LOGGER.Severity.VERBOSE
           and severity == TRT_LOGGER.Severity.VERBOSE):
            print(current_date + "[V] " + msg, file=self.stdout)

        if severity == TRT_LOGGER.Severity.INFO:
            print(current_date + "[I] " + msg, file=self.stdout)

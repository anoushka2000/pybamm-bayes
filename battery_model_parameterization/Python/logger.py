#
# Logging (from PyBaMM)
#
import logging

FORMAT = (
    "%(asctime)s.%(msecs)03d - [%(levelname)s] %(module)s.%(funcName)s(%(lineno)d): "
    "%(message)s"
)
LOG_FORMATTER = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S", fmt=FORMAT)


def _get_new_logger(name, filename=None):
    new_logger = logging.getLogger(name)
    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename)
    handler.setFormatter(LOG_FORMATTER)
    new_logger.addHandler(handler)
    return new_logger


# Only the function for getting a new logger with filename not None is exposed
def get_new_logger(name, filename):
    if filename is None:
        raise ValueError("filename must be specified")
    return _get_new_logger(name, filename)


# Create a custom logger
logger = _get_new_logger(__name__)

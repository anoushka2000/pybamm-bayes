import logging

from csv_logger import CsvLogger

#   LOGGER    #

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


#    CSV LOGGER    #
# For solve time and solve time logging


# Create csv logger
def csv_logger(filename):
    return CsvLogger(
        filename=filename,
        delimiter=",",
        level=logging.INFO,
        fmt="%(asctime)s,%(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        max_size=1024 * 30,  # 30 kilobytes,
        header=["date", "level", "solve", "solve_time [s]"],
    )

from csv_logger import CsvLogger
from logging import Logger
import unittest

from pybamm_bayes.logging import csv_logger, logger


class TestLogging(unittest.TestCase):
    def test_exceptions(self):
        self.assertIsInstance(logger, Logger)

    def test_csv_logger(self):
        cl = csv_logger(filename="test_csv_logger")
        self.assertIsInstance(cl, CsvLogger)

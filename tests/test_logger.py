import logging
import unittest

from battery_model_parameterization.Python.logger import logger


class TestLogger(unittest.TestCase):

    def test_exceptions(self):
        self.assertIsInstance(logger, logging.Logger)

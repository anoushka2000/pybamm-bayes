import unittest
import os

here = os.path.abspath(os.path.dirname(__file__))


class TestDiagnostics(unittest.TestCase):
    sampling_problem = None

    @classmethod
    def setUpClass(cls):
        # setup variables
        pass

    def test_gelman_rubin_convergence_test(self):
        pass
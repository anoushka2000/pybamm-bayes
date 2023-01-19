import unittest
import os
from battery_model_parameterization import gelman_rubin_convergence_test

here = os.path.abspath(os.path.dirname(__file__))


class TestDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup variables
        cls.logs_dir_path = os.path.join(here, "TEST_LOGS")

    def test_gelman_rubin_convergence_test(self):
        compare = {"j0_n": 1.9093914624656876, "j0_p": 2.6124131415991827}
        result = gelman_rubin_convergence_test(
            logs_dir_path=self.logs_dir_path, burnin=0
        )
        self.assertAlmostEqual(compare["j0_n"], result["j0_n"])
        self.assertAlmostEqual(compare["j0_p"], result["j0_p"])

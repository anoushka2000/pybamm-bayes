import unittest
import os
from pybamm_bayes import gelman_rubin_convergence_test

here = os.path.abspath(os.path.dirname(__file__))


class TestDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup variables
        cls.logs_dir_path = os.path.join(here, "TEST_LOGS")

    def test_gelman_rubin_convergence_test(self):
        compare = {"j0_n": 0.03307210478247, "Ds_n": 0.0515632271658}
        result = gelman_rubin_convergence_test(
            logs_dir_path=self.logs_dir_path, burnin=0
        )
        self.assertAlmostEqual(compare["j0_n"], result["j0_n"])
        self.assertAlmostEqual(compare["Ds_n"], result["Ds_n"])

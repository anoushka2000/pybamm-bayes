import unittest

from pybamm.expression_tree.binary_operators import Multiplication

from pybamm_bayes import (electrolyte_diffusivity_Capiglia1999,
                          electrolyte_diffusivity_PeymanMPM,
                          graphite_diffusivity_PeymanMPM,
                          graphite_mcmb2528_diffusivity_Dualfoil1998,
                          lico2_diffusivity_Dualfoil1998)


class TestDiffusivityFunctions(unittest.TestCase):
    def test_electrolyte_diffusivity_Capiglia1999(self):
        c_e, T = 1, 280
        result = electrolyte_diffusivity_Capiglia1999(c_e, T)
        self.assertIsInstance(result, Multiplication)

    def test_electrolyte_diffusivity_PeymanMPM(self):
        c_e, T = 1, 280
        result = electrolyte_diffusivity_PeymanMPM(c_e, T)
        self.assertIsInstance(result, Multiplication)

    def test_graphite_mcmb2528_diffusivity_Dualfoil1998(self):
        sto, T = 0.8, 280
        result = graphite_mcmb2528_diffusivity_Dualfoil1998(sto, T)
        self.assertIsInstance(result, Multiplication)

    def test_graphite_diffusivity_PeymanMPM(self):
        sto, T = 0.8, 280
        result = graphite_diffusivity_PeymanMPM(sto, T)
        self.assertIsInstance(result, Multiplication)

    def test_lico2_diffusivity_Dualfoil1998(self):
        sto, T = 0.8, 280
        result = lico2_diffusivity_Dualfoil1998(sto, T)
        self.assertIsInstance(result, Multiplication)

import unittest

from battery_model_parameterization import (
    graphite_electrolyte_exchange_current_density_Dualfoil1998,
    graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
    lico2_electrolyte_exchange_current_density_Dualfoil1998,
    nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
    NMC_electrolyte_exchange_current_density_PeymanMPM,
    graphite_electrolyte_exchange_current_density_PeymanMPM,
)
from pybamm.expression_tree.binary_operators import Multiplication


class TestCurrentDensityFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_values = 1, 1, 280  # c_e, c_s_surf, T

    def test_lico2_electrolyte_exchange_current_density_Dualfoil1998(self):
        f = lico2_electrolyte_exchange_current_density_Dualfoil1998(
            alpha_input=False, j0_input=False
        )
        result = f(*self.test_values)
        self.assertIsInstance(result, Multiplication)

    def test_nmc_LGM50_electrolyte_exchange_current_density_Chen2020(self):
        f = nmc_LGM50_electrolyte_exchange_current_density_Chen2020(
            alpha_input=False, j0_input=False
        )
        result = f(*self.test_values)
        self.assertIsInstance(result, Multiplication)

    def NMC_electrolyte_exchange_current_density_PeymanMPM(self):
        f = NMC_electrolyte_exchange_current_density_PeymanMPM(
            alpha_input=False, j0_input=False
        )
        result = f(*self.test_values)
        self.assertIsInstance(result, Multiplication)

    def test_graphite_electrolyte_exchange_current_density_Dualfoil1998(self):
        f = graphite_electrolyte_exchange_current_density_Dualfoil1998(
            alpha_input=False, j0_input=False
        )
        result = f(*self.test_values)
        self.assertIsInstance(result, Multiplication)

    def test_graphite_LGM50_electrolyte_exchange_current_density_Chen2020(self):
        f = graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
            alpha_input=False, j0_input=False
        )
        result = f(*self.test_values)
        self.assertIsInstance(result, Multiplication)

    def test_graphite_electrolyte_exchange_current_density_PeymanMPM(self):
        f = graphite_electrolyte_exchange_current_density_PeymanMPM(
            alpha_input=False, j0_input=False
        )
        result = f(*self.test_values)
        self.assertIsInstance(result, Multiplication)

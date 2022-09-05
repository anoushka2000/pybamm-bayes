import unittest

from battery_model_parameterization import (
    graphite_electrolyte_exchange_current_density_Dualfoil1998,
    graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
    lico2_electrolyte_exchange_current_density_Dualfoil1998,
    nmc_LGM50_electrolyte_exchange_current_density_Chen2020)
from pybamm.expression_tree.binary_operators import Multiplication


class TestCurrentDensityFunctions(unittest.TestCase):
    def test_lico2_electrolyte_exchange_current_density_Dualfoil1998(self):
        c_e, c_s_surf, T = 1, 1, 280
        result = lico2_electrolyte_exchange_current_density_Dualfoil1998(
            c_e, c_s_surf, T
        )
        self.assertIsInstance(result, Multiplication)

    def test_graphite_electrolyte_exchange_current_density_Dualfoil1998(self):
        c_e, c_s_surf, T = 1, 1, 280
        result = graphite_electrolyte_exchange_current_density_Dualfoil1998(
            c_e, c_s_surf, T
        )
        self.assertIsInstance(result, Multiplication)

    def test_graphite_LGM50_electrolyte_exchange_current_density_Chen2020(self):
        c_e, c_s_surf, T = 1, 1, 280
        result = graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
            c_e, c_s_surf, T
        )
        self.assertIsInstance(result, Multiplication)

    def test_nmc_LGM50_electrolyte_exchange_current_density_Chen2020(self):
        c_e, c_s_surf, T = 1, 1, 280
        result = nmc_LGM50_electrolyte_exchange_current_density_Chen2020(
            c_e, c_s_surf, T
        )
        self.assertIsInstance(result, Multiplication)

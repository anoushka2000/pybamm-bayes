from battery_model_parameterization.Python.battery_simulation.model_setup import (
    dfn_constant_current_discharge,
)
import pybamm
import unittest


class TestModelSetup(unittest.TestCase):
    def test_dfn_constant_current_discharge(self):
        model, param = dfn_constant_current_discharge(0.1)
        model.check_well_posedness()
        self.assertIsInstance(
            model, pybamm.models.full_battery_models.lithium_ion.dfn.DFN
        )
        self.assertIsInstance(param, pybamm.parameters.parameter_values.ParameterValues)

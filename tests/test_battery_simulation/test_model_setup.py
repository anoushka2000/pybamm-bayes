import unittest

import pybamm

from battery_model_parameterization.Python.battery_models.model_setup import *


class TestModelSetup(unittest.TestCase):
    def test_default_dfn(self):
        model, param = default_dfn()
        model.check_well_posedness()
        self.assertIsInstance(
            model, pybamm.models.full_battery_models.lithium_ion.dfn.DFN
        )
        self.assertIsInstance(param, pybamm.parameters.parameter_values.ParameterValues)

    def test_default_spme(self):
        model, param = default_spme()
        model.check_well_posedness()
        self.assertIsInstance(
            model, pybamm.models.full_battery_models.lithium_ion.spme.SPMe
        )
        self.assertIsInstance(param, pybamm.parameters.parameter_values.ParameterValues)

import unittest

import pints
import pybamm
from battery_model_parameterization import (
    Variable,
    chen_2020,
    marquis_2019,
    mohtat_2020,
)


class TestModelSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)
        Dsn = Variable(name="Ds_n", value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", value=-4.698, prior=log_prior_j0_n)

        cls.variables = [Dsn, j0_n]

    def test_chen_2020(self):
        param = chen_2020(self.variables)
        self.assertIsInstance(param, pybamm.ParameterValues)

    def test_marquis_2019(self):
        param = marquis_2019(self.variables)
        self.assertIsInstance(param, pybamm.ParameterValues)

    def test_mohtat_2020(self):
        param = mohtat_2020(self.variables)
        self.assertIsInstance(param, pybamm.ParameterValues)

import unittest

import functools
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
        log_prior_D = pints.GaussianLogPrior(-13, 1)

        Dsn = Variable(name="Ds_n", value=-13.45, prior=log_prior_D)
        Dsp = Variable(name="Ds_p", value=-13.45, prior=log_prior_D)

        cls.variables = [Dsn, Dsp]

    def test_chen_2020(self):
        param = chen_2020(self.variables)
        self.assertIsInstance(param, pybamm.ParameterValues)
        self.assertIsInstance(
            param[
                "Negative electrode diffusivity [m2.s-1]"
            ], pybamm.InputParameter)
        self.assertIsInstance(
            param[
                "Positive electrode diffusivity [m2.s-1]"
            ], pybamm.InputParameter)
        self.assertIsInstance(
            param[
                "Positive electrode exchange-current density [A.m-2]"
            ], functools.partial
        )
        self.assertIsInstance(
            param[
                "Negative electrode exchange-current density [A.m-2]"
            ], functools.partial
        )

    def test_marquis_2019(self):
        param = marquis_2019(self.variables)
        self.assertIsInstance(param, pybamm.ParameterValues)
        self.assertIsInstance(
            param[
                "Positive electrode exchange-current density [A.m-2]"
            ], functools.partial
        )
        self.assertIsInstance(
            param[
                "Negative electrode exchange-current density [A.m-2]"
            ], functools.partial
        )

    def test_mohtat_2020(self):
        param = mohtat_2020(self.variables)
        self.assertIsInstance(param, pybamm.ParameterValues)
        self.assertIsInstance(
            param[
                "Negative electrode diffusivity [m2.s-1]"
            ], pybamm.InputParameter)
        self.assertIsInstance(
            param[
                "Positive electrode exchange-current density [A.m-2]"
            ], functools.partial
        )
        self.assertIsInstance(
            param[
                "Negative electrode exchange-current density [A.m-2]"
            ], functools.partial
        )

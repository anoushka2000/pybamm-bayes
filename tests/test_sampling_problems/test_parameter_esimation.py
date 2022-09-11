import os
import shutil
import unittest

import pandas as pd
import pints
import pybamm
from battery_model_parameterization import (ParameterEstimation, Variable,
                                            marquis_2019)

here = os.path.abspath(os.path.dirname(__file__))


class TestParameterEstimation(unittest.TestCase):
    parameter_esimation_problem = None

    @classmethod
    def setUpClass(cls):

        # setup variables
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)
        Dsn = Variable(name="Ds_n", value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", value=-4.698, prior=log_prior_j0_n)
        cls.variables = [Dsn, j0_n]

        # setup battery simulation
        model = pybamm.lithium_ion.DFN()
        cls.parameter_values = marquis_2019(cls.variables)
        cls.simulation = pybamm.Simulation(
            model,
            solver=pybamm.CasadiSolver("fast"),
            experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
        )

        cls.parameter_esimation_problem = ParameterEstimation(
            data=pd.read_csv("test_data.csv"),
            battery_simulation=cls.simulation,
            parameter_values=cls.parameter_values,
            variables=cls.variables,
            transform_type="log10",
            project_tag="test",
        )

    def test_init_(self):
        # test data generation
        self.assertEqual(
            len(self.parameter_esimation_problem.data),
            len(self.parameter_esimation_problem.times),
        )

    def test_metadata(self):
        self.assertIsInstance(self.parameter_esimation_problem.metadata, dict)

    def test_initial_values(self):
        self.assertEqual(
            self.parameter_esimation_problem.initial_values.shape[0],
            len(self.parameter_esimation_problem.variables),
        )

    def test_simulate(self):
        output = self.parameter_esimation_problem.simulate(
            self.parameter_esimation_problem.initial_values,
            self.parameter_esimation_problem.times,
        )
        self.assertFalse(output is None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TestParameterEstimation.identifiability_problem.logs_dir_path)

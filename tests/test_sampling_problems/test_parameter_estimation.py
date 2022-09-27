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
    parameter_estimation_problem = None

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

        cls.parameter_estimation_problem = ParameterEstimation(
            data=pd.read_csv(os.path.join(here, "test_data.csv")),
            battery_simulation=cls.simulation,
            parameter_values=cls.parameter_values,
            variables=cls.variables,
            transform_type="log10",
            project_tag="test",
        )

    def test_init_(self):
        # test data generation
        self.assertEqual(
            len(self.parameter_estimation_problem.data),
            len(self.parameter_estimation_problem.times),
        )

    def test_metadata(self):
        self.assertIsInstance(self.parameter_estimation_problem.metadata, dict)

    def test_initial_values(self):
        self.assertEqual(
            self.parameter_estimation_problem.initial_values.shape[0],
            len(self.parameter_estimation_problem.variables) + 1,
        )

    def test_simulate(self):
        output = self.parameter_estimation_problem.simulate(
            self.parameter_estimation_problem.initial_values,
            self.parameter_estimation_problem.times,
        )
        self.assertFalse(output is None)

    def test_run(self):
        burnin = 2
        n_iteration = 5
        n_chains = 3
        n_workers = 3

        chains = self.parameter_estimation_problem.run(
            burnin, n_iteration, n_chains, n_workers
        )

        self.assertEqual(len(chains.columns), len(self.variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(
            TestParameterEstimation.parameter_estimation_problem.logs_dir_path
        )

import os
import shutil
import unittest

import pints
import pybamm
from battery_model_parameterization import (IdentifiabilityAnalysis, Variable,
                                            marquis_2019)

here = os.path.abspath(os.path.dirname(__file__))


class TestIdentifiabilityAnalysis(unittest.TestCase):
    identifiability_problem = None

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

        cls.identifiability_problem = IdentifiabilityAnalysis(
            battery_simulation=cls.simulation,
            parameter_values=cls.parameter_values,
            variables=cls.variables,
            transform_type="log10",
            noise=0.00,
            project_tag="test",
        )

    def test_init_(self):
        # test generated_data flag
        self.assertEqual(self.identifiability_problem.generated_data, True)
        # test data generation
        self.assertEqual(
            len(self.identifiability_problem.data),
            len(self.identifiability_problem.times),
        )

    def test_metadata(self):
        self.assertIsInstance(self.identifiability_problem.metadata, dict)

    def test_true_values(self):
        self.assertEqual(
            self.identifiability_problem.true_values.shape[0],
            len(self.identifiability_problem.variables),
        )

    def test_simulate(self):
        output = self.identifiability_problem.simulate(
            self.identifiability_problem.true_values, self.identifiability_problem.times
        )
        self.assertFalse(output is None)

    def test_run(self):
        burnin = 2
        n_iteration = 5
        n_chains = 3
        n_workers = 3

        chains = self.identifiability_problem.run(
            burnin, n_iteration, n_chains, n_workers
        )

        self.assertEqual(len(chains.columns), len(self.variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TestIdentifiabilityAnalysis.identifiability_problem.logs_dir_path)

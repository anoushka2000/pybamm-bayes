import os
import shutil
import unittest

import pandas as pd
import pints
import pybamm
from pybamm_bayes import BaseSamplingProblem, Variable, marquis_2019
from matplotlib.testing.compare import compare_images

here = os.path.abspath(os.path.dirname(__file__))


class TestBaseSamplingProblem(unittest.TestCase):
    sampling_problem = None

    @classmethod
    def setUpClass(cls):
        # setup variables
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)
        Dsn = Variable(name="Ds_n", value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", value=-4.698, prior=log_prior_j0_n)
        cls.variables = [Dsn, j0_n]

        # setup battery simulation
        model = pybamm.lithium_ion.SPMe()
        cls.parameter_values = marquis_2019(cls.variables)
        cls.simulation = pybamm.Simulation(
            model,
            solver=pybamm.CasadiSolver("fast"),
            experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
        )

        cls.sampling_problem = BaseSamplingProblem(
            battery_simulation=cls.simulation,
            parameter_values=cls.parameter_values,
            variables=cls.variables,
            output="Terminal voltage [V]",
            transform_type="log10",
            project_tag="test",
        )
        cls.test_data = pd.read_csv(os.path.join(here, "test_data.csv"))

    def test_create_logs_dir(self):
        self.assertTrue(os.path.exists(self.sampling_problem.logs_dir_path))

    def test_log_prior(self):
        self.assertIsInstance(
            self.sampling_problem.log_prior, pints._log_priors.ComposedLogPrior
        )
        self.assertEqual(
            self.sampling_problem.log_prior.n_parameters(),
            len(self.sampling_problem.variables),
        )

    def test_plot_priors(self):
        expected = os.path.join(here, "baseline_plots", "prior.png")
        self.sampling_problem.plot_priors()
        actual = os.path.join(self.sampling_problem.logs_dir_path, "prior.png")
        compare_images(expected, actual, tol=0.5)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.sampling_problem.logs_dir_path)

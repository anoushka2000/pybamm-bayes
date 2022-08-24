import os
import shutil
import unittest

import pints
import pybamm
from matplotlib.testing.compare import compare_images

from battery_model_parameterization.Python.battery_simulation.model_setup import \
    dfn_constant_current_discharge
from battery_model_parameterization.Python.identifiability_problem import \
    IdentifiabilityProblem
from battery_model_parameterization.Python.variable import Variable


class TestIdentifiabilityProblem(unittest.TestCase):
    identifiability_problem = None

    @classmethod
    def setUpClass(cls):
        # setup
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)
        Dsn = Variable(name="Ds_n", true_value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", true_value=-4.698, prior=log_prior_j0_n)

        cls.variables = [Dsn, j0_n]
        cls.battery_model, cls.parameter_values = dfn_constant_current_discharge(
            d_rate=0.1
        )
        cls.timespan = 60 * 60 * 10
        TestIdentifiabilityProblem.identifiability_problem = IdentifiabilityProblem(
            battery_model=cls.battery_model,
            variables=cls.variables,
            parameter_values=cls.parameter_values,
            transform_type="log10",
            resolution=10,
            timespan=cls.timespan,
            noise=0.00,
        )

    def test_init_(self):
        # test generated_data flag
        self.assertEqual(self.identifiability_problem.generated_data, True)
        # test data generation
        self.assertEqual(
            len(self.identifiability_problem.data),
            len(self.identifiability_problem.times),
        )
        # test time array generation
        self.assertEqual(
            self.identifiability_problem.times.max(),
            self.identifiability_problem.timespan,
        )

    def test_battery_simulation(self):
        self.assertIsInstance(
            self.identifiability_problem.battery_simulation,
            pybamm.simulation.Simulation,
        )
        self.assertDictEqual(
            self.identifiability_problem.battery_simulation.parameter_values,
            self.identifiability_problem.parameter_values,
        )

    def test_create_logs_dir(self):
        self.assertTrue(os.path.exists(self.identifiability_problem.logs_dir_path))

    def test_log_prior(self):
        self.assertIsInstance(
            self.identifiability_problem.log_prior, pints._log_priors.ComposedLogPrior
        )
        self.assertEqual(
            self.identifiability_problem.log_prior.n_parameters(),
            len(self.identifiability_problem.variables),
        )

    def test_metadata(self):
        self.assertIsInstance(self.identifiability_problem.metadata, dict)

    def test_true_values(self):
        self.assertEqual(
            self.identifiability_problem.true_values.shape[0],
            len(self.identifiability_problem.variables),
        )

    def setup_battery_simulation(self):
        # battery_model
        self.assertIsInstance(
            self.identifiability_problem.battery_model, pybamm.BaseBatteryModel
        )
        # parameter_values
        self.assertIsInstance(
            self.identifiability_problem.parameter_values,
            pybamm.parameters.parameter_values.ParameterValues,
        )

    def test_simulate(self):
        output = self.identifiability_problem.simulate(self.true_values, self.times)
        self.assertFalse(output is None)

    def test_plot_priors(self):
        expected = os.path.join("baseline_plots", "prior")
        actual = os.path.join(self.identifiability_problem.logs_dir_path, "prior")
        compare_images(expected, actual, tol=0.2)

    def test_plot_data(self):
        expected = os.path.join("baseline_plots", "data")
        actual = os.path.join(self.identifiability_problem.logs_dir_path, "data")
        compare_images(expected, actual)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TestIdentifiabilityProblem.identifiability_problem.logs_dir_path)

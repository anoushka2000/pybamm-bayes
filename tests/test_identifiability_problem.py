import unittest
import pints
import pybamm

from battery_model_parameterization.Python.battery_simulation.model_setup import (
    dfn_constant_current_discharge,
)
from battery_model_parameterization.Python.identifiability_problem import (
    IdentifiabilityProblem,
)
from battery_model_parameterization.Python.variable import Variable


class TestIdentifiabilityProblem(unittest.TestCase):

    def test_init_(self):
        # setup
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)
        Dsn = Variable(name="Ds_n", true_value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", true_value=-4.698, prior=log_prior_j0_n)

        variables = [Dsn, j0_n]
        battery_model, parameter_values = dfn_constant_current_discharge(d_rate=0.1)
        ten_hours = 60 * 60 * 10
        identifiability_problem = IdentifiabilityProblem(
            battery_model=battery_model,
            variables=variables,
            parameter_values=parameter_values,
            transform_type="log10",
            resolution=10,
            timespan=ten_hours,
            noise=0.005,
        )

        # test generated_data flag
        self.assertEqual(identifiability_problem.generated_data, True)

        # test data generation
        self.assertEqual(len(identifiability_problem.data), len(identifiability_problem.times))

        # test properties from args
        ## battery_model
        self.assertIsInstance(identifiability_problem.battery_model, pybamm.BaseBatteryModel)
        ## parameter_values
        self.assertIsInstance(identifiability_problem.parameter_values,
                              pybamm.parameters.parameter_values.ParameterValues)
        ## variables
        for var in identifiability_problem.variables:
            self.assertIsInstance(var, Variable)
        ## resolution vs timespan
        self.assertLess(identifiability_problem.resolution, identifiability_problem.timespan)
        ## noise vs data range
        self.assertLess(identifiability_problem.noise * 2, identifiability_problem.data.max())

        # test internally set properties
        ## log_prior
        self.assertIsInstance(identifiability_problem.log_prior, pints._log_priors.ComposedLogPrior)
        self.assertEqual(identifiability_problem.log_prior.n_parameters(), len(identifiability_problem.variables))
        ## true_values
        self.assertEqual(identifiability_problem.true_values.shape[0], len(identifiability_problem.variables))
        ## times
        self.assertEqual(identifiability_problem.times.max(), identifiability_problem.timespan)
        ## battery_simulation
        self.assertIsInstance(identifiability_problem.battery_simulation, pybamm.simulation.Simulation)
        self.assertDictEqual(identifiability_problem.battery_simulation.parameter_values,
                             identifiability_problem.parameter_values)

        def test_simulate(self):
            # theta, times =
            pass

        def test_plot_priors(self):
            pass

        def test_plot_data(self):
            pass

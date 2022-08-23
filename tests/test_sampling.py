import pints
import unittest

import pints
from battery_model_parameterization.Python import sampling
from battery_model_parameterization.Python.battery_simulation.model_setup import \
    dfn_constant_current_discharge
from battery_model_parameterization.Python.identifiability_problem import \
    IdentifiabilityProblem
from battery_model_parameterization.Python.variable import Variable


class TestSampling(unittest.TestCase):
    def test_run_mcmc(self):
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)

        Dsn = Variable(name="Ds_n", true_value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", true_value=-4.698, prior=log_prior_j0_n)
        variables = [Dsn, j0_n]

        model, param = dfn_constant_current_discharge(d_rate=0.1)

        ten_hours = 60 * 60 * 10

        identifiability_problem = IdentifiabilityProblem(
            model,
            variables,
            parameter_values=param,
            transform_type="log10",
            resolution=10,
            timespan=ten_hours,
            noise=0.005,
        )

        burnin = 2
        n_iteration = 5
        n_chains = 3
        n_workers = 3

        chains = sampling.run_mcmc(
            identifiability_problem, burnin, n_iteration, n_chains, n_workers
        )

        self.assertEqual(len(chains.columns), len(variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

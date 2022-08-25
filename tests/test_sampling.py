import unittest

import pints

from battery_model_parameterization.Python import sampling
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

        identifiability_problem = IdentifiabilityProblem(
            battery_model="default_dfn",
            variables=variables,
            operating_conditions=["Discharge at C/10 for 10 hours"],
            project_tag="test_exp",
            transform_type="log10",
            resolution="10 minutes",
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

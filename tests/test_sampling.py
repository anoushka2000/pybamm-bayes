import unittest

import pints
from battery_model_parameterization import (IdentifiabilityProblem, Variable,
                                            run_mcmc)


class TestSampling(unittest.TestCase):
    def test_run_mcmc(self):
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)

        Dsn = Variable(name="Ds_n", true_value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", true_value=-4.698, prior=log_prior_j0_n)
        variables = [Dsn, j0_n]

        identifiability_problem = IdentifiabilityProblem(
            battery_model="default_dfn",
            operating_conditions=[
                "Discharge at C/10 for 10 hours",
            ],
            variables=variables,
            transform_type="log10",
            resolution="1 minute",
            noise=0.005,
            project_tag="test",
        )

        burnin = 2
        n_iteration = 5
        n_chains = 3
        n_workers = 3

        chains = run_mcmc(
            identifiability_problem, burnin, n_iteration, n_chains, n_workers
        )

        self.assertEqual(len(chains.columns), len(variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

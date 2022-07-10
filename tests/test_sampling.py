import unittest

import pints

from battery_model_parameterization.Python import sampling
from battery_model_parameterization.Python.battery_simulation.model_setup import \
    dfn_constant_current_discharge
from battery_model_parameterization.Python.identifiability_problem import \
    IdentifiabilityProblem
from battery_model_parameterization.Python.variable import Variable

CHAIN_RECORD = [
    (0, -16.74293161, -5.8482002),
    (1, -16.52496226, -5.85054706),
    (2, -15.96513726, -5.82824975),
    (3, -15.89443343, -5.56774417),
    (4, -15.89443343, -5.56774417),
    (5, -15.92989206, -5.56421062),
    (6, -15.66723845, -5.63038721),
    (7, -15.34968708, -5.16882776),
    (8, -15.23981286, -5.3478517),
    (9, -15.23981286, -5.3478517),
    (10, -14.65790656, -5.11991413),
    (11, -14.65790656, -5.11991413),
    (12, -14.47355606, -5.0526986),
    (13, -14.47355606, -5.0526986),
    (14, -14.12038002, -4.95777944),
]


class TestSampling(unittest.TestCase):
    def test_run_mcmc(self):
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)

        Dsn = Variable(name="Ds_n", true_value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", true_value=-4.698, prior=log_prior_j0_n)
        variables = [Dsn, j0_n]

        model, param = dfn_constant_current_discharge(d_rate=0.1)

        ten_hours = 60 * 60 * 10

        identifiability_problem = IdentifiabilityProblem(model, variables, parameter_values=param,
                                                         transform_type="log10",
                                                         resolution=10, timespan=ten_hours, noise=0.005)
        identifiability_problem.plot_data()
        identifiability_problem.plot_priors()

        burnin = 2
        n_iteration = 5
        n_chains = 3
        n_workers = 3

        chains = sampling.run_mcmc(identifiability_problem, burnin, n_iteration, n_chains, n_workers)

        self.assertEqual(len(chains.columns), len(variables))
        self.assertEqual(len(chains), n_iteration*n_chains)
        self.assertEqual(CHAIN_RECORD, chains.to_records())
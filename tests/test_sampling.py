import unittest
import pybamm
import pints
import numpy as np
from battery_model_parameterization import (IdentifiabilityAnalysis,
                                            ParameterEstimation,
                                            Variable,
                                            marquis_2019,
                                            run_identifiability_analysis,
                                            run_parameter_estimation)


class TestSampling(unittest.TestCase):

    def setUpClass(cls):
        # setup variables
        log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
        log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)
        Dsn = Variable(name="Ds_n", value=-13.45, prior=log_prior_Dsn)
        j0_n = Variable(name="j0_n", value=-4.698, prior=log_prior_j0_n)

        cls.variables = [Dsn, j0_n]

        # setup battery simulation
        model = pybamm.lithium_ion.DFN()
        cls.parameter_values = marquis_2019()
        cls.simulation = pybamm.Simulation(model,
                                           solver=pybamm.CasadiSolver("fast"),
                                           experiment=pybamm.Experiment(["Discharge at C/10 for 1 hour"]),
                                           )

    def test_run_identifiability_analysis(self):

        identifiability_problem = IdentifiabilityAnalysis(
            battery_simulation=self.simulation,
            variables=self.variables,
            transform_type="log10",
            noise=0.005,
            times=np.linspace(0, 3600, 360),
            project_tag="test",
        )

        burnin = 2
        n_iteration = 5
        n_chains = 3
        n_workers = 3

        chains = run_identifiability_analysis(
            identifiability_problem, burnin, n_iteration, n_chains, n_workers
        )

        self.assertEqual(len(chains.columns), len(self.variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

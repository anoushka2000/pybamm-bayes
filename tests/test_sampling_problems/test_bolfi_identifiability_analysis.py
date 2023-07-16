import os
import shutil
import unittest

import elfi
import pybamm
from scipy.spatial.distance import cdist
from pybamm_bayes import (
    BOLFIIdentifiabilityAnalysis,
    Variable,
    marquis_2019,
)

here = os.path.abspath(os.path.dirname(__file__))


class TestBOLFIIdentifiabilityAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup variables
        prior_Ds_n = elfi.Prior("norm", 13.1, 1, name="Ds_n")
        prior_Ds_p = elfi.Prior("norm", 12.5, 1, name="Ds_p")

        Ds_n = Variable(name="Ds_n", value=12.5, prior=prior_Ds_n, bounds=(12, 14))
        Ds_p = Variable(name="Ds_p", value=13, prior=prior_Ds_p, bounds=(12, 14))

        cls.variables = [Ds_n, Ds_p]

        # setup battery simulation
        model = pybamm.lithium_ion.SPMe()
        cls.parameter_values = marquis_2019(cls.variables)
        cls.simulation = pybamm.Simulation(
            model,
            parameter_values=cls.parameter_values,
            experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
        )

        cls.identifiability_problem = BOLFIIdentifiabilityAnalysis(
            battery_simulation=cls.simulation,
            parameter_values=cls.parameter_values,
            variables=cls.variables,
            output="Terminal voltage [V]",
            transform_type="negated_log10",
            noise=0.001,
            target_resolution=30,
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

    def test_custom_discrepancy_metrics(self):
        func = self.identifiability_problem.discrepancy_metrics["wasserstein_distance"]
        distance = func([0, 1, 3], [5, 6, 8])
        self.assertEqual(5, distance)

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
        n_iteration = 200
        n_chains = 4
        n_evidence = 500

        chains = self.identifiability_problem.run(
            sampling_iterations=n_iteration,
            n_chains=n_chains,
            n_evidence=n_evidence,
        )

        self.assertEqual(len(chains.columns), len(self.variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

    def test_run_with_callable_discrepancy(self):
        n_iteration = 200
        n_chains = 4
        n_evidence = 500

        chains = self.identifiability_problem.run(
            sampling_iterations=n_iteration,
            n_chains=n_chains,
            n_evidence=n_evidence,
            discrepancy_metric=lambda x, y: cdist(x, y, metric="euclidean"),
        )

        self.assertEqual(len(chains.columns), len(self.variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

    def test_plot_pairwise(self):
        self.identifiability_problem.plot_pairwise()
        self.assertTrue(
            os.path.join(self.identifiability_problem.logs_dir_path, "pairwise_plot")
        )

    def test_plot_discrepancy(self):
        self.identifiability_problem.plot_discrepancy()
        self.assertTrue(
            os.path.join(self.identifiability_problem.logs_dir_path, "discrepancy")
        )

    def test_acquistion_surface(self):
        self.identifiability_problem.plot_acquistion_surface()
        self.assertTrue(
            os.path.join(self.identifiability_problem.logs_dir_path, "acquisition_surface")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(
            TestBOLFIIdentifiabilityAnalysis.identifiability_problem.logs_dir_path
        )

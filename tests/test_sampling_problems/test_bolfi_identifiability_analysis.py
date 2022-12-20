import os
import shutil
import unittest

import elfi
import pybamm
from battery_model_parameterization import (
    BOLFIIdentifiabilityAnalysis,
    Variable,
    marquis_2019,
)

here = os.path.abspath(os.path.dirname(__file__))


class TestBOLFIIdentifiabilityAnalysis(unittest.TestCase):
    identifiability_problem = None

    @classmethod
    def setUpClass(cls):
        # setup variables

        prior_j0_n = elfi.Prior("norm", 5.5, 0.5, name="j0_n")
        prior_j0_p = elfi.Prior("norm", 6.5, 0.5, name="j0_p")
        j0_n = Variable(name="j0_n", value=4.698, prior=prior_j0_n, bounds=(4, 6))
        j0_p = Variable(name="j0_p", value=6.22, prior=prior_j0_p, bounds=(5, 7))

        cls.variables = [j0_n, j0_p]

        # setup battery simulation
        model = pybamm.lithium_ion.SPMe()
        cls.parameter_values = marquis_2019(cls.variables)
        cls.simulation = pybamm.Simulation(
            model,
            experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
        )

        cls.identifiability_problem = BOLFIIdentifiabilityAnalysis(
            battery_simulation=cls.simulation,
            parameter_values=cls.parameter_values,
            variables=cls.variables,
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
        n_iteration = 50
        n_chains = 4
        n_evidence = 1500

        chains = self.identifiability_problem.run(
            sampling_iterations=n_iteration, n_chains=n_chains,
            n_evidence=n_evidence,
        )

        self.assertEqual(len(chains.columns), len(self.variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(
            TestBOLFIIdentifiabilityAnalysis.identifiability_problem.logs_dir_path
        )

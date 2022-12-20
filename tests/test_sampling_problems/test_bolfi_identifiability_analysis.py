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

        prior_Ds_n = elfi.Prior("uniform", -12, -10, name="Ds_n")
        prior_Ds_p = elfi.Prior("uniform", -12, -10, name="Ds_p")

        Ds_n = Variable(name="Ds_n", value=-11, prior=prior_Ds_n, bounds=(-12, -10))
        Ds_p = Variable(name="Ds_p", value=-11, prior=prior_Ds_p, bounds=(-12, -10))

        cls.variables = [Ds_n, Ds_p]

        # setup battery simulation
        model = pybamm.lithium_ion.SPMe()
        cls.parameter_values = marquis_2019(cls.variables)
        cls.simulation = pybamm.Simulation(
            model,
            solver=pybamm.CasadiSolver("fast"),
            experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
        )

        cls.identifiability_problem = BOLFIIdentifiabilityAnalysis(
            battery_simulation=cls.simulation,
            parameter_values=cls.parameter_values,
            variables=cls.variables,
            transform_type="log10",
            noise=0.005,
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
        n_iteration = 5
        n_chains = 3

        chains = self.identifiability_problem.run(
            sampling_iterations=n_iteration, n_chains=n_chains
        )

        self.assertEqual(len(chains.columns), len(self.variables))
        self.assertEqual(len(chains), n_iteration * n_chains)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TestIdentifiabilityAnalysis.identifiability_problem.logs_dir_path)

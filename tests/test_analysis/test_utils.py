import unittest
import os
from battery_model_parameterization.Python.analysis.postprocessing import (
    load_metadata,
    load_chains,
)
from battery_model_parameterization.Python.analysis.utils import (
    sample_from_prior,
    sample_from_posterior,
)

here = os.path.abspath(os.path.dirname(__file__))


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup variables
        cls.chains = load_chains(logs_dir_path=os.path.join(here, "TEST_LOGS"))
        cls.metadata = load_metadata(logs_dir_path=os.path.join(here, "TEST_LOGS"))

    def test_sample_from_prior(self):
        # Gaussian Log Priors with:
        #   "mean": -5.5, "sd": 1.0
        #   "mean": -6.5, "sd": 1.0
        samples = sample_from_prior(metadata=self.metadata, n_samples=7000)
        self.assertAlmostEqual(list(samples.values())[0].mean(), -5.5, places=1)
        self.assertAlmostEqual(list(samples.values())[1].mean(), -6.5, places=1)
        self.assertAlmostEqual(list(samples.values())[0].std(), 1.0, places=1)
        self.assertAlmostEqual(list(samples.values())[1].std(), 1.0, places=1)

    def test_sample_from_posterior(self):
        samples = sample_from_posterior(self.chains, n_samples=7000)

        self.assertAlmostEqual(
            self.chains.mean().values[0], samples[:, 0].mean(), delta=0.11
        )
        self.assertAlmostEqual(
            self.chains.mean().values[1], samples[:, 1].mean(), delta=0.11
        )

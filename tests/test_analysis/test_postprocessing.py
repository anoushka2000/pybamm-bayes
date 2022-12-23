import unittest
import os
import pandas as pd
from battery_model_parameterization.Python.analysis.postprocessing import (
    load_metadata,
    load_chains,
    load_chains_with_residual,
    generate_residual_over_posterior,
    run_forward_model_over_posterior,
)

here = os.path.abspath(os.path.dirname(__file__))


class TestPostprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs_dir_path = os.path.join(here, "TEST_LOGS")

    def test_load_metadata(self):
        metadata = load_metadata(logs_dir_path=self.logs_dir_path)
        self.assertIsInstance(metadata, dict)
        self.assertIn("battery model", metadata)
        self.assertIn("parameter values", metadata)

    def test_load_chains(self):
        chains = load_chains(logs_dir_path=self.logs_dir_path)
        self.assertIsInstance(chains, pd.DataFrame)

    def test_load_chains_with_residual(self):
        chains = load_chains_with_residual(logs_dir_path=self.logs_dir_path)
        self.assertIsInstance(chains, pd.DataFrame)

    def test_generate_residual_over_posterior(self):
        generated = generate_residual_over_posterior(logs_dir_path=self.logs_dir_path)
        compare = pd.read_csv(
            os.path.join(self.logs_dir_path, "residual_over_posterior.csv"),
            index_col=False,
        )
        compare.drop(
            columns=[
                "Unnamed: 0",
            ],
            inplace=True,
        )
        pd.testing.assert_index_equal(generated.columns, compare.columns)

    def test_run_forward_model_over_posterior(self):
        generated = run_forward_model_over_posterior(logs_dir_path=self.logs_dir_path)
        compare = pd.read_csv(
            os.path.join(self.logs_dir_path, "forward_model_over_posterior.csv"),
            index_col=False,
        )
        compare.drop(
            columns=[
                "Unnamed: 0",
            ],
            inplace=True,
        )
        pd.testing.assert_index_equal(generated.columns, compare.columns)

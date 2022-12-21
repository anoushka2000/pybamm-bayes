import unittest
import os

here = os.path.abspath(os.path.dirname(__file__))


class TestPostprocessing(unittest.TestCase):
    sampling_problem = None

    @classmethod
    def setUpClass(cls):
        # setup variables
        pass

    def test_view_data(self):
        pass

    def test_load_metadata(self):
        pass

    def test_load_chains(self):
        pass

    def test_load_chains_with_residual(self):
        pass

    def test_generate_residual_over_posterior(self):
        pass

    def test_run_forward_model_over_posterior(self):
        pass

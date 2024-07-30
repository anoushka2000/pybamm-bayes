import os
import unittest

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from pybamm_bayes import (compare_chain_convergence, pairwise,
                          plot_chain_convergence, plot_confidence_intervals,
                          plot_forward_model_posterior_distribution,
                          plot_residual)

here = os.path.abspath(os.path.dirname(__file__))


class TestPlotting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup variables
        cls.logs_dir_path = os.path.join(here, "TEST_LOGS")
        cls.images_dir_path = os.path.join(here, "test_images")

    def test_plot_chain_convergence(self):
        plot_chain_convergence(logs_dir_path=self.logs_dir_path)
        baseline_plot = os.path.join(self.images_dir_path, "chain_convergence.png")
        generated_plot = os.path.join(self.logs_dir_path, "chain_convergence.png")
        compare_images(baseline_plot, generated_plot, 0.001)
        os.remove(generated_plot)

    def test_compare_chain_convergence(self):
        compare_chain_convergence(
            log_dir_paths=[
                self.logs_dir_path,
            ]
        )
        baseline_plot = os.path.join(
            self.images_dir_path, "comparison_chain_convergence.png"
        )
        generated_plot = os.path.join(
            self.logs_dir_path, "comparison_chain_convergence.png"
        )
        compare_images(baseline_plot, generated_plot, 0.001)
        os.remove(generated_plot)

    def test_pairwise(self):
        pairwise(logs_dir_path=self.logs_dir_path)
        baseline_plot = os.path.join(self.images_dir_path, "pairwise_correlation.png")
        generated_plot = os.path.join(self.logs_dir_path, "pairwise_correlation.png")
        compare_images(baseline_plot, generated_plot, 0.001)
        os.remove(generated_plot)

    def test_plot_confidence_intervals(self):
        fig = plot_confidence_intervals(logs_dir_path=self.logs_dir_path)
        baseline_plot = os.path.join(self.images_dir_path, "confidence_intervals.png")
        generated_plot = os.path.join(self.logs_dir_path, "confidence_intervals.png")
        fig.write_image(generated_plot, height=525, width=985)
        compare_images(baseline_plot, generated_plot, 0.001)
        os.remove(generated_plot)

    def test_plot_residual(self):
        plot_residual(logs_dir_path=self.logs_dir_path)
        baseline_plot = os.path.join(self.images_dir_path, "residual.png")
        generated_plot = os.path.join(self.logs_dir_path, "residual.png")
        plt.savefig(generated_plot)
        plt.clf()
        compare_images(baseline_plot, generated_plot, 0.001)
        os.remove(generated_plot)

    def test_plot_forward_model_posterior_distribution(self):
        plot_forward_model_posterior_distribution(logs_dir_path=self.logs_dir_path)
        baseline_plot = os.path.join(self.images_dir_path, "forward.png")
        generated_plot = os.path.join(self.logs_dir_path, "forward.png")
        plt.savefig(generated_plot)
        plt.clf()
        compare_images(baseline_plot, generated_plot, 0.001)
        os.remove(generated_plot)

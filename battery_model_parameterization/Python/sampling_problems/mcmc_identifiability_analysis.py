import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import pints
import pybamm
from scipy.interpolate import interp1d

from battery_model_parameterization.Python.sampling_problems.base_sampling_problem import (  # noqa: E501
    BaseSamplingProblem,
)
from battery_model_parameterization.Python.variable import Variable
from battery_model_parameterization.Python.sampling_problems.utils import (
    _fmt_variables,
    _fmt_parameters,
)


class MCMCIdentifiabilityAnalysis(BaseSamplingProblem):
    """
    Class for conducting non-linear identifiability analysis on
    battery simulation parameters using Markov Chain Monte Carlo (MCMC)
    methods.

    Parameters
    ----------
    battery_simulation: pybamm.Simulation
        Battery simulation for which parameter identifiability is being tested.
    parameter_values: pybamm.ParameterValues
        Parameter values for the simulation.
    variables: List[Variable]
        List of variables being identified in problem.
        Each variable listed in `variables` must be initialized
        as a pybamm.InputParameter in `parameter_values`.
    transform_type: str
        Transformation variable value input to battery model
        and sampling space.
        (only `log10` implemented for now)
    noise: float
        Scale of zero-mean noise added to synthetic data used to identify parameters.
    times: np.ndarray
        Array of times at which simulation is evaluated.
    project_tag: str
        Project identifier (prefix to logs directory name).
    """

    def __init__(
            self,
            battery_simulation: pybamm.Simulation,
            parameter_values: pybamm.ParameterValues,
            variables: List[Variable],
            transform_type: str,
            noise: float,
            error_axis: str = "y",
            times: Optional[np.ndarray] = None,
            project_tag: str = "",
    ):

        super().__init__(
            battery_simulation=battery_simulation,
            parameter_values=parameter_values,
            variables=variables,
            transform_type=transform_type,
            project_tag=project_tag,
        )

        self.method = "MCMC"
        self.generated_data = False
        self.true_values = np.array([v.value for v in self.variables])
        self.noise = noise
        self.error_axis = error_axis

        if battery_simulation.operating_mode == "without experiment":
            if times is None:
                raise ValueError(
                    """If battery simulation is not operated using an experiment,\n
                an array of times to evaluate simulation at must be passed."""
                )

            self.times = times

        else:
            inputs = dict(
                zip([v.name for v in variables], [v.value for v in variables])
            )
            battery_simulation.solve(inputs=inputs)
            self.times = battery_simulation.solution["Time [s]"].entries

        data = self.simulate(theta=self.true_values, times=self.times)
        self.observable = data + np.random.normal(0, self.noise, data.shape)

        if self.error_axis == "y":
            self.data = self.observable
        else:
            self.data = self._interpolate_time_over_y_values(times=self.times, y_values=self.observable)

        with open(os.path.join(self.logs_dir_path, "metadata.json"), "w") as outfile:
            outfile.write(json.dumps(self.metadata))

        self.battery_simulation.save(
            os.path.join(self.logs_dir_path, "battery_simulation")
        )

    @property
    def metadata(self):
        return {
            "battery model": self.battery_simulation.model.name,
            "parameter values": _fmt_parameters(self.parameter_values),
            "default inputs": self.default_inputs,
            "variables": _fmt_variables(self.variables),
            "transform type": self.transform_type,
            "noise": self.noise,
            "project": self.project_tag,
            "times": str(self.times),
            "data": str(self.observable),
        }

    def _interpolate_time_over_y_values(self, times, y_values):
        """
        Parameters
        ----------
        times: np.ndarray
            Times at which simulation was evaluated.
        y_values: np.ndarray
            Time series simulation output.

        Returns
        ----------
        new_time: np.ndarray
            Time interpolated over y-axis values.
        """
        y_function = interp1d(x=y_values, y=times)
        min_y = y_values.min()
        max_y = y_values.max()
        new_y = np.linspace(start=min_y * (1 + 1e-8), stop=max_y * (1 - 1e-8), num=int(y_values.shape[0]))
        return y_function(new_y)

    def simulate(self, theta, times):
        """
        Simulate method used by pints sampler.
        Parameters
        ----------
        theta: np.ndarray
            Vector of input variable values.
        times: np.ndarray
            Array of times (in seconds) at which model is solved.
        Returns
        ----------
        output: np.ndarray
            Voltage time series.
        """
        variable_names = [v.name for v in self.variables]

        inputs = dict(zip(variable_names, [self.inverse_transform(t) for t in theta]))

        try:
            # solve with CasadiSolver
            self.battery_simulation.solve(
                inputs=inputs, solver=pybamm.CasadiSolver("fast"), t_eval=self.times
            )
            solution = self.battery_simulation.solution
            V = solution["Terminal voltage [V]"]
            output = V.entries

        except pybamm.SolverError:
            # CasadiSolver "fast" failed
            try:
                self.battery_simulation.solve(
                    inputs=inputs, solver=pybamm.CasadiSolver("safe"), t_eval=self.times
                )
                solution = self.battery_simulation.solution
                V = solution["Terminal voltage [V]"]
                output = V.entries

            except pybamm.SolverError:
                #  ScipySolver solver failed
                try:
                    self.battery_simulation.solve(
                        inputs=inputs, solver=pybamm.ScipySolver(), t_eval=self.times
                    )
                    solution = self.battery_simulation.solution
                    V = solution["Terminal voltage [V]"]
                    output = V.entries

                except pybamm.SolverError as e:

                    with open(os.path.join(self.logs_dir_path, "errors"), "a") as log:
                        log.write("**************\n")
                        log.write(np.array2string(theta) + "\n")
                        log.write(repr(e) + "\n")

                    # array of zeros to maximize residual if solution did not converge
                    output = np.zeros(self.data.shape)

        if self.error_axis == "x":
            output = self._interpolate_time_over_y_values(times=self.times, y_values=output)

        if self.generated_data:
            try:
                ess = np.sum(np.square((output - self.data) / self.noise)) / len(output)

            except ValueError:
                # arrays of unequal size due to incomplete solution
                ess = np.sum(np.square(self.data / self.noise)) / len(self.data)
                output = np.zeros(self.data.shape)

            self.residuals.append(ess)
        self.generated_data = True

        return output

    def run(
            self,
            burnin: int = 0,
            n_iteration: int = 2000,
            n_chains: int = 12,
            n_workers: int = 4,
            sampling_method: str = "MetropolisRandomWalkMCMC",
    ):
        """
        Parameters
        __________
        burnin: int
            Initial iterations discarded from each chain.
        n_iteration: int
            Number of samples per chain.
        n_chains: int
            Number of chain.
        n_workers: int
            Number of parallel processes.
        sampling_method: str
            Name of MCMC sampling class (from pints)
            For a full list of samplers see:
            https://pints.readthedocs.io/en/stable/mcmc_samplers/index.html
            Defaults to MetropolisRandomWalkMCMC.

        Returns
        -------
        chains: np.ndarray
            Sampling chains (shape: iteration, chains, parameters).
        """
        sampling_method = "pints." + sampling_method

        problem = pints.SingleOutputProblem(
            self,
            self.times,
            self.data,
        )
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, self.noise)
        log_posterior = pints.LogPosterior(log_likelihood, self.log_prior)
        xs = [x * self.true_values for x in np.random.normal(1, 0.2, n_chains)]

        # Create MCMC routine
        mcmc = pints.MCMCController(
            log_posterior, n_chains, xs, method=eval(sampling_method)
        )

        # Add stopping criterion
        mcmc.set_max_iterations(n_iteration)

        # Logging
        mcmc.set_log_to_screen(True)
        mcmc.set_chain_filename(os.path.join(self.logs_dir_path, "chain.csv"))
        mcmc.set_chain_storage(store_in_memory=True)
        mcmc.set_log_pdf_filename(os.path.join(self.logs_dir_path, "log_pdf.csv"))

        # Parallelization
        # TODO: ForkingPickler(file, protocol).dump(obj)
        #  TypeError: cannot pickle 'SwigPyObject' object
        # mcmc.set_parallel(parallel=n_workers)

        # Run
        print("Running...")
        chains = mcmc.run()
        print("Done!")
        summary_stats = pints.MCMCSummary(
            chains=chains,
            time=mcmc.time(),
            parameter_names=[v.name for v in self.variables],
        )

        chains = pd.DataFrame(
            chains.reshape(chains.shape[0] * chains.shape[1], chains.shape[2])
        )

        self.chains = chains

        #  evaluate optimal value for each parameter
        theta_optimal = np.array(
            [float(chains[column].mode().iloc[0]) for column in chains.columns]
        )

        #  find residual at optimal value
        y_hat = self.simulate(theta_optimal, times=self.times)
        error_at_optimal = np.sum(abs(y_hat - self.data)) / len(self.data)

        # chi_sq = distance in residuals between optimal value and all others
        pd.DataFrame(
            {
                "residuals": self.residuals,
                "chi_sq": self.residuals - error_at_optimal,
            }
        ).to_csv(os.path.join(self.logs_dir_path, "residuals.csv"))

        with open(
                os.path.join(self.logs_dir_path, "metadata.json"),
                "r",
        ) as outfile:
            metadata = json.load(outfile)

        metadata.update(
            {
                "burnin": burnin,
                "n_iteration": n_iteration,
                "n_chains": n_chains,
                "sampling_method": sampling_method,
                "theta_optimal": theta_optimal.tolist(),
                "error_at_optimal": error_at_optimal,
                "ess": summary_stats.ess().tolist(),
                "ess_per_second": summary_stats.ess_per_second().tolist(),
                "mean": summary_stats.mean().tolist(),
                "std": summary_stats.std().tolist(),
                "rhat": summary_stats.rhat().tolist(),
                "time": summary_stats.time(),
            }
        )

        with open(
                os.path.join(self.logs_dir_path, "metadata.json"),
                "w",
        ) as outfile:
            json.dump(metadata, outfile)

        return chains

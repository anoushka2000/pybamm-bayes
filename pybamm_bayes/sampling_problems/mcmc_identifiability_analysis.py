import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import pints
import pybamm

from battery_model_parameterization.Python.sampling_problems.base_sampling_problem import (  # noqa: E501
    BaseSamplingProblem,
)
from battery_model_parameterization.Python.sampling_problems.utils import (
    _fmt_variables,
    _fmt_parameters,
    interpolate_time_over_y_values
)
from battery_model_parameterization.Python.logging import logger
from battery_model_parameterization.Python.variable import Variable


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
    output: str
        Name of battery simulation output time series to use for identifiability
        e.g "Terminal voltage [V]", "Terminal power [W]" or "Current [A]".
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
            output: str,
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
            output=output,
            transform_type=transform_type,
            project_tag=project_tag,
        )

        self.method = "MCMC"
        self.generated_data = False
        self.true_values = np.array([v.value for v in self.variables])
        self.noise = noise
        self.error_axis = error_axis
        self.t_eval = times

        # automatically updated the first time `simulate` method is called
        # (if reference axis is not time)
        self.data_reference_axis_values = times
        self.data_output_axis_values = self.generate_synthetic_data()

        with open(os.path.join(self.logs_dir_path, "metadata.json"), "w") as outfile:
            outfile.write(json.dumps(self.metadata))

        self.battery_simulation.save(
            os.path.join(self.logs_dir_path, "battery_simulation")
        )

    def generate_synthetic_data(self):

        inputs = dict(
            zip([v.name for v in self.variables],
                [v.value for v in self.variables])
        )

        if self.battery_simulation.operating_mode == "without experiment":
            if self.t_eval is None:
                raise ValueError(
                    """If battery simulation is not operated using an experiment,\n
                an array of times to evaluate simulation at must be passed."""
                )
        else:
            # solve simulation initialized with experiment
            self.battery_simulation.solve(inputs=inputs)
            # set `t_eval` attribute if argument not passed
            self.t_eval = self.battery_simulation.solution["Time [s]"].entries

        data = self.simulate(theta=self.true_values, times=self.t_eval)

        if self.error_axis == "y":
            reference_axis_values = self.t_eval
            output_values = data
        else:
            dx = np.diff(data)
            if not (np.all(dx >= 0) or np.all(dx <= 0)):
                raise NotImplementedError(
                    "Only `error_axis-y` supported for non-monotonic profiles."
                )

            # 'reference axis' is `y` (e.g Voltage): interpolating over this axis
            # `output_values` are time: error calculation is done w.r.t these
            reference_axis_values, output_values = interpolate_time_over_y_values(
                times=self.t_eval,
                y_values=data,
            )
        output_values = output_values + np.random.normal(0, self.noise, data.shape)
        return output_values

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
            "output": self.output,
            "error axis": self.error_axis,
            "t_eval": str(self.t_eval),
            "data_reference_axis_values": str(self.data_reference_axis_values),
            "data_output": str(self.data_output_axis_values),
        }

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
            Output time series.
        """
        variable_names = [v.name for v in self.variables]

        inputs = dict(zip(variable_names, [self.inverse_transform(t) for t in theta]))

        try:
            # solve with CasadiSolver
            self.battery_simulation.solve(
                inputs=inputs, solver=pybamm.CasadiSolver("fast"), t_eval=self.t_eval
            )
            solution = self.battery_simulation.solution
            output = solution[self.output].entries

            self.csv_logger.info(["Casadi fast", solution.solve_time.value])

        except pybamm.SolverError:
            # CasadiSolver "fast" failed
            try:
                self.battery_simulation.solve(
                    inputs=inputs,
                    solver=pybamm.CasadiSolver("safe"),
                    t_eval=self.t_eval
                )
                solution = self.battery_simulation.solution
                output = solution[self.output].entries

                self.csv_logger.info(["Casadi safe", solution.solve_time.value])

            except pybamm.SolverError:
                #  ScipySolver solver failed
                try:
                    self.battery_simulation.solve(
                        inputs=inputs,
                        solver=pybamm.ScipySolver(),
                        t_eval=self.t_eval
                    )
                    solution = self.battery_simulation.solution
                    output = solution[self.output].entries

                    self.csv_logger.info(["Scipy", solution.solve_time.value])

                except pybamm.SolverError as e:

                    with open(os.path.join(self.logs_dir_path, "errors"), "a") as log:
                        log.write("**************\n")
                        log.write(np.array2string(theta) + "\n")
                        log.write(repr(e) + "\n")

                    # array of zeros to maximize residual if solution did not converge
                    output = np.zeros(self.data_output_axis_values.shape)

        if self.error_axis == "x":
            if self.generated_data:
                _, output = interpolate_time_over_y_values(
                    times=self.t_eval,
                    y_values=output,
                    new_y=self.data_reference_axis_values
                )
            else:
                reference_values, output = interpolate_time_over_y_values(
                    times=self.t_eval,
                    y_values=output,
                )
                self.data_reference_axis_values = reference_values

        elif not self.generated_data:
            self.data_reference_axis_values = solution["Time [s]"].entries

            if self.t_eval is None:
                self.t_eval = solution["Time [s]"].entries

        if self.generated_data:
            try:
                ess = np.sum(
                    np.square((output - self.data_output_axis_values) / self.noise)
                ) / len(output)

            except ValueError:
                # arrays of unequal size due to incomplete solution
                ess = np.sum(
                    np.square(self.data_output_axis_values / self.noise)
                ) / len(self.data_output_axis_values)
                output = np.zeros(self.data_output_axis_values.shape)

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
            self.data_reference_axis_values,
            self.data_output_axis_values,
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
        logger.info("Running...")
        chains = mcmc.run()
        logger.info("Done!")
        summary_stats = pints.MCMCSummary(
            chains=chains,
            time=mcmc.time(),
            parameter_names=[v.name for v in self.variables],
        )
        self.csv_logger.info(["pints", mcmc.time()])

        chains = pd.DataFrame(
            chains.reshape(chains.shape[0] * chains.shape[1], chains.shape[2])
        )

        self.chains = chains

        #  evaluate optimal value for each parameter
        theta_optimal = np.array(
            [float(chains[column].mode().iloc[0]) for column in chains.columns]
        )

        #  find residual at optimal value
        y_hat = self.simulate(theta_optimal, times=self.t_eval)
        error_at_optimal = np.sum(
            abs(y_hat - self.data_output_axis_values)
        ) / len(y_hat)

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

import json
import os
from typing import List

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
    _fmt_parameters,
    _fmt_variables,
)
from battery_model_parameterization.Python.logging import logger


# TODO: add error axis argument
class ParameterEstimation(BaseSamplingProblem):
    """
    Class for conducting parameter estimation on a battery model
    using Markov Chain Monte Carlo (MCMC) methods.

    Parameters
    ----------
    data: pd.DataFrame
        Experimental data profile as a DataFrame with columns
        `time` (time in seconds) and `output`.
        (Ensure units of output data match dimensions of battery simulation
        output variable).
    battery_simulation: pybamm.Simulation
        Battery simulation for which parameter identifiability is being tested.
    parameter_values: pybamm.ParameterValues
        Parameter values for the simulation with `variables` as inputs.[
    variables: List[Variable]
        List of variables being identified in problem.
    output: str
        Name of battery simulation output corresponding to observed quantity
        recorded in data e.g "Terminal voltage [V]", "Terminal power [W]"
        or "Current [A]".
    transform_type: str
        Transformation variable value input to battery model
        and sampling space.
        (only `log10` implemented for now)
     initial_soc: float
        SOC to initialize battery simulation at.
    project_tag: str
        Project identifier (prefix to logs dir name).
    """

    def __init__(
            self,
            data: pd.DataFrame,
            battery_simulation: pybamm.Simulation,
            parameter_values: pybamm.ParameterValues,
            variables: List[Variable],
            output: str,
            transform_type: str,
            initial_soc: float = 1,
            project_tag: str = "",
    ):

        super().__init__(
            battery_simulation=battery_simulation,
            parameter_values=parameter_values,
            variables=variables,
            output=output,
            transform_type=transform_type,
            project_tag=project_tag
        )

        initial_values = [v.value for v in self.variables]
        # use 1e-6 as initial value for noise
        initial_values.append(1e-6)
        self.initial_values = np.array(initial_values)
        self.times = data["time"]
        self.data = data["output"]

        self.battery_simulation.solve(
            inputs=self.default_inputs,
            solver=pybamm.CasadiSolver("safe"),
            t_eval=self.times,
            initial_soc=initial_soc
        )

        simulation_end_time = max(self.battery_simulation.solution["Time [s]"].entries)
        if simulation_end_time > max(data["time"]):
            raise ValueError(
                f"""
            Experiment time span is {max(data["time"])} s.\n
            Simulation time span is {simulation_end_time} s.\n
            Time span and operating conditons of experimental data\n
            and simulation must match.
            """
            )

        if not np.array_equal(
                self.battery_simulation.solution["Time [s]"].entries, self.times
        ):
            # if simulation did not solve at times in data
            # (e.g. for experiments)
            # interpolate data to times in simulation

            interpolant = interp1d(x=self.times, y=self.data, fill_value="extrapolate")
            self.times = battery_simulation.solution["Time [s]"].entries
            self.data = interpolant(battery_simulation.solution["Time [s]"].entries)

        with open(os.path.join(self.logs_dir_path, "metadata.json"), "w") as outfile:
            outfile.write(json.dumps(self.metadata))

    @property
    def metadata(self):
        return {
            "battery model": self.battery_simulation.model.name,
            "parameter values": _fmt_parameters(self.parameter_values),
            "default inputs": self.default_inputs,
            "variables": _fmt_variables(self.variables),
            "transform type": self.transform_type,
            "project": self.project_tag,
            "times": str(self.times),
            "data": str(self.data),
        }

    @property
    def log_prior(self):
        # extra prior for unknown noise
        priors = [v.prior for v in self.variables]
        priors.append(pints.GaussianLogPrior(0, 1))
        return pints.ComposedLogPrior(*priors)

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
            Simulated time series.
        """
        variable_names = [v.name for v in self.variables]
        theta = theta[: len(variable_names) + 1]
        inputs = dict(zip(variable_names, [self.inverse_transform(t) for t in theta]))

        try:
            # solve with CasadiSolver
            self.battery_simulation.solve(
                inputs=inputs,
                solver=pybamm.CasadiSolver("fast"),
                t_eval=self.times,
                initial_soc=initial_soc
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
                    t_eval=self.times,
                    initial_soc=initial_soc
                )
                solution = self.battery_simulation.solution
                output = solution[self.output].entries

                self.csv_logger.info(["Casadi safe", solution.solve_time.value])

            except pybamm.SolverError as e:

                with open(os.path.join(self.logs_dir_path, "errors"), "a") as log:
                    log.write("**************\n")
                    log.write(np.array2string(theta) + "\n")
                    log.write(repr(e) + "\n")

                # array of zeros to maximize residual if solution did not converge
                output = np.zeros(self.data.shape)

        try:
            ess = np.sum(np.square((output - self.data))) / len(output)

        except ValueError:
            # arrays of unequal size due to incomplete solution
            ess = np.sum(np.square(self.data)) / len(self.data)
            output = np.zeros(self.data.shape)

        self.residuals.append(ess)
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
            model=self,
            times=self.times,
            values=self.data,
        )

        log_likelihood = pints.GaussianLogLikelihood(problem=problem)

        log_posterior = pints.LogPosterior(log_likelihood, self.log_prior)

        xs = [x * self.initial_values for x in np.random.normal(1, 0.2, n_chains)]

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
        self.chains = chains
        logger.info("Done!")

        self.csv_logger.info(["pints", mcmc.time() * 1000])
        chains = pd.DataFrame(
            chains.reshape(chains.shape[0] * chains.shape[1], chains.shape[2])
        )
        # drop noise estimation column
        chains = chains[chains.columns[:-1]]

        #  evaluate optimal value for each parameter
        theta_optimal = np.array(
            [float(chains[column].mode().iloc[0]) for column in chains.columns]
        )

        #  find residual at optimal value
        y_hat = self.simulate(theta_optimal, times=self.times)
        error_at_optimal = np.sum(abs(y_hat - self.data)) / len(self.data)

        pd.DataFrame(
            {
                "residuals": self.residuals,
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
            }
        )

        with open(
                os.path.join(self.logs_dir_path, "metadata.json"),
                "w",
        ) as outfile:
            json.dump(metadata, outfile)

        return chains

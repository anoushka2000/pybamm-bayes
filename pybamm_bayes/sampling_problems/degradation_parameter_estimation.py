import json
import os
from typing import List

import numpy as np
import pandas as pd
import pints
import pybamm
from scipy.interpolate import interp1d

from pybamm_bayes.logging import logger
from pybamm_bayes.sampling_problems.base_sampling_problem import (
    BaseSamplingProblem,
)  # noqa: E501
from pybamm_bayes.sampling_problems.utils import _fmt_parameters, _fmt_variables
from pybamm_bayes.variable import Variable


class DegradationParameterEstimation(BaseSamplingProblem):
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

        initial_values = [v.value for v in self.variables]
        # use 1e-6 as initial value for noise
        initial_values.append(1e-6)
        self.initial_values = np.array(initial_values)

        self.times = data["time"].values
        self.data_reference_axis_values = self.times

        self.data = data["output"].values
        self.data_output_axis_values = self.data 

        self.symbolic_params = pybamm.LithiumIonParameters()
        self.soh_parameter_values = self.parameter_values.copy()
        self.soh_parameter_values["Current function [A]"] = 3.35

        self.parameter_values["Initial concentration in negative electrode [mol.m-3]"] = pybamm.InputParameter("c_init_n")
        self.parameter_values["Initial concentration in positive electrode [mol.m-3]"] = pybamm.InputParameter("c_init_p")

        with open(os.path.join(self.logs_dir_path, "metadata.json"), "w") as outfile:
            outfile.write(json.dumps(self.metadata))

    @property
    def metadata(self):
        return {
            "battery model": self.battery_simulation.model.name,
            "parameter values": _fmt_parameters(self.parameter_values),
            "variables": _fmt_variables(self.variables),
            "transform type": self.transform_type,
            "project": self.project_tag,
            "output": self.output,
            "times": np.array2string(self.times),
            "data": np.array2string(self.data),
        }

    @property
    def log_prior(self):
        priors = [v.prior for v in self.variables]
        # extra prior for unknown noise
        priors.append(pints.GaussianLogPrior(0, 1))
        return pints.ComposedLogPrior(*priors)
    
    def _get_state_of_health(self, inputs):

        eps_p = inputs["eps_p"]
        eps_n = inputs["eps_n"]
        Q_Li = inputs.pop("Q_Li")
        
        self.soh_parameter_values["Negative electrode active material volume fraction"] = eps_n
        self.soh_parameter_values["Positive electrode active material volume fraction"] = eps_p        

        Q_n = self.soh_parameter_values.evaluate(self.symbolic_params.n.Q_init)
        Q_p = self.soh_parameter_values.evaluate(self.symbolic_params.p.Q_init)
        U_n = self.symbolic_params.n.prim.U
        U_p = self.symbolic_params.p.prim.U
        T_ref = self.symbolic_params.T_ref
        
        # Get voltage thresholds from data
        Vmin = self.data.min()
        Vmax = self.data.max()
        Vinit = self.data[0]
        print(f"eps_p {eps_p} eps_n {eps_n} Q_Li {Q_Li} Vinit {Vinit} Vmin {Vmin} Vmax {Vmax}")

        # Define Electrode Capacity Model
        model = pybamm.BaseModel()

        # equation for 100% SOC
        x_100 = pybamm.Variable("x_100")
        y_100 = (Q_Li - x_100 * Q_n) / Q_p

        # equation for 0% SOC
        x_0 = pybamm.Variable("x_0")
        Q = Q_n * (x_100 - x_0)
        y_0 = y_100 + Q / Q_p

        # equation for SOC
        soc = pybamm.Variable("SOC")
        x = x_0 + soc * (x_100 - x_0)
        y = y_0 - soc * (y_0 - y_100)

        model.algebraic = {
            x_100: U_p(y_100, T_ref) - U_n(x_100, T_ref) - Vmax,
            x_0: U_p(y_0, T_ref) - U_n(x_0, T_ref) - Vmin,
            soc: U_p(y, T_ref) - U_n(x, T_ref) - Vinit
            }

        model.initial_conditions = {x_100: 0.995, x_0: 0.02, soc: 0.9}

        model.variables = {
            "x": x, 
            "y": y,
            "SOC": soc
            }

        soh_solver = pybamm.Simulation(
            model, 
            parameter_values=self.soh_parameter_values 
        )

        sol = soh_solver.solve([0])

        x = sol["x"].data[0]
        y = sol["y"].data[0]
        soc = sol["SOC"].data[0]

        return soc, x, y



        
    def simulate(self, theta: np.ndarray, times: np.ndarray):
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
        initial_soc, x, y = self._get_state_of_health(inputs=inputs)
        # print(f"initial_soc {initial_soc}, x {x}, y {y}")

        c_p_max = self.parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
        c_n_max = self.parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
        inputs["c_init_p"] = c_p_max * y
        inputs["c_init_n"] = c_n_max * x

        try:
            # solve with CasadiSolver
            self.battery_simulation.solve(
                inputs=inputs, solver=pybamm.CasadiSolver("safe"), t_eval=self.times,
                # initial_soc=initial_soc
            )
            solution = self.battery_simulation.solution
            output = solution[self.output].entries

            self.csv_logger.info(["Casadi safe", solution.solve_time.value])

        # except pybamm.SolverError:
        #     # CasadiSolver "fast" failed
        #     try:
        #         self.battery_simulation.solve(
        #             inputs=inputs, solver=pybamm.CasadiSolver("safe"), t_eval=self.times,
        #             # initial_soc=initial_soc
        #         )
        #         solution = self.battery_simulation.solution
        #         output = solution[self.output].entries

        #         self.csv_logger.info(["Casadi safe", solution.solve_time.value])

        except pybamm.SolverError:
            #  Casadi solver failed
            try:
                self.battery_simulation.solve(
                    inputs=inputs, solver=pybamm.ScipySolver(), t_eval=self.times,
                    # initial_soc=initial_soc
                )
                solution = self.battery_simulation.solution
                output = solution[self.output].entries

                self.csv_logger.info(["Scipy", solution.solve_time.value])

            except pybamm.SolverError as e:

                with open(os.path.join(self.logs_dir_path, "errors"), "a") as log:
                    log.write("**************\n")
                    log.write(np.array2string(theta) + "\n")
                    log.write(repr(e) + "\n")

                # array of zeros to as residual if solution did not converge
                output = np.zeros(*self.data.shape)

        try:
            ess = np.sum(np.square((output - self.data))) / len(output)

        except ValueError:
            # arrays of unequal size due to incomplete solution
            ess = np.sum(np.square(self.data)) / len(self.data)
            output = np.zeros(*self.data.shape)
        
        if output.max() > 100: 
            output = np.random.rand(*self.data.shape)
        self.residuals.append(ess)

        print(f"output {output}")
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

        xs = [x * self.initial_values for x in np.random.normal(1, 0.01, n_chains)]

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

        self.csv_logger.info(["pints", mcmc.time() * 1000])
        chains = pd.DataFrame(
            chains.reshape(chains.shape[0] * chains.shape[1], chains.shape[2])
        )
        # drop noise estimation column
        chains = chains[chains.columns[:-1]]
        self.chains = chains

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

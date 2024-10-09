import json
import os
from typing import List

import numpy as np
import pandas as pd
import pints
import pybamm

from pybamm_bayes.logging import logger
from pybamm_bayes.sampling_problems.base_sampling_problem import (
    BaseSamplingProblem,
)  # noqa: E501
from pybamm_bayes.sampling_problems.utils import _fmt_parameters, _fmt_variables
from pybamm_bayes.variable import Variable

import signal




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
        project_tag: str = "",
    ):

        super().__init__(
            battery_simulation=battery_simulation,
            parameter_values=parameter_values,
            variables=variables,
            output=output,
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
            "project": self.project_tag,
            "output": self.output,
            "times": np.array2string(self.times),
            "data": np.array2string(self.data),
        }

    @property
    def log_prior(self):
        priors = [v.prior for v in self.variables]
        # extra prior for unknown noise
        priors.append(pints.UniformLogPrior(0, 1))
        return pints.ComposedLogPrior(*priors)
    
    def _get_state_of_health(self, inputs):

        if "eps_n" in inputs.keys():
            eps_n = inputs.get("eps_n")
            self.soh_parameter_values["Negative electrode active material volume fraction"] = eps_n
            self.parameter_values["Negative electrode active material volume fraction"] = eps_n

        if "eps_p" in inputs.keys():
            eps_p = inputs.get("eps_p")
            self.soh_parameter_values["Positive electrode active material volume fraction"] = eps_p      
            self.parameter_values["Positive electrode active material volume fraction"] = eps_p      

        if "Q_Li" in inputs.keys():
            Q_Li = inputs.pop("Q_Li")
        else: 
            Q_Li = self.soh_parameter_values.evaluate(self.symbolic_params.Q_Li_particles_init)
        
        Q_n = self.soh_parameter_values.evaluate(self.symbolic_params.n.Q_init)
        Q_p = self.soh_parameter_values.evaluate(self.symbolic_params.p.Q_init)
        U_n = self.symbolic_params.n.prim.U
        U_p = self.symbolic_params.p.prim.U
        T_ref = self.symbolic_params.T_ref
        
        # Get voltage thresholds from data
        Vmin = self.parameter_values["V_min"]
        Vmax = self.parameter_values["V_max"]
        Vinit = self.data[0]
        print(f"Q_Li {Q_Li} Vinit {Vinit} Vmin {Vmin} Vmax {Vmax}")
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

        self.soh_parameter_values.process_model(model)
        try:
            solver = pybamm.CasadiAlgebraicSolver()
            sol = solver.solve(model)

            x = sol["x"].data[0]
            y = sol["y"].data[0]
            soc = sol["SOC"].data[0]
            return soc, x, y
        
        except pybamm.SolverError as e:
            try:
                model.initial_conditions = {
                    x_100: 0.9, 
                    x_0: 0.1, 
                    soc: (Vinit - Vmin) / (Vmax - Vmin)
                }
                self.soh_parameter_values.process_model(model)
                solver = pybamm.CasadiAlgebraicSolver()
                sol = solver.solve(model)
                x = sol["x"].data[0]
                y = sol["y"].data[0]
                soc = sol["SOC"].data[0]
            except pybamm.SolverError as e:
                # fails to solve for eSOH
                return None, None, None


    def simulate(self, theta: np.ndarray, times: np.ndarray):
        """
        Wrapper around _simulate method to handle timeouts.
        """
        def handler(signum, frame):
            raise TimeoutError
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60)
        try:
            output = self._simulate(theta, times)
        except TimeoutError as e:
            with open(os.path.join(self.logs_dir_path, "errors"), "a") as log:
                log.write("**************\n")
                log.write(np.array2string(theta) + "\n")
                log.write(repr(e) + "\n")
            output = np.zeros(*self.data.shape)
        signal.alarm(0)
        return output


        
    def _simulate(self, theta: np.ndarray, times: np.ndarray):
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
        inverse_transforms = [v.inverse_transform for v in self.variables]
        theta = theta[: len(variable_names) + 1]
        inputs = dict(
            zip(variable_names, [inverse_transforms[i](theta[i]) for i in range(len(theta))])
            )
        initial_soc, x, y = self._get_state_of_health(inputs=inputs)
        if not (initial_soc and x and y):
            output = np.zeros(*self.data.shape)
            return output
        c_p_max = self.parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
        c_n_max = self.parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
        inputs["c_init_p"] = c_p_max * y
        inputs["c_init_n"] = c_n_max * x
        
        try:
            # solve with CasadiSolver
            self.battery_simulation.solve(
                inputs=inputs, 
                solver=pybamm.CasadiSolver("safe"), 
                t_eval=self.times,
                # initial_soc=initial_soc
            )
            solution = self.battery_simulation.solution
            output = solution[self.output].entries

            self.csv_logger.info(["Casadi safe", solution.solve_time.value])
        
        except pybamm.SolverError as e:

            with open(os.path.join(self.logs_dir_path, "errors"), "a") as log:
                log.write("**************\n")
                log.write(np.array2string(theta) + "\n")
                log.write(repr(e) + "\n")

            # array of zeros to as residual if solution did not converge
            output = np.zeros(*self.data.shape)

        output = np.nan_to_num(output, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        if len(output) < len(self.data):
            fill_length = len(self.data) - len(output)
            # arrays of unequal size due to incomplete solution
            output = np.append(output, np.zeros(fill_length))
      
        ess = np.sum(np.square((output - self.data))) / len(output)
        print(ess)
        if output.max() > 100: 
            output = np.random.rand(*self.data.shape)
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

        xs = [x*self.initial_values for x in np.random.normal(1, 0.1, n_chains)]

        # Create MCMC routine
        mcmc = pints.MCMCController(
            log_posterior, n_chains, xs, method=eval(sampling_method),
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
        error_at_optimal = np.sum((y_hat - self.data)**2) / len(self.data)

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

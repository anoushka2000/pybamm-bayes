import os
from typing import List
import json
import numpy as np
import pandas as pd
import pybamm
import pints
from battery_model_parameterization.Python.sampling_problems.base_sampling_problem import (  # noqa: E501
    BaseSamplingProblem,
)
from battery_model_parameterization.Python.variable import Variable
from scipy.interpolate import interp1d


def _fmt_variables(variables):
    lst = []
    for v in variables:
        var = v.__dict__.copy()
        var["prior"] = var["prior"].__dict__
        lst.append(var)
    return lst


def _fmt_parameters(parameters):
    return {k: str(v) for k, v in parameters.items()}


class ParameterEstimation(BaseSamplingProblem):
    """
    Defines parameter estimations problem for a battery model.

    Parameters
    ----------
    data: pd.DataFrame
        Experimental voltage profile as a data frame with columns
        `time` (time in seconds) and `voltage` (voltage in volts).
    battery_simulation: pybamm.Simulation
        Battery simulation for which parameter identifiability is being tested.
    parameter_values: pybamm.ParameterValues
        Parameter values for the simulation with `variables` as inputs.[
    variables: List[Variable]
        List of variables being identified in problem.
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
            transform_type: str,
            project_tag: str = "",
    ):

        super().__init__(
            battery_simulation, parameter_values, variables, transform_type, project_tag
        )

        initial_values = [v.value for v in self.variables]
        # use 1e-6 as initial value for noise
        initial_values.append(1e-6)
        self.initial_values = np.array(initial_values)
        self.times = data["time"]
        self.data = data["voltage"]

        self.battery_simulation.solve(
            inputs=self.default_inputs,
            solver=pybamm.CasadiSolver("safe"),
            t_eval=self.times,
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
            Voltage time series.
        """
        variable_names = [v.name for v in self.variables]
        theta = theta[: len(variable_names) + 1]
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
                #  Casadi solver failed
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

        try:
            ess = np.sum(np.square((output - self.data))) / len(output)

        except ValueError:
            # arrays of unequal size due to incomplete solution
            ess = np.sum(np.square(self.data)) / len(self.data)
            output = np.zeros(self.data.shape)

        self.residuals.append(ess)
        return output

import os
from typing import List

import numpy as np
import pandas as pd
import pybamm
from battery_model_parameterization import BaseSamplingProblem, Variable


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
        `time` and `voltage`.
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

        self.initial_values = np.array([v.value for v in self.variables])

        self.battery_simulation.solve(
            inputs=self.default_inputs, solver=pybamm.CasadiSolver("fast")
        )

        self.times = data["time"]
        self.data = data["voltage"]

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
        inputs = self.default_inputs
        assert set(variable_names) - set(inputs.keys()) == set()

        inputs.update(
            dict(zip(variable_names, [self.inverse_transform(t) for t in theta]))
        )
        try:
            # solve with CasadiSolver
            self.battery_simulation.solve(
                inputs=inputs, solver=pybamm.CasadiSolver("fast")
            )
            solution = self.battery_simulation.solution
            V = solution["Terminal voltage [V]"]
            output = V.entries

        except pybamm.SolverError:
            # CasadiSolver "fast" failed
            try:
                self.battery_simulation.solve(
                    inputs=inputs, solver=pybamm.CasadiSolver("safe")
                )
                solution = self.battery_simulation.solution
                V = solution["Terminal voltage [V]"]
                output = V.entries

            except pybamm.SolverError:
                #  ScipySolver solver failed
                try:
                    self.battery_simulation.solve(
                        inputs=inputs, solver=pybamm.ScipySolver()
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

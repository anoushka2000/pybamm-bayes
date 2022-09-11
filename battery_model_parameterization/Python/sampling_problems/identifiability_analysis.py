import os
from typing import List

import numpy as np
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


class IdentifiabilityAnalysis(BaseSamplingProblem):
    """
    Class for conducting non-linear identifiability analysis on battery simulation parameters.

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
        times: np.ndarray,
        project_tag: str = "",
    ):

        super().__init__(
            battery_simulation,
            parameter_values,
            variables,
            transform_type,
            project_tag,
        )

        self.generated_data = False
        self.noise = noise
        self.true_values = np.array([v.value for v in self.variables])
        self.residuals = []

        if battery_simulation.operating_mode == "without experiment":
            self.times = times
        else:
            battery_simulation.solve(self.default_inputs)
            self.times = battery_simulation.solution["Time [s]"].entries

        data = self.simulate(self.true_values, self.times)
        self.data = data + np.random.normal(0, self.noise, data.shape)

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
            "times": self.times,
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

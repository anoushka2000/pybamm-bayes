import json
import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pints
import pints.plot
import pybamm
from battery_model_parameterization import BaseSamplingProblem


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
    battery_simulation: pybamm.Simulation
        Battery simulation for which parameter identifiability is being tested.
    parameter_values: pybamm.ParameterValues
        Parameter values for the simulation with `variables` as inputs.
    default_inputs: Optional[Dict[str, float]]
        Possible dictionary keys are: [
    variables: List[Variable]
        List of variables being identified in problem.
    transform_type: str
        Transformation variable value input to battery model
        and sampling space.
        (only `log10` implemented for now)
    resolution: int
        Resolution of data to used for parameter identification
        (number and time unit e.g `1 minute`.)
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
            resolution: int,
            noise: float,
            project_tag: str = "",
    ):

        super().__init__()
        self.generated_data = False
        self.battery_simulation = battery_simulation
        self.noise = noise
        self.parameter_values = parameter_values
        self.default_inputs = default_inputs
        self.variables = variables
        self.true_values = np.array([v.true_value for v in self.variables])
        self.transform_type = transform_type
        self.inverse_transform = INVERSE_TRANSFORMS[self.transform_type]
        self.resolution = resolution
        self.project_tag = project_tag
        self.logs_dir_path = self.create_logs_dir()
        self.residuals = []

        self.battery_simulation.solve(
            inputs=self.default_inputs, solver=pybamm.CasadiSolver("fast")
        )
        self.times = self.battery_simulation.solution["Time [s]"].entries

        data = self.simulate(
            self.true_values,
            times=[
                0,
            ],
        )
        self.data = data + np.random.normal(0, self.noise, data.shape)

        if not os.path.isdir(self.logs_dir_path):
            os.makedirs(self.logs_dir_path)

        with open(os.path.join(self.logs_dir_path, "metadata.json"), "w") as outfile:
            outfile.write(json.dumps(self.metadata))

    @property
    def log_prior(self):
        return pints.ComposedLogPrior(*[v.prior for v in self.variables])

    def create_logs_dir(self):
        logs_dir_name = [self.project_tag] + [p.name for p in self.variables]
        logs_dir_name.append(datetime.utcnow().strftime(format="%d%m%y_%H%M"))
        current_path = os.getcwd()
        return os.path.join(current_path, "logs", "__".join(logs_dir_name))

    @property
    def battery_simulation(self):
        try:
            return self._battery_simulation
        except AttributeError:
            self.setup_battery_simulation()
            return self._battery_simulation

    def setup_battery_simulation(self):
        self.experiment = pybamm.Experiment(
            operating_conditions=self.operating_conditions, period=self.resolution
        )
        self._battery_simulation = pybamm.Simulation(
            self.battery_model,
            experiment=self.experiment,
            parameter_values=self.parameter_values,
        )

    @property
    def metadata(self):
        return {
            "battery model": self.battery_model.name,
            "operating conditions": self.operating_conditions,
            "parameter values": _fmt_parameters(self.parameter_values),
            "default inputs": self.default_inputs,
            "variables": _fmt_variables(self.variables),
            "transform type": self.transform_type,
            "noise": self.noise,
            "project": self.project_tag,
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

    def n_parameters(self):
        """
        Return dimension of the variable vector.
        """
        return len(self.variables)

    def plot_priors(self):
        """
        Plot priors for all variables and save.
        """
        i = 0
        fig, axs = plt.subplots(len(self.variables))
        fig.subplots_adjust(hspace=0.9)
        fig.suptitle("Prior Distributions")
        for variable in self.variables:
            n, bins, patches = axs[i].hist(
                variable.prior.sample(7000), bins=80, alpha=0.6
            )
            axs[i].plot(
                [
                    variable.true_value,
                    variable.true_value,
                ],
                [0, max(n)],
            )
            axs[i].set(xlabel=f"{variable.name} (transformed)", ylabel="Frequency")
            i += 1
        plt.savefig(os.path.join(self.logs_dir_path, "prior"))

    def plot_data(self):
        """
        Plot of voltage profile used for fitting and save.
        """
        plt.plot(self.battery_simulation.solution["Time [s]"].entries, self.data)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.savefig(os.path.join(self.logs_dir_path, "data"))

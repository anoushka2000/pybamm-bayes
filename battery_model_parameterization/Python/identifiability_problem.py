import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.plot
import pybamm
import battery_model_parameterization.Python.battery_models.model_setup as models


def _inverse_log10(x):
    return 10 ** x


INVERSE_TRANSFORMS = {"log10": _inverse_log10}
MODEL_LOOKUP = {
    "default_dfn": models.default_dfn(),
    "default_spme": models.default_spme()
}


def _fmt_variables(variables):
    lst = []
    for v in variables:
        var = v.__dict__.copy()
        var["prior"] = var["prior"].__dict__
        lst.append(var)
    return lst


def _fmt_parameters(parameters):
    return {k: str(v) for k, v in parameters.items()}


class IdentifiabilityProblem(pints.ForwardModel):
    """
    Defines parameter identifiablility problem for a battery model.

    Parameters
    ----------
    battery_model: str
        Name of battery model to be parameterised
        One of "default_spme" or "default_dfn".
    operating_conditions: List[Tuple[str]]/ List[str]
        List of operating conditions (passed to pybamm.Experiment).
    variables: List[Variable]
        List of variables being identified in problem.
    transform_type: str
        Transformation variable value input to battery model
        and sampling space.
        (only `log10` implemented for now)
    resolution: str
        Resolution of data used for parameter identification
        (number and time unit e.g `1 minute`.)
    noise: float
        Noise added to data used for identification.
    project_tag: str
        Project identifier (prefix to logs dir name).
    """

    def __init__(
            self,
            battery_model,
            operating_conditions,
            variables,
            transform_type,
            resolution,
            noise,
            project_tag=" ",
    ):
        super().__init__()
        self.generated_data = False
        self.battery_model = MODEL_LOOKUP[battery_model][0]
        self.operating_conditions = operating_conditions
        self.parameter_values = MODEL_LOOKUP[battery_model][1]
        self.default_inputs = {
            "Ds_n": 1e-3,
            "Ds_p": 1e-3,
            "De": 1e-3,
            "j0_n": 1e-3,
            "j0_p": 1e-3,
        }
        self.variables = variables
        self.true_values = np.array([v.true_value for v in self.variables])

        self.transform_type = transform_type
        self.inverse_transform = INVERSE_TRANSFORMS[self.transform_type]
        self.resolution = resolution
        self.noise = noise
        self.project_tag = project_tag
        self.logs_dir_path = self.create_logs_dir()
        self.residuals = []

        self.battery_simulation.solve(inputs=self.default_inputs,
                                      solver=pybamm.CasadiSolver("fast"))
        self.times = self.battery_simulation.solution["Time [s]"].entries

        data = self.simulate(self.true_values, times=[0, ])
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
        self.experiment = pybamm.Experiment(operating_conditions=self.operating_conditions,
                                            period=self.resolution)
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

        except Exception:
            # CasadiSolver "fast" failed
            try:
                self.battery_simulation.solve(
                    inputs=inputs, solver=pybamm.CasadiSolver("safe")
                )
                solution = self.battery_simulation.solution
                V = solution["Terminal voltage [V]"]
                output = V.entries

            except Exception:
                #  IDAKLUSolver "casadi" solver failed
                try:
                    self.battery_simulation.solve(
                        inputs=inputs, solver=pybamm.IDAKLUSolver()
                    )
                    solution = self.battery_simulation.solution
                    V = solution["Terminal voltage [V]"]
                    output = V.entries

                except Exception as e:

                    with open(os.path.join(self.logs_dir_path, "errors"), "a") as log:
                        log.write("**************\n")
                        log.write(np.array2string(theta) + "\n")
                        log.write(repr(e) + "\n")

                    # array of zeros to maximize residual if solution did not converge
                    output = np.zeros(self.battery_simulation.solution["Time [s]"].entries.shape)

        if self.generated_data:
            ess = np.sum(np.square((output - self.data) / self.noise)) / len(output)
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

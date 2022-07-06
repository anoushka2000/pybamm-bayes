import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.plot
import pybamm

INVERSE_TRANSFORMS = {"log10": lambda x: 10 ** x}


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
    battery_model: pybamm.BaseBatteryModel
        Battery model to be parameterised.
    variables: list[Variable]
        List of variables being identified in problem.
    transform_type: str
        Transformation variable value and sampling space.
    resolution:
        Time resolution of data used for parameter identification.
    timespan: int
        Time up to which data is simulated.
    noise: float
        Noise added to data used for identification.
    """

    def __init__(
            self,
            battery_model,
            variables,
            transform_type,
            parameter_values,
            resolution,
            timespan,
            noise,
    ):
        super().__init__()
        self.generated_data = False
        self.battery_model = battery_model
        self.parameter_values = parameter_values
        self.default_inputs = {
            "Ds_n": 1e-3,
            "Ds_p": 1e-3,
            "De": 1e-3,
            "j0_n": 1e-3,
            "j0_p": 1e-3,
        }
        self.variables = variables
        self.transform_type = transform_type
        self.inverse_transform = INVERSE_TRANSFORMS[self.transform_type]
        self.resolution = resolution
        self.timespan = timespan
        self.noise = noise
        self.times = np.linspace(0, timespan, num=(timespan // resolution))
        self.logs_dir_path = self.create_logs_dir()
        self.chi_sq = []

        data = self.simulate(self.true_values, self.times)
        self.data = data + np.random.normal(0, self.noise, data.shape)

        if not os.path.isdir(self.logs_dir_path):
            os.makedirs(self.logs_dir_path)

        with open(os.path.join(self.logs_dir_path, "metadata.json"), "w") as outfile:
            outfile.write(json.dumps(self.metadata))

    @property
    def log_prior(self):
        return pints.ComposedLogPrior(*[v.prior for v in self.variables])

    @property
    def true_values(self):
        return np.array([v.true_value for v in self.variables])

    def create_logs_dir(self):
        logs_dir_name = [p.name for p in self.variables]
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

    @property
    def metadata(self):
        return {
            "battery model": self.battery_model.name,
            "parameter values": _fmt_parameters(self.parameter_values),
            "default inputs": self.default_inputs,
            "variables": _fmt_variables(self.variables),
            "transform type": self.transform_type,
        }

    def setup_battery_simulation(self):
        self._battery_simulation = pybamm.Simulation(
            self.battery_model,
            parameter_values=self.parameter_values,
            solver=pybamm.CasadiSolver("fast"),
        )

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
            self.battery_simulation.solve(times, inputs=inputs)
            solution = self.battery_simulation.solution
            V = solution["Terminal voltage [V]"]
            output = V.entries.reshape(times.shape)

            if self.generated_data:
                ess = np.sum(np.square((output - self.data) / self.noise)) / len(output)
                self.chi_sq.append(ess)

        #             solution.save_data(os.path.join(self.logs_dir_path, f"sol_data_{self.iteration_counter}.csv"),
        #                                             ["Time [s]", "Terminal voltage [V]"], to_format="csv")
        #             self.iteration_counter += 1

        except Exception as e:
            print(e)
            # TODO: adjust parameters and retry solve or return last residual?
            # array of zeros to maximize resolution if solution did not converge
            output = np.zeros(times.shape)

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
        fig.suptitle('Prior Distributions')
        for variable in self.variables:
            n, bins, patches = axs[i].hist(variable.prior.sample(7000), bins=80, alpha=0.6)
            axs[i].plot(
                [
                    variable.true_value,
                    variable.true_value,
                ],
                [0, max(n)],
            )
            axs[i].set(xlabel=f"{variable.name} (transformed)", ylabel="Frequency")
            i += 1
        plt.savefig(os.path.join(self.logs_dir_path, f"prior"))

    def plot_data(self):
        """
        Plot of voltage profile used for fitting and save.
        """
        plt.plot(self.times, self.data)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.savefig(os.path.join(self.logs_dir_path, f"data"))

import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pints
import pints.plot
import pybamm
import seaborn as sns
import tqdm
from battery_model_parameterization.Python.variable import Variable
from scipy import stats


def _fmt_variables(variables):
    lst = []
    for v in variables:
        var = v.__dict__.copy()
        var["prior"] = var["prior"].__dict__
        lst.append(var)
    return lst


def _fmt_parameters(parameters):
    return {k: str(v) for k, v in parameters.items()}


class BaseSamplingProblem(pints.ForwardModelS1):
    """
    Base class for using MCMC sampling  of battery simulation parameters.

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
    project_tag: str
        Project identifier (prefix to logs dir name).
    """

    def __init__(
        self,
        battery_simulation: pybamm.Simulation,
        parameter_values: pybamm.ParameterValues,
        variables: List[Variable],
        transform_type: str,
        project_tag: str = "",
    ):

        super().__init__()
        self.method = None
        self.battery_simulation = battery_simulation
        self.parameter_values = parameter_values
        self.variables = variables
        self.transform_type = transform_type
        self.project_tag = project_tag
        self.logs_dir_path = self.create_logs_dir()
        self.default_inputs = {v.name: v.value for v in self.variables}
        self.residuals = []
        self.chains = pd.DataFrame()

        if not os.path.isdir(self.logs_dir_path):
            os.makedirs(self.logs_dir_path)

    @property
    def transforms(self):
        return {
            "log10": lambda x: 10 ** float(x),
            "None": lambda x: x,
            "negated_log10": lambda x: 10 ** float(-x),
        }

    @property
    def inverse_transform(self):
        return self.transforms[self.transform_type]

    @property
    def log_prior(self):
        return pints.ComposedLogPrior(*[v.prior for v in self.variables])

    def create_logs_dir(self):
        logs_dir_name = [self.project_tag] + [p.name for p in self.variables]
        logs_dir_name.append(datetime.utcnow().strftime(format="%d%m%y_%H%M"))
        current_path = os.getcwd()
        return os.path.join(current_path, "logs", "__".join(logs_dir_name))

    @property
    def metadata(self):
        return {
            "battery model": self.battery_simulation.model.name,
            "parameter values": _fmt_parameters(self.parameter_values),
            "default inputs": self.default_inputs,
            "variables": _fmt_variables(self.variables),
            "transform type": self.transform_type,
            "project": self.project_tag,
        }

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
            if self.method == "BOLFI":
                lower, rng = (
                    variable.bounds[0],
                    variable.bounds[1] - variable.bounds[0],
                )
                sample = (
                    lower
                    + variable.prior.distribution.rvs(
                        size=7000,
                    )
                    * rng
                )
            else:
                sample = variable.prior.sample(7000).flatten()

            sns.distplot(
                sample,
                hist=True,
                kde=False,
                bins=80,
                color="darkblue",
                ax=axs[i],
            )

            if variable.value:
                axs[i].axvline(
                    x=variable.value, ymax=0.7, color="darkblue", ls="--", lw=1
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

    def plot_results_summary(self):

        variable_names = [var.name for var in self.variables]

        # Fit a truncated normal distribution to chains
        loc = self.chains.mean()
        scale = self.chains.std()
        # Fit a truncated normal distribution to chains
        posterior_samples = stats.truncnorm(
            a=(self.chains.min() - loc) / scale,
            b=(self.chains.max() - loc) / scale,
            loc=loc,
            scale=scale,
        ).rvs(size=(7000, len(variable_names)))

        results = []
        summary = []

        if len(posterior_samples) > 1:
            for i, input_set in tqdm.tqdm(enumerate(posterior_samples)):

                inputs = dict(zip(variable_names, input_set))
                if self.method == "BOLFI":
                    solution_V = self.simulate(*list(input_set))
                else:
                    solution_V = self.simulate(theta=list(input_set), times=self.times)
                summary.append(
                    {
                        **inputs,
                        "Residual": abs(self.data - solution_V).sum() / len(solution_V),
                    }
                )

                for t, V in zip(self.times, solution_V):
                    results.append(
                        {
                            **inputs,
                            "Time [s]": t,
                            "Voltage [V]": V,
                            "run": i,
                        }
                    )
            df = pd.DataFrame(results)
            df_summary = pd.DataFrame(summary)
            # used to generate color bar for residual plot
            residual_plot = plt.scatter(
                df_summary[variable_names[0]],
                df_summary[variable_names[1]],
                c=df_summary.Residual,
                cmap=sns.cubehelix_palette(as_cmap=True),
            )
            plt.clf()

            voltage_scratch_plots = []

            for var in variable_names:
                plot = plt.scatter(
                    df["Time [s]"],
                    df["Voltage [V]"],
                    c=df[var],
                    cmap=sns.cubehelix_palette(as_cmap=True),
                )

                voltage_scratch_plots.append(plot)
                plt.clf()

            # Set up axis
            rows = len(variable_names) + 1
            fig, ax = plt.subplots(rows, 2, figsize=(8, rows * 3))
            plt.colorbar(residual_plot, ax=ax[0][1], label="Residual")

            # Add prior plots to column 0 (histograms)
            for i, variable in list(zip([i for i in range(1, rows)], self.variables)):
                if self.method == "BOLFI":
                    lower, rng = (
                        variable.bounds[0],
                        variable.bounds[1] - variable.bounds[0],
                    )
                    sample = (
                        lower
                        + variable.prior.distribution.rvs(
                            size=7000,
                        )
                        * rng
                    )
                else:
                    sample = variable.prior.sample(7000).flatten()

                sns.distplot(
                    sample,
                    hist=False,
                    kde=True,
                    bins=50,
                    color="darkblue",
                    kde_kws={"linewidth": 4},
                    ax=ax[i][0],
                )

                if variable.value:
                    ax[i][0].axvline(
                        x=variable.value, ymax=0.7, color="darkblue", ls="--", lw=1
                    )

            # Generate plots using df and summary df
            for i, var in list(zip([i for i in range(1, rows)], variable_names)):
                # column 0: plot a histogram for each variable
                sns.distplot(
                    df[var],
                    hist=True,
                    kde=False,
                    bins=50,
                    color="darkblue",
                    ax=ax[i][0],
                )

                # column 1: plot voltage colored by variable for each variable
                sns.lineplot(
                    data=df,
                    x="Time [s]",
                    y="Voltage [V]",
                    hue=df[var],
                    ax=ax[i][1],
                    palette=sns.cubehelix_palette(as_cmap=True),
                )
                # add color bar
                plt.colorbar(voltage_scratch_plots[i - 1], ax=ax[i][1], label=var)
                # remove discrete legend
                ax[i][1].legend_.remove()

            sns.scatterplot(
                data=df_summary,
                x=variable_names[0],
                y=variable_names[1],
                hue="Residual",
                ax=ax[0][1],
            )
            # remove discrete legend
            ax[0][1].legend_.remove()

            sns.lineplot(
                data=df, x="Time [s]", y="Voltage [V]", errorbar=("sd", 1), ax=ax[0][0]
            )
            ax[0][0].set_ylabel("Voltage with one standard deviation")
            ax[1][0].set_ylabel("Frequency")
            ax[2][0].set_ylabel("Frequency")

            fig.tight_layout()
        plt.savefig(os.path.join(self.logs_dir_path, "results_summary"))

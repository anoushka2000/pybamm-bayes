import json
import os
from typing import List, Optional

import elfi
import numpy as np
import pandas as pd
import pybamm

from battery_model_parameterization.Python.sampling_problems.base_sampling_problem import \
    BaseSamplingProblem  # noqa: E501
from battery_model_parameterization.Python.variable import Variable


def _fmt_variables(variables):
    lst = []
    for v in variables:
        var = v.__dict__.copy()
        var["prior"] = "ElfiPrior"
        lst.append(var)
    return lst


def _fmt_parameters(parameters):
    return {k: str(v) for k, v in parameters.items()}


class BOLFIIdentifiabilityAnalysis(BaseSamplingProblem):
    """
    Class for conducting non-linear identifiability analysis on
    battery simulation parameters using the Bayesian Optimization
    for Likelihood-Free Inference (BOLFI) framework.

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
        (only `log10`  and `None` implemented for now)
    noise: float
        Scale of zero-mean noise added to synthetic data used to identify parameters.
    target_resolution: int
        Frequency of points over which discrepancy is calculated when training
        Gaussian Process surrogate.
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
            target_resolution: int = 30,
            times: Optional[np.ndarray] = None,
            project_tag: str = "",
    ):

        super().__init__(
            battery_simulation=battery_simulation,
            parameter_values=parameter_values,
            variables=variables,
            transform_type=transform_type,
            project_tag=project_tag,
        )

        self.method = "BOLFI"
        self.generated_data = False
        self.true_values = np.array([v.value for v in self.variables])
        self.noise = noise
        self.target_resolution = target_resolution

        if battery_simulation.operating_mode == "without experiment":
            if times is None:
                raise ValueError(
                    """If battery simulation is not operated using an experiment,\n
                an array of times to evaluate simulation at must be passed."""
                )

            self.times = times

        else:
            inputs = dict(
                zip([v.name for v in variables], [v.value for v in variables])
            )
            battery_simulation.solve(inputs=inputs)
            self.times = battery_simulation.solution["Time [s]"].entries

        data = self.simulate(self.true_values)
        self.data = data + np.random.normal(0, self.noise, data.shape)

        for k, v in self.metadata.items():
            print(k)
            print(type(v))

        with open(os.path.join(self.logs_dir_path, "metadata.json"), "w") as outfile:
            print(self.metadata.items())
            outfile.write(json.dumps(self.metadata))

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
            "times": str(self.times),
        }

    def simulate(self, *theta, batch_size=1, random_state=0):
        """
        Simulate method used by pints sampler.
        Parameters
        ----------
        theta: Tuple[float]
            Vector of input variable values.
        batch_size: int
            Batch size used to update GP surrogate
            (for compatibility with ELFI API).
        random_state: int
            Seed (for compatibility with ELFI API).
        Returns
        ----------
        output: np.ndarray
            Voltage time series.
        """
        variable_names = [v.name for v in self.variables]

        theta = np.array(theta).flatten().tolist()

        if not isinstance(theta[0], int) and not isinstance(theta[0], float):
            theta = [t[0] for t in theta]

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

    def plot_pairwise(self):
        plot = self.bolfi.plot_discrepancy()
        fig = plot[0].get_figure()
        fig.savefig(os.path.join(self.logs_dir_path, "discrepancy"))

    def plot_discrepancy(self):
        plot = self.bolfi.plot_discrepancy()
        fig = plot[0].get_figure()
        fig.savefig(os.path.join(self.logs_dir_path, "discrepancy"))

    def run(
            self,
            batch_size: int = 1,
            initial_evidence: int = 50,
            update_interval: int = 10,
            acq_noise_var: float = 0.1,
            n_evidence: int = 1500,
            sampling_iterations: int = 1000,
            n_chains=4,

    ):
        """
        Parameters
        __________
        batch_size: int
            Batch size used to train Gaussian Process (GP) surrogate.
        initial_evidence: int
            Number of number of initialization points (sampled straight
            from the priors before starting to optimize the acquisition
            of points).
        update_interval: float
             How often the GP hyperparameters are optimized.
        acq_noise_var: int
            Diagonal covariance of noise added to the acquired points
            (defaults to 0.1).
        n_evidence: str
            Number of evidence points used to fit GP surrogate
            (including `initial_evidence`).
        sampling_iterations: int
            Number of requested samples from the posterior for each chain.
        n_chains: int
            Number of independent chains.

        Returns
        -------
        chains: np.ndarray
            Sampling chains (shape: iteration, chains, parameters).
        """
        bounds = {var.name: var.bounds for var in self.variables}
        print(bounds)

        model = elfi.ElfiModel()

        for var in self.variables:

            elfi.Prior(distribution=var.prior.distribution, model=model, name=var.name)

        elfi.Simulator(self.simulate, *[model[var.name] for var in self.variables],
                       observed=self.data, name='elfi_simulator')
        sumstats = []
        for i in range(0, len(self.data), self.target_resolution):
            sumstats.append(elfi.Summary(lambda x: x[i], model['elfi_simulator'], name=f'point {i}'))
        elfi.Distance('euclidean', *sumstats, name='euclidean_distance')

        self.bolfi = elfi.BOLFI(elfi.Operation(np.log, model['euclidean_distance']),
                                batch_size=batch_size,
                                initial_evidence=initial_evidence,
                                update_interval=update_interval,
                                acq_noise_var=acq_noise_var,
                                bounds=bounds)
        # Fit the surrogate model
        # (a Gaussian Process regression model for the discrepancy given the parameters.)
        posterior_surrogate = self.bolfi.fit(n_evidence=n_evidence)
        opt_res = {param: value[0] for param, value in self.bolfi.extract_result().x_min.items()}
        sampled_posterior = self.bolfi.sample(sampling_iterations, info_freq=sampling_iterations // 10, n_chains=n_chains)

        with open(
                os.path.join(self.logs_dir_path, "metadata.json"),
                "r",
        ) as outfile:
            metadata = json.load(outfile)

        metadata.update(
            {
                "initial_evidence": initial_evidence,
                "update_interval": update_interval,
                "acq_noise_var": acq_noise_var,
                "sampling_method": "BOLFI",
                "min_discrepancy": opt_res,
                "sample_means_and_95CIs": sampled_posterior.sample_means_and_95CIs,
            }
        )

        chain_columns = [f"p{i}" for i in range(sampled_posterior.chains[0].shape[-1])]
        chain_idx = 0
        for chain in sampled_posterior.chains:
            pd.DataFrame(chain, columns=chain_columns).to_csv(path_or_buf=f"chain_{chain_idx}.csv")
            chain_idx += 1

        with open(
                os.path.join(self.logs_dir_path, "metadata.json"),
                "w",
        ) as outfile:
            json.dump(metadata, outfile)

        return sampled_posterior.chains

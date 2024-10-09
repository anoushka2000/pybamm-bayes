import glob
import json
import os

import numpy as np
import pandas as pd
import pybamm
import tqdm
from IPython.display import Image, display

from pybamm_bayes.analysis.utils import _get_logs_path, sample_from_posterior
from pybamm_bayes.sampling_problems.utils import interpolate_time_over_y_values

TRANSFORMS = {
    "log10": lambda x: 10 ** float(x),
    "None": lambda x: x,
    "negated_log10": lambda x: 10 ** float(-x),
}


def view_data(logs_dir_name=None, logs_dir_path=None):
    """
    Helper function to display data (time series profile) plot in notebook.
    """
    if logs_dir_path is None and logs_dir_name:
        logs_dir_path = _get_logs_path(logs_dir_name)
    path = os.path.join(logs_dir_path, "data.png")
    display(Image(filename=path))


def load_metadata(logs_dir_name=None, logs_dir_path=None):
    """
    Parameters
    ----------
    logs_dir_name: str
       Name of directory logging idenfiability problem results.
    logs_dir_path: str
        Absolute path to directory logging idenfiability problem results.

    Returns
    -------
    Metadata associated with idenfiability problem.
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)

    # load metadata
    with open(os.path.join(logs_dir_path, "metadata.json"), "r") as j:
        metadata = json.loads(j.read())
    return metadata


def load_chains(logs_dir_name=None, logs_dir_path=None, concat=True):
    """
    Parameters
    ----------
    logs_dir_name: str
       Name of directory logging idenfiability problem results.
    logs_dir_path: str
       Absolute path to directory logging idenfiability problem results.
    concat: bool
        Concantenate chain DataFrames if True
        else return list (defaults to True).

    Returns
    -------
    DataFrame of chains. Each column is a parameter being sampled
    (column names 'p0', 'p1' ...).
    Chains are appended sequentially.
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)

    chain_file_names = glob.glob(f"{logs_dir_path}/chain_?.csv")
    df_list = []
    for name in chain_file_names:
        df = pd.read_csv(name)
        if "Unnamed: 0" in df.columns:
            df.drop(columns="Unnamed: 0", inplace=True)
        df_list.append(df)
    if concat:
        chains = pd.concat(df_list)
    else:
        chains = df_list

    return chains


def load_chains_with_residual(logs_dir_name=None, logs_dir_path=None):
    """
    Parameters
    ----------
    logs_dir_name: str
       Name of directory logging idenfiability problem results.
    logs_dir_path: str
       Absolute path to directory logging idenfiability problem results.

    Returns
    -------
    DataFrame of chains and residual .
    Chains are appended interweaved to match order of evaluation.
    (chain 1 sample 1, chain 2 sample 1,...chain n sample 1,...chain n sample n).
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)

    metadata = load_metadata(logs_dir_path=logs_dir_path)

    # recover variable definition from metadata
    variable_names = [
        f"{var['name']}" for var in metadata["variables"]
    ]

    # load chains
    chain_file_names = glob.glob(f"{logs_dir_path}/chain_?.csv")
    df_list = []
    for name in chain_file_names:
        df_list.append(pd.read_csv(name))

    # load residual df
    residuals = pd.read_csv(os.path.join(logs_dir_path, "residuals.csv"))
    result = pd.concat(df_list).sort_index(kind="merge")
    result = result.reset_index()
    result["residuals"] = residuals.residuals

    # calculate chi_sq
    result["chi_sq"] = (
        result.residuals - result.nsmallest(1, "residuals").residuals.values[0]
    )
    try:
        result.columns = ["sample number"] + variable_names + ["residuals", "chi_sq"]
    except ValueError:
        result.columns = (
            ["sample number"]
            + variable_names
            + ["noise estimator", "residuals", "chi_sq"]
        )
    return result


def generate_residual_over_posterior(logs_dir_path, n_evaluations=20):
    """
    Calculates residual between simulated output and data for
    values sampled from posterior distribution.

    Parameters
    __________
    logs_dir_path: str
        Path to directory logging idenfiability problem results.
        Overrides `logs_dir_name` if both are passed.
    n_evaluations: int
        Number of samples from posterior to calculate residual for.
        Defaults to `20`.

    Returns
    _______
    DataFrame with columns for each parameters,
    rows for each combination of parameters sampled and
    a 'Residual' column with the residual for each combination.
    """
    summary = []
    simulation = pybamm.load(os.path.join(logs_dir_path, "battery_simulation"))
    chains = load_chains(logs_dir_path=logs_dir_path)
    metadata = load_metadata(logs_dir_path=logs_dir_path)

    # recover variable definition from metadata
    variable_names = [var["name"] for var in metadata["variables"]]
    output = metadata["output"]
    error_axis = metadata["error axis"]
    times = metadata["t_eval"]
    data = np.fromstring(metadata["data_output"][1:-1], sep=" ")
    ref_axis = np.fromstring(metadata["data_reference_axis_values"][1:-1], sep=" ")

    posterior_samples = sample_from_posterior(chains, n_samples=n_evaluations)

    if len(posterior_samples) > 1:
        for i, input_set in tqdm.tqdm(enumerate(posterior_samples)):

            input_set = [TRANSFORMS[metadata["transform type"]](i) for i in input_set]
            inputs = dict(zip(variable_names, input_set))

            solution = simulation.solve(
                t_eval=times,
                inputs=inputs.copy(),
            )
            solution_var = solution[output].entries

            if error_axis == "x":
                _, solution_var = interpolate_time_over_y_values(
                    times, solution_var, new_y=ref_axis
                )

            summary.append(
                {
                    **inputs,
                    "Residual": abs(data - solution_var).sum() / len(solution_var),
                }
            )
    return pd.DataFrame(summary)


def run_forward_model_over_posterior(
    logs_dir_name=None, logs_dir_path=None, n_evaluations=20
):
    """
    Generates time series curves for parameter values sampled
    from the posterior distribution.

    Parameters
    __________
    logs_dir_name: str
        Name of directory logging identifiability problem results.
    logs_dir_path: str
        Path to directory logging identifiability problem results.
        Overrides `logs_dir_name` if both are passed.
    n_evaluations: int
        Number of samples from posterior to calculate residual for.
        Defaults to `20`.

    Returns
    _______
    DataFrame with columns for output, time and corresponding parameter
    values.
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)
    results = []
    simulation = pybamm.load(os.path.join(logs_dir_path, "battery_simulation"))
    chains = load_chains(logs_dir_path=logs_dir_path)
    metadata = load_metadata(logs_dir_path=logs_dir_path)

    # recover variable definition from metadata
    variable_names = [var["name"] for var in metadata["variables"]]
    output = metadata["output"]

    posterior_samples = sample_from_posterior(chains, n_samples=n_evaluations)

    if len(posterior_samples) > 1:
        for i, input_set in tqdm.tqdm(enumerate(posterior_samples)):
            input_set = [TRANSFORMS[metadata["transform type"]](i) for i in input_set]
            inputs = dict(zip(variable_names, input_set))

            solution = simulation.solve(
                t_eval=np.fromstring(metadata["times"][1:-1], sep=" "),
                inputs=inputs.copy(),
            )
            solution_var = solution[output].entries

            for t, V in zip(solution.t, solution_var):
                results.append(
                    {
                        **inputs,
                        "Reference": t,
                        "Output": V,
                        "run": i,
                    }
                )

    return pd.DataFrame(results)

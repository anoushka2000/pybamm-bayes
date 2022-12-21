import glob
import json
import os

import numpy as np
import pandas as pd
import pybamm
import tqdm
from IPython.display import Image, display

from battery_model_parameterization.Python.analysis.utils import (
    _get_logs_path,
    sample_from_posterior,
)


def view_data(logs_dir_name=None, logs_dir_path=None):
    """
    Helper function to display data (voltage profile) plot in notebook.
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
        f"{metadata['transform type']} {var['name']}" for var in metadata["variables"]
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
    result.columns = ["sample number"] + variable_names + ["residuals", "chi_sq"]
    return result


def generate_residual_over_posterior(logs_dir_path, n_evaluations=20):
    """
    Calculates residual between simulated voltage and data for
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

    data = metadata["data"]

    posterior_samples = sample_from_posterior(chains, n_samples=n_evaluations)

    if len(posterior_samples) > 1:
        for i, input_set in tqdm.tqdm(enumerate(posterior_samples)):
            inputs = dict(zip(variable_names, input_set))

            solution = simulation.solve(
                t_eval=np.fromstring(metadata["times"][1:-1], sep=" "),
                inputs=inputs.copy(),
            )
            solution_V = solution["Terminal voltage [V]"].entries
            summary.append(
                {
                    **inputs,
                    "Residual": abs(data - solution_V).sum() / len(solution_V),
                }
            )
    return pd.DataFrame(summary)


def run_forward_model_over_posterior(
    logs_dir_name=None, logs_dir_path=None, n_evaluations=20
):
    """
    Generates voltage curves for parameter values sampled from posterior distribution.

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
    DataFrame with columns for voltage, time and correspomding parameter
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
    posterior_samples = sample_from_posterior(chains, n_samples=n_evaluations)

    if len(posterior_samples) > 1:
        for i, input_set in tqdm.tqdm(enumerate(posterior_samples)):
            inputs = dict(zip(variable_names, input_set))

            solution = simulation.solve(
                t_eval=np.fromstring(metadata["times"][1:-1], sep=" "),
                inputs=inputs.copy(),
            )
            solution_V = solution["Terminal voltage [V]"].entries

            for t, V in zip(solution.t, solution_V):
                results.append(
                    {
                        **inputs,
                        "Time [s]": t,
                        "Voltage [V]": V,
                        "run": i,
                    }
                )

    return pd.DataFrame(results)

import glob
import json
import os

import elfi  # noqa: F401
import numpy as np
import pandas as pd
import pints  # noqa: F401
import pybamm
import tqdm
from IPython.display import Image, display
from scipy import stats

from battery_model_parameterization.Python.analysis.utils import _get_logs_path


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
        df_list.append(pd.read_csv(name))
    if concat:
        chains = pd.concat(df_list)
    else:
        chains = df_list
    if "Unnamed: 0" in chains.columns:
        chains.drop(columns="Unnamed: 0", inplace=True)
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
    theta_optimal = result.nsmallest(1, "residuals")[["p0", "p1"]].values.flatten()

    # filter to samples within one order of magnitude of true value
    result = result[result.p0 > theta_optimal[0] - 1]
    result = result[result.p0 < theta_optimal[0] + 1]
    result = result[result.p1 > theta_optimal[1] - 1]
    result = result[result.p1 < theta_optimal[1] + 1]
    result["chi_sq"] = (
        result.residuals - result.nsmallest(1, "residuals").residuals.values[0]
    )
    result = result[result.chi_sq < 10]
    result.columns = ["sample number"] + variable_names + ["residuals", "chi_sq"]

    return result


def generate_residual_over_posterior(logs_dir_path, n_evaluations=10):
    summary = []
    simulation = pybamm.load(os.path.join(logs_dir_path, "battery_simulation"))
    chains = load_chains(logs_dir_path=logs_dir_path)
    metadata = load_metadata(logs_dir_path=logs_dir_path)

    # recover variable definition from metadata
    variable_names = [var["name"] for var in metadata["variables"]]

    variable_values = [var["value"] for var in metadata["variables"]]

    data = simulation.solve(
        t_eval=np.fromstring(metadata["times"][1:-1], sep=" "),
        inputs=dict(zip(variable_names, variable_values)),
    )
    data = data["Terminal voltage [V]"].entries

    # Fit a truncated normal distribution to chains
    loc = chains.mean()
    scale = chains.std()
    # Fit a truncated normal distribution to chains
    posterior_samples = stats.truncnorm(
        a=(chains.min() - loc) / scale,
        b=(chains.max() - loc) / scale,
        loc=loc,
        scale=scale,
    ).rvs(size=(n_evaluations, len(chains.columns)))

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

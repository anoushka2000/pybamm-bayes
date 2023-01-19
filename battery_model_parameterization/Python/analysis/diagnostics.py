from battery_model_parameterization.Python.analysis.postprocessing import (
    load_chains,
    load_metadata,
)
from battery_model_parameterization.Python.analysis.utils import _get_logs_path
import numpy as np


def gelman_rubin_convergence_test(logs_dir_name=None, logs_dir_path=None, burnin=500):
    """
    Gelman Rubin statistic to diagnose if a chain has converged.
    (If Gelman Rubin statistic < 1.1, parameter is identifiable).

     Parameters
    ----------
    logs_dir_name: str
       Name of directory logging idenfiability problem results.
    logs_dir_path: str
        Absolute path to directory logging idenfiability problem results.
    burnin: int
        Number of samples to ignore as burn-in in calculation.

    Returns
    --------
    Dictionary with variable name as key and Gelman Rubin statistic as value.
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)
    metadata = load_metadata(logs_dir_path=logs_dir_path)
    variable_names = [var["name"] for var in metadata["variables"]]

    chains = load_chains(logs_dir_path=logs_dir_path, concat=False)
    gelman_rubin_factors = []
    n_chains = len(chains)

    for parameter in chains[0].columns:

        chain_mean_list = []
        chain_var_list = []

        for chain in chains:
            chain = chain[burnin:]

            p_chain = chain[parameter]

            # posterior mean of parameter
            p_mean = p_chain.mean()
            chain_mean_list.append(p_mean)

            # variance of samples in chain for parameter
            intra_chain_var = ((p_chain - p_mean) ** 2).sum() / (len(p_chain) - 1)
            chain_var_list.append(intra_chain_var)

        n_valid_iterations = len(chain)
        mean_all_chains = sum(chain_mean_list) / len(chain_mean_list)
        B = (n_valid_iterations / (n_chains - 1)) * (
            (np.array(chain_mean_list) - mean_all_chains) ** 2
        ).sum()
        W = sum(chain_var_list) / len(chain_var_list)
        V = ((n_valid_iterations - 1) / n_valid_iterations) * W + (n_chains + 1) / (
            n_chains * n_valid_iterations
        ) * B
        gelman_rubin_factors.append(V)
        chains = load_chains(logs_dir_path=logs_dir_path, concat=False)

    gelman_rubin_factors = []

    n_chains = len(chains)

    for parameter in chains[0].columns:

        chain_mean_list = []
        chain_var_list = []

        for chain in chains:
            chain = chain[burnin:]

            p_chain = chain[parameter]

            # posterior mean of parameter
            p_mean = p_chain.mean()
            chain_mean_list.append(p_mean)

            # variance of samples in chain for parameter
            intra_chain_var = ((p_chain - p_mean) ** 2).sum() / (len(p_chain) - 1)
            chain_var_list.append(intra_chain_var)

        n_valid_iterations = len(chain)
        mean_all_chains = sum(chain_mean_list) / len(chain_mean_list)
        B = (n_valid_iterations / (n_chains - 1)) * (
            (chain_mean_list - mean_all_chains) ** 2
        ).sum()
        W = sum(chain_var_list) / len(chain_var_list)
        V = ((n_valid_iterations - 1) / n_valid_iterations) * W + (n_chains + 1) / (
            n_chains * n_valid_iterations
        ) * B
        gelman_rubin_factors.append(V)

    return dict(zip(variable_names, gelman_rubin_factors))

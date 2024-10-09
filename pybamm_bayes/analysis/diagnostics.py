import numpy as np

from pybamm_bayes.analysis.postprocessing import load_chains, load_metadata
from pybamm_bayes.analysis.utils import _get_logs_path

# TODO: correct this
# def gelman_rubin_convergence_test(logs_dir_name=None, logs_dir_path=None, burnin=500):
#     """
#     Gelman Rubin statistic to diagnose if a chain has converged.
#     (If Gelman Rubin statistic < 1.1, parameter is identifiable).

#      Parameters
#     ----------
#     logs_dir_name: str
#        Name of directory logging idenfiability problem results.
#     logs_dir_path: str
#         Absolute path to directory logging idenfiability problem results.
#     burnin: int
#         Number of samples to ignore as burn-in in calculation.

#     Returns
#     --------
#     Dictionary with variable name as key and Gelman Rubin statistic as value.
#     """
#     if logs_dir_path is None:
#         logs_dir_path = _get_logs_path(logs_dir_name)
#     metadata = load_metadata(logs_dir_path=logs_dir_path)
#     variable_names = [var["name"] for var in metadata["variables"]]

#     chains = load_chains(logs_dir_path=logs_dir_path, concat=False)
#     gelman_rubin_factors = []
#     # Number of chains
#     J = len(chains)

#     for parameter in chains[0].columns:

#         chain_mean_list = []
#         chain_var_list = []

#         for chain in chains:
#             chain = chain[burnin:]

#             p_chain = chain[parameter].values

#             # mean of chain
#             p_mean = p_chain.mean()
#             chain_mean_list.append(p_mean)

#             # variance of samples in chain for parameter
#             intra_chain_var = ((p_chain - p_mean) ** 2).sum()
#             chain_var_list.append(intra_chain_var)

#         L = len(chain)
#         # mean of the means of all chains
#         mean_all_chains = sum(chain_mean_list) / len(chain_mean_list)
#         # variance of the means of the chains
#         B = (L / (J - 1)) * (
#             (np.array(chain_mean_list) - mean_all_chains) ** 2
#         ).sum()

#         # averaged variances of the individual chains across all chains
#         W = sum(chain_var_list) / len(chain_var_list)
#         num = W*(L-1)/L + B/L
#         R = num/W
#         gelman_rubin_factors.append(R)
#         chains = load_chains(logs_dir_path=logs_dir_path, concat=False)

#     return dict(zip(variable_names, gelman_rubin_factors))

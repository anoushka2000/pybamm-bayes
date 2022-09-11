import json
import os

import battery_model_parameterization as bmp
import numpy as np
import pandas as pd
import pints


def run_mcmc(
    identifiability_problem: bmp.IdentifiabilityAnalysis,
    burnin: int = 0,
    n_iteration: int = 2000,
    n_chains: int = 12,
    n_workers: int = 4,
    sampling_method: str = "MetropolisRandomWalkMCMC",
):
    """
    Parameters
    __________
    identifiability_problem: IdentifiabilityProblem
        Problem to run MCMC for.
    burnin: int
        Initial iterations discarded from each chain.
    n_iteration: int
        Number of samples per chain.
    n_chains: int
        Number of chain.
    n_workers: int
        Number of parallel processes.
    sampling_method: str
        Name of MCMC sampling class (from pints)
        For a full list of samplers see:
        https://pints.readthedocs.io/en/stable/mcmc_samplers/index.html
        Defaults to MetropolisRandomWalkMCMC.

    Returns
    -------
    chains: np.ndarray
        Sampling chains (shape: iteration, chains, parameters).
    """
    sampling_method = "pints." + sampling_method

    problem = pints.SingleOutputProblem(
        identifiability_problem,
        identifiability_problem.times,
        identifiability_problem.data,
    )
    log_likelihood = pints.GaussianKnownSigmaLogLikelihood(
        problem, identifiability_problem.noise
    )
    log_posterior = pints.LogPosterior(
        log_likelihood, identifiability_problem.log_prior
    )
    xs = [
        x * identifiability_problem.true_values
        for x in np.random.normal(1, 0.2, n_chains)
    ]

    # Create MCMC routine
    mcmc = pints.MCMCController(
        log_posterior, n_chains, xs, method=eval(sampling_method)
    )

    # Add stopping criterion
    mcmc.set_max_iterations(n_iteration)

    # Logging
    mcmc.set_log_to_screen(True)
    mcmc.set_chain_filename(
        os.path.join(identifiability_problem.logs_dir_path, "chain.csv")
    )
    mcmc.set_chain_storage(store_in_memory=True)
    mcmc.set_log_pdf_filename(
        os.path.join(identifiability_problem.logs_dir_path, "log_pdf.csv")
    )

    # Parallelization
    # TODO: ForkingPickler(file, protocol).dump(obj)
    #  TypeError: cannot pickle 'SwigPyObject' object
    # mcmc.set_parallel(parallel=n_workers)

    # Run
    print("Running...")
    chains = mcmc.run()
    print("Done!")

    chains = pd.DataFrame(
        chains.reshape(chains.shape[0] * chains.shape[1], chains.shape[2])
    )

    #  evaluate optimal value for each parameter
    theta_optimal = np.array(
        [float(chains[column].mode().iloc[0]) for column in chains.columns]
    )

    #  find residual at optimal value
    y_hat = identifiability_problem.simulate(
        theta_optimal, times=identifiability_problem.times
    )
    error_at_optimal = np.sum(abs(y_hat - identifiability_problem.data)) / len(
        identifiability_problem.data
    )

    # chi_sq = distance in residuals between optimal value and all others
    pd.DataFrame(
        {
            "residuals": identifiability_problem.residuals,
            "chi_sq": identifiability_problem.residuals - error_at_optimal,
        }
    ).to_csv(os.path.join(identifiability_problem.logs_dir_path, "residuals.csv"))

    with open(
        os.path.join(identifiability_problem.logs_dir_path, "metadata.json"),
        "r",
    ) as outfile:
        metadata = json.load(outfile)

    metadata.update(
        {
            "burnin": burnin,
            "n_iteration": n_iteration,
            "n_chains": n_chains,
            "sampling_method": sampling_method,
            "theta_optimal": theta_optimal.tolist(),
            "error_at_optimal": error_at_optimal,
        }
    )

    with open(
        os.path.join(identifiability_problem.logs_dir_path, "metadata.json"),
        "w",
    ) as outfile:
        json.dump(metadata, outfile)

    return chains

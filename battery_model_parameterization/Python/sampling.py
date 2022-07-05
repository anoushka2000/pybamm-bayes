import json
import os

import numpy as np
import pandas as pd
import pints


def run_mcmc(
    identifiability_problem,
    burnin=500,
    n_iteration=2000,
    n_chains=12,
    n_workers=4,
    sampling_method="pints.MetropolisRandomWalkMCMC",
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
        Name of MCMC sampling method (from pints).

    Returns
    -------
    chains: np.ndarray
        Sampling chains (shape: iteration, chains, parameters).
    """
    with open(
        os.path.join(identifiability_problem.logs_dir_path, "metadata.json"), "r"
    ) as outfile:
        metadata = json.load(outfile)

    metadata.update(
        {
            "burnin": burnin,
            "n_iteration": n_iteration,
            "n_chains": n_chains,
            "sampling_method": sampling_method,
        }
    )

    with open(
        os.path.join(identifiability_problem.logs_dir_path, "metadata.json"), "w"
    ) as outfile:
        json.dump(metadata, outfile)

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

    # Create mcmc routine
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
    # TODO: ForkingPickler(file, protocol).dump(obj) -> TypeError: cannot pickle 'SwigPyObject' object
    # mcmc.set_parallel(parallel=n_workers)

    # Run
    print("Running...")
    chains = mcmc.run()
    print("Done!")
    pd.DataFrame({"chi_sq": identifiability_problem.chi_sq}).to_csv(
        os.path.join(identifiability_problem.logs_dir_path, "chi_sq.csv")
    )

    return chains

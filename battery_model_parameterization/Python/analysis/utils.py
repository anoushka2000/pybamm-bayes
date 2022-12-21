import os

import elfi  # noqa: F401
import pints  # noqa: F401
from scipy import stats


def _get_logs_path(logs_dir_name):
    return os.path.join(os.getcwd(), "logs", logs_dir_name)


def _parse_priors(metadata):
    if metadata["sampling_method"] == "BOLFI":
        priors = [
            eval(
                'elfi.Prior("'
                + v["prior_type"]
                + f'", {v["prior_loc"]}, {v["prior_scale"]})'
            )
            for v in metadata["variables"]
        ]
        bounds = [v["bounds"] for v in metadata["variables"]]
    else:
        priors = [
            eval(
                f"pints.{v['prior_type']}({list(v['prior'].values())[0]},"
                + f"{list(v['prior'].values())[1]})"
            )
            for v in metadata["variables"]
        ]
        bounds = None
    return priors, bounds


def sample_from_prior(metadata, n_samples=7000):
    """
    Parses `pints` or `elfi` prior objects stored in metadata
    and samples from them.
    """
    priors, bounds = _parse_priors(metadata)
    for i in range(len(metadata["variables"])):
        variable = metadata["variables"][i]
        if metadata["sampling_method"] == "BOLFI":
            lower, rng = (
                variable["bounds"][0],
                variable["bounds"][1] - variable["bounds"][0],
            )
            sample = (
                lower
                + priors[i].distribution.rvs(
                    size=n_samples,
                )
                * rng
            )
        else:
            sample = priors[i].sample(n_samples).flatten()
    return sample


def sample_from_posterior(chains, n_samples):
    """
    Approximates posterior distribution by fitting a truncated normal distribution to chains,
    Parameters
    __________
    chains: pd.DataFrame
        DataFrame of sampling chains
        (each column of data frame corresponds contains
         values for one parameter sampled.)
    n_samples: int
        Number of samples to generate from posterior.
    """
    # Fit a truncated normal distribution to chains
    loc = chains.mean()
    scale = chains.std()
    # Fit a truncated normal distribution to chains

    posterior = stats.truncnorm(
        a=(chains.min() - loc) / scale,
        b=(chains.max() - loc) / scale,
        loc=loc,
        scale=scale,
    )
    return posterior.rvs(size=(n_samples, len(chains.columns)))

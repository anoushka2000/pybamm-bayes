import glob
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def _get_logs_path(logs_dir_name):
    return os.path.join(os.getcwd(), "logs", logs_dir_name)


def load_chains(logs_dir_path):
    chain_file_names = glob.glob(f"{logs_dir_path}/chain_?.csv")
    df_list = []
    for name in chain_file_names:
        df_list.append(pd.read_csv(name))
    return pd.concat(df_list)


def plot_chain_convergence(logs_dir_name):
    """
    Evoluation of chains with sampling iterations.
    Parameters
    ----------
    logs_dir_name: str
       Name of directory logging idenfiability problem results.
    """
    logs_dir_path = _get_logs_path(logs_dir_name)
    # load metadata
    with open(os.path.join(logs_dir_path, "metadata.json"), "r") as j:
        metadata = json.loads(j.read())

    # recover variable definition from metadata
    variable_names = [
        f"{metadata['transform type']} {var['name']}" for var in metadata["variables"]
    ]
    true_values = [var["true_value"] for var in metadata["variables"]]

    priors = [
        eval(
            f"pints.{v['prior_type']}({list(v['prior'].values())[0]},/"
            f"{list(v['prior'].values())[1]})"
        )
        for v in metadata["variables"]
    ]

    # load chains
    chains = load_chains(logs_dir_path)
    n_chains = metadata["n_chains"]
    n_param = len(variable_names)
    samples = chains.to_numpy().reshape(n_chains, int(len(chains) / n_chains), n_param)

    # set up figure
    fig, axes = plt.subplots(n_param, 2, figsize=(12, 2 * n_param), squeeze=False)

    # range across all samples
    stacked_chains = np.vstack(samples)
    xmin = np.min(stacked_chains, axis=0)
    xmax = np.max(stacked_chains, axis=0)
    xbins = np.linspace(xmin, xmax, 80)

    for i in range(n_param):
        # variable to store mode across all chains
        max_n = 0

        for j_list, samples_j in enumerate(samples):
            # add histogram subplot
            axes[i, 0].set_xlabel(variable_names[i])
            axes[i, 0].set_ylabel("Frequency")
            n, bins, patches = axes[i, 0].hist(
                samples_j[:, i], bins=xbins[:, i], alpha=0.6
            )

            if max(n) > max_n:
                max_n = int(max(n))

            # add trace subplot
            axes[i, 1].set_xlabel("Iteration")
            axes[i, 1].set_ylabel(variable_names[i])
            axes[i, 1].plot(samples_j[:, i], alpha=0.8)

        # set y limit
        axes[i, 1].set_ylim([round(true_values[i] - 2), round(true_values[i] + 2)])

        # add prior histogram
        axes[i, 0].hist(
            priors[i].sample(int(len(chains) / n_chains)),
            bins=xbins[:, i],
            alpha=0.5,
            color="black",
        )

        # plot true value on histogram
        axes[i, 0].plot([true_values[i], true_values[i]], [0.0, max_n], "--", c="k")

        # plot true value on chain line plot
        xmin_tv, xmax_tv = axes[i, 1].get_xlim()
        axes[i, 1].plot([0.0, xmax_tv], [true_values[i], true_values[i]], "--", c="k")

    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir_path, "chain_convergence"))


def pairwise(logs_dir_name, kde=False, heatmap=False, opacity=None, n_percentiles=None):
    """
    (Adapted from pint.plot.pairwise)
    Creates a set of pairwise scatterplots for all parameters
     and histograms of each individual parameter on the diagonal.

    Parameters
    ----------
    logs_dir_name: str
       Name of directory logging idenfiability problem results.
    kde: bool
        Set to ``True`` to use kernel-density estimation for the
        histograms and scatter plots. Cannot use together with ``heatmap``.
    heatmap: bool
        Set to ``True`` to plot heatmap for the pairwise plots.
        Cannot be used together with ``kde``.
    opacity: float
        This value can be used to manually set the opacity of the
        points in the scatter plots (when ``kde=False`` and ``heatmap=False``
        only).
    n_percentiles: float
        Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.
    """
    logs_dir_path = _get_logs_path(logs_dir_name)
    # load metadata
    with open(os.path.join(logs_dir_path, "metadata.json"), "r") as j:
        metadata = json.loads(j.read())

    # recover variable definition from metadata
    variable_names = [
        f"{metadata['transform type']} {var['name']}" for var in metadata["variables"]
    ]
    true_values = [var["true_value"] for var in metadata["variables"]]
    priors = [
        eval(
            f"pints.{v['prior_type']}({list(v['prior'].values())[0]},/"
            f"{list(v['prior'].values())[1]})"
        )
        for v in metadata["variables"]
    ]

    # load chains
    chain_file_names = glob.glob(f"{logs_dir_path}/chain_?.csv")
    df_list = []
    for name in chain_file_names:
        df_list.append(pd.read_csv(name))

    chains = pd.concat(df_list)
    n_param = len(variable_names)

    samples = chains.to_numpy().reshape(len(chains), n_param)
    # create figure
    fig_size = (3 * n_param, 3 * n_param)
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

    bins = 50
    for i in range(n_param):
        for j in range(n_param):
            if i == j:

                # Diagonal: Plot a histogram
                if n_percentiles is None:
                    xmin, xmax = np.min(samples[:, i]), np.max(samples[:, i])
                else:
                    xmin = np.percentile(samples[:, i], 50 - n_percentiles / 2.0)
                    xmax = np.percentile(samples[:, i], 50 + n_percentiles / 2.0)
                xbins = np.linspace(xmin, xmax, bins)
                axes[i, j].set_xlim(xmin, xmax)

                axes[i, j].hist(samples[:, i], bins=xbins)  # , density=True)

                # add prior histogram
                axes[i, j].hist(
                    priors[i].sample(len(samples)),
                    bins=xbins,
                    alpha=0.5,
                    color="black",
                )

                # Add kde plot
                if kde:
                    x = np.linspace(xmin, xmax, 100)
                    axes[i, j].plot(x, stats.gaussian_kde(samples[:, i])(x))

                # add true values for reference
                ymin_tv, ymax_tv = axes[i, j].get_ylim()
                axes[i, j].plot(
                    [true_values[i], true_values[i]],
                    [0.0, ymax_tv],
                    "--",
                    c="k",
                )

                axes[i, j].set_ylabel("Frequency")

            elif i < j:
                # Top-right: no plot
                axes[i, j].axis("off")

            else:
                # Lower-left: Plot the samples as density map
                if n_percentiles is None:
                    xmin, xmax = np.min(samples[:, j]), np.max(samples[:, j])
                    ymin, ymax = np.min(samples[:, i]), np.max(samples[:, i])
                else:
                    xmin = np.percentile(samples[:, j], 50 - n_percentiles / 2.0)
                    xmax = np.percentile(samples[:, j], 50 + n_percentiles / 2.0)
                    ymin = np.percentile(samples[:, i], 50 - n_percentiles / 2.0)
                    ymax = np.percentile(samples[:, i], 50 + n_percentiles / 2.0)
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].set_ylim(ymin, ymax)

                if not kde and not heatmap:
                    # Create scatter plot
                    # Determine point opacity
                    num_points = len(samples[:, i])
                    if opacity is None:
                        if num_points < 10:
                            opacity = 1.0
                        else:
                            opacity = 1.0 / np.log10(num_points)

                    # Scatter points
                    axes[i, j].scatter(
                        samples[:, j], samples[:, i], alpha=opacity, s=0.1
                    )

                elif kde:
                    # Plot values
                    values = np.vstack([samples[:, j], samples[:, i]])
                    axes[i, j].imshow(
                        np.rot90(values),
                        cmap=plt.cm.Blues,
                        extent=[xmin, xmax, ymin, ymax],
                    )

                    # Create grid
                    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])

                    # Get kernel density estimate and plot contours
                    kernel = stats.gaussian_kde(values)
                    f = np.reshape(kernel(positions).T, xx.shape)
                    axes[i, j].contourf(xx, yy, f, cmap="Blues")
                    axes[i, j].contour(xx, yy, f, colors="k")

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UnicodeWarning)
                        axes[i, j].set_aspect((xmax - xmin) / (ymax - ymin))

                elif heatmap:
                    # Create a heatmap-based plot
                    # Create bins
                    xbins = np.linspace(xmin, xmax, bins)
                    ybins = np.linspace(ymin, ymax, bins)

                    # Plot heatmap
                    axes[i, j].hist2d(
                        samples[:, j],
                        samples[:, i],
                        bins=(xbins, ybins),
                        cmap=plt.cm.Blues,
                    )

                    axes[i, j].set_aspect((xmax - xmin) / (ymax - ymin))

                axes[i, j].plot(
                    [true_values[j], true_values[j]], [ymin, ymax], "--", c="k"
                )
                axes[i, j].plot(
                    [xmin, xmax], [true_values[i], true_values[i]], "--", c="k"
                )

            # Set tick labels
            if i < n_param - 1:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            else:
                # Rotate the x tick labels to fit in the plot
                for tl in axes[i, j].get_xticklabels():
                    tl.set_rotation(45)

            if j > 0:
                # Only show y tick labels for the first column
                axes[i, j].set_yticklabels([])

        # Set axis labels
        axes[-1, i].set_xlabel(variable_names[i])
        if i == 0:
            # The first one is not a parameter
            axes[i, 0].set_ylabel("Frequency")
        else:
            axes[i, 0].set_ylabel(variable_names[i])

    plt.savefig(os.path.join(logs_dir_path, "pairwise_correlation"))

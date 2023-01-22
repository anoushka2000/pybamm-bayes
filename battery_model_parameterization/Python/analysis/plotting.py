import glob
import os
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns
from plotly.subplots import make_subplots
import elfi  # noqa: F401
import pints  # noqa: F401

from battery_model_parameterization.Python.analysis.postprocessing import (
    load_chains,
    load_chains_with_residual,
    load_metadata,
    generate_residual_over_posterior,
    run_forward_model_over_posterior,
)
from battery_model_parameterization.Python.analysis.utils import (
    _get_logs_path,
    _parse_priors,
)


def plot_chain_convergence(logs_dir_name=None, logs_dir_path=None):
    """
    Line plot of sample vs sampling iterations for each chain.
    Parameters
    ----------
    logs_dir_name: str
       Name of directory logging idenfiability problem results.
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)
    metadata = load_metadata(logs_dir_path=logs_dir_path)

    # recover variable definition from metadata
    if metadata["transform type"] == "None":
        transfrom_type = " "
    else:
        transfrom_type = metadata["transform type"]

    variable_names = [
        f"{transfrom_type} {var['name']}" for var in metadata["variables"]
    ]
    true_values = [var["value"] for var in metadata["variables"]]

    priors, bounds = _parse_priors(metadata)

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

            # set x limit for histogram subplot
            axes[i, 0].set_xlim([round(true_values[i] - 2), round(true_values[i] + 2)])

            # add trace subplot
            axes[i, 1].set_xlabel("Iteration")
            axes[i, 1].set_ylabel(variable_names[i])
            axes[i, 1].plot(samples_j[:, i], alpha=0.8)

            # set y limit for trace subplot
            axes[i, 1].set_ylim([round(true_values[i] - 2), round(true_values[i] + 2)])

        # add prior histogram
        if metadata["sampling_method"] == "BOLFI":
            lower, rng = bounds[i][0], bounds[i][1] - bounds[i][0]
            axes[i, 0].hist(
                lower
                + priors[i].distribution.rvs(
                    size=int(len(chains) / n_chains),
                )
                * rng,
                bins=xbins[:, i],
                alpha=0.5,
                color="black",
            )
        else:
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


def compare_chain_convergence(log_dir_names=None, log_dir_paths=None):
    """
    Line plot of sample vs sampling iterations for each chain.
    Parameters
    ----------
    log_dir_names: List[str]
       List of name of directories logging identifiability problem results to compare.

    log_dir_paths: List[str]
       List of paths to directories logging identifiability problem results to compare.
       Overides `log_dir_names`.
    """
    color_list = list(mcolors.TABLEAU_COLORS) * 10
    line_styles = ["-", "--", "-.", ":"] * 10
    if log_dir_paths is None:
        log_dir_paths = [_get_logs_path(name) for name in log_dir_names]
    for i in range(len(log_dir_paths)):
        line_style = line_styles[i]
        color = color_list[i]
        logs_dir_path = log_dir_paths[i]
        metadata = load_metadata(logs_dir_path=logs_dir_path)

        # recover variable definition from metadata
        variable_names = [
            f"{metadata['transform type']} {var['name']}"
            for var in metadata["variables"]
        ]
        true_values = [var["value"] for var in metadata["variables"]]
        priors, bounds = _parse_priors(metadata)

        # load chains
        chains = load_chains(logs_dir_path=logs_dir_path)
        n_chains = metadata["n_chains"]
        n_param = len(variable_names)
        samples = chains.to_numpy().reshape(
            n_chains, int(len(chains) / n_chains), n_param
        )

        if i < 1:
            # set up figure first time
            fig, axes = plt.subplots(
                n_param, 2, figsize=(12, 2 * n_param), squeeze=False
            )

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
                    samples_j[:, i], bins=xbins[:, i], alpha=0.8, color=color
                )

                if max(n) > max_n:
                    max_n = int(max(n))

                # set x limit for histogram subplot
                axes[i, 0].set_xlim(
                    [round(true_values[i] - 2), round(true_values[i] + 2)]
                )

                # add trace subplot
                axes[i, 1].set_xlabel("Iteration")
                axes[i, 1].set_ylabel(variable_names[i])
                axes[i, 1].plot(samples_j[:, i], alpha=0.8, color=color)

                # set y limit for trace subplot
                axes[i, 1].set_ylim(
                    [round(true_values[i] - 2), round(true_values[i] + 2)]
                )

            # add prior histogram
            axes[i, 0].hist(
                priors[i].sample(int(len(chains) / n_chains)),
                bins=xbins[:, i],
                alpha=0.5,
                color="black",
            )

            # plot true value on histogram
            axes[i, 0].plot(
                [true_values[i], true_values[i]], [0.0, max_n], line_style, c="black"
            )

            # plot true value on chain line plot
            xmin_tv, xmax_tv = axes[i, 1].get_xlim()
            axes[i, 1].plot(
                [0.0, xmax_tv], [true_values[i], true_values[i]], line_style, c="black"
            )

    plt.tight_layout()

    # save in each project directory
    for logs_dir_path in log_dir_paths:
        plt.savefig(os.path.join(logs_dir_path, "comparison_chain_convergence"))


def pairwise(
        logs_dir_name=None,
        logs_dir_path=None,
        kde=False,
        heatmap=False,
        opacity=None,
        n_percentiles=None,
):
    """
    (Adapted from pint.plot.pairwise)
    Creates a set of pairwise scatterplots for all parameters
     and histograms of each individual parameter on the diagonal.

    Parameters
    ----------
    logs_dir_name: str
       Name of directory logging idenfiability problem results.
    logs_dir_path: str
        Absolute path to directory logging idenfiability problem results.
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
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)

    metadata = load_metadata(logs_dir_path=logs_dir_path)

    # recover variable definition from metadata
    variable_names = [
        f"{metadata['transform type']} {var['name']}" for var in metadata["variables"]
    ]
    true_values = [var["value"] for var in metadata["variables"]]
    priors, bounds = _parse_priors(metadata)

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

                if metadata["sampling_method"] == "BOLFI":
                    lower, rng = (
                        bounds[i][0],
                        bounds[i][1] - bounds[i][0],
                    )

                    prior_samples = (
                            lower
                            + priors[i].distribution.rvs(
                        size=len(samples),
                    )
                            * rng
                    )
                else:
                    prior_samples = priors[i].sample(len(samples))
                # add prior histogram
                axes[i, j].hist(
                    prior_samples,
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
                    if opacity is None:
                        opacity = 1.0

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


def _plot_confidence_intervals_grid(
        n_variables, logs_dir_name=None, logs_dir_path=None, chi_sq_limit=10
):
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)

    metadata = load_metadata(logs_dir_path=logs_dir_path)
    result = load_chains_with_residual(logs_dir_path=logs_dir_path)

    theta_optimal = result.nsmallest(1, "residuals")[
        result.columns[1: n_variables + 1]
    ].values.flatten()

    # recover true values from metadata
    true_values = [var["value"] for var in metadata["variables"]]

    result = result[result.chi_sq < chi_sq_limit]

    # plotting
    fig = make_subplots(rows=n_variables, cols=n_variables)
    for i in range(n_variables):
        for j in range(n_variables):
            fig.add_trace(
                go.Scattergl(
                    x=result[result.columns[i + 1]],
                    y=result[result.columns[j + 1]],
                    mode="markers",
                    showlegend=False,
                    marker_size=7,
                    marker_colorbar=dict(len=0.7, title="χ2"),
                    marker_color=result.chi_sq,
                    marker_colorscale="agsunset",
                ),
                row=i + 1,
                col=j + 1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=[true_values[i]],
                    y=[true_values[j]],
                    name="True Value",
                    marker_color="grey",
                    mode="markers",
                    showlegend=(i < 1 and j < 1),
                    marker_size=10,
                ),
                row=i + 1,
                col=j + 1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=[theta_optimal[i]],
                    y=[theta_optimal[j]],
                    name="Optimal Value",
                    marker_color="gold",
                    mode="markers",
                    showlegend=(i < 1 and j < 1),
                    marker_symbol="square",
                    marker_size=10,
                ),
                row=i + 1,
                col=j + 1,
            )
            fig.update_xaxes(title=result.columns[i + 1], row=i + 1, col=j + 1)
            fig.update_yaxes(title=result.columns[j + 1], row=i + 1, col=j + 1)
    fig.layout["coloraxis"]["colorbar"]["y"] = 0.4
    fig.update_layout(
        paper_bgcolor="white",
        template="simple_white",
        height=320 * n_variables,
        width=420 * n_variables,
    )

    return fig


def _plot_confidence_intervals_bivariate(
        logs_dir_name=None, logs_dir_path=None, chi_sq_limit=10
):
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)

    result = load_chains_with_residual(logs_dir_path=logs_dir_path)

    theta_optimal = result.nsmallest(1, "residuals")[
        result.columns[1:3]
    ].values.flatten()

    metadata = load_metadata(logs_dir_path=logs_dir_path)

    # recover true values from metadata
    true_values = [var["value"] for var in metadata["variables"]]
    result = result[result.chi_sq < chi_sq_limit]

    # plotting
    fig = px.scatter(
        result,
        result.columns[1],
        result.columns[2],
        color="chi_sq",
        template="simple_white",
        labels={"chi_sq": "χ2"},
    )
    fig.add_trace(
        go.Scattergl(
            x=[true_values[0]],
            y=[true_values[1]],
            name="True Value",
            marker_color="grey",
            mode="markers",
            marker_size=10,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[theta_optimal[0]],
            y=[theta_optimal[1]],
            name="Optimal Value",
            marker_color="gold",
            mode="markers",
            marker_symbol="square",
            marker_size=10,
        )
    )
    fig.layout["coloraxis"]["colorbar"]["y"] = 0.4
    fig.update_layout(coloraxis={"colorscale": "agsunset"})

    return fig


def plot_confidence_intervals(logs_dir_name=None, logs_dir_path=None, chi_sq_limit=10):
    """
    Local confidence regions from sampled parameter pairs.

    Parameters
    ----------
    logs_dir_name: str
       Name of directory logging identifiability problem results.
    logs_dir_path: str
       Path to directory logging idenfiability problem results.
    chi_sq_limit: float
        Plot only parameters with chi_sq < chi_sq_limit
        (allows greater resolution)
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name=logs_dir_name)

    metadata = load_metadata(logs_dir_path=logs_dir_path)
    n_variables = len(metadata["variables"])

    if n_variables < 3:
        # don't need grid
        fig = _plot_confidence_intervals_bivariate(
            logs_dir_path=logs_dir_path, chi_sq_limit=chi_sq_limit
        )
    else:
        fig = _plot_confidence_intervals_grid(
            logs_dir_path=logs_dir_path,
            n_variables=n_variables,
            chi_sq_limit=chi_sq_limit,
        )

    return fig


def plot_residual(
        logs_dir_name=None,
        logs_dir_path=None,
        from_stored=True,
        variables=None,
        n_evaluations=20,
):
    """
    Scatter plot of samples from posterior, coloured by residual
    (mean squared error) between data and simulation using parameter values.

    Parameters
    __________
    logs_dir_name: str
        Name of directory logging idenfiability problem results.
    logs_dir_path: str
        Path to directory logging idenfiability problem results.
        Overrides `logs_dir_name` if both are passed.
    from_stored: bool
        Use pre-calculated residuals (stored in `residual_over_posterior.csv`
        in logs directory.
    variables: List[str]
        List of (two) variable names to plot. Variable names
        should match variable names used when running sampling problem.
    n_evaluations: int
        Number of samples from posterior to calculate residual for.
        Only used if `from_stored` set to `False`.
        Defaults to `20`.
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)

    if from_stored:
        df = pd.read_csv(os.path.join(logs_dir_path, "residual_over_posterior.csv"))
    else:
        df = generate_residual_over_posterior(
            logs_dir_path=logs_dir_path, n_evaluations=n_evaluations
        )
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)

    variables = variables or list(df.columns)
    if "Residual" in variables:
        variables.remove("Residual")

    metadata = load_metadata(logs_dir_path=logs_dir_path)

    # used to generate color bar for residual plot
    residual_plot = plt.scatter(
        df[variables[0]],
        df[variables[1]],
        c=df.Residual,
        cmap=sns.cubehelix_palette(as_cmap=True),
    )
    plt.clf()

    # Set up axis
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    plt.colorbar(residual_plot, ax=ax, label="Residual")

    sns.scatterplot(data=df, x=variables[0], y=variables[1], hue="Residual", ax=ax)

    plt.xlabel(f"{metadata['transform type']} {variables[0]}")
    plt.ylabel(f"{metadata['transform type']} {variables[1]}")
    ax.legend_.remove()

    return fig


def plot_forward_model_posterior_distribution(
        logs_dir_name=None, logs_dir_path=None, from_stored=False, n_evaluations=20
):
    """
    Time series curves for parameter values sampled from posterior,
    coloured by residual parameter values.

    Parameters
    __________
    logs_dir_name: str
        Name of directory logging identifiability problem results.
    logs_dir_path: str
        Path to directory logging identifiability problem results.
        Overrides `logs_dir_name` if both are passed.
    from_stored: bool
        Use pre-calculated residuals (stored in `residual_over_posterior.csv`
        in logs directory.
    n_evaluations: int
        Number of samples from posterior to calculate residual for.
        Only used if `from_stored` set to `False`.
        Defaults to `20`.
    """
    if logs_dir_path is None:
        logs_dir_path = _get_logs_path(logs_dir_name)

    if from_stored:
        df = pd.read_csv(
            os.path.join(logs_dir_path, "forward_model_over_posterior.csv")
        )
    else:
        df = run_forward_model_over_posterior(
            logs_dir_path=logs_dir_path, n_evaluations=n_evaluations
        )

    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)

    metadata = load_metadata(logs_dir_path=logs_dir_path)
    variable_names = [var["name"] for var in metadata["variables"]]
    output = metadata["output"]
    if metadata["error axis"] == "y":
        time_column = "Reference"
        output_column = "Output"

    elif metadata["error axis"] == "x":
        time_column = "Output"
        output_column = "Reference"

    else:
        raise NotImplementedError("'error axis' must be one of: 'x' or 'y'")

    # used to generate color bars
    scratch_plots = []

    for var in variable_names:
        plot = plt.scatter(
            df[time_column],
            df[output_column],
            c=df[var],
            cmap=sns.cubehelix_palette(as_cmap=True),
        )

        scratch_plots.append(plot)
        plt.clf()

    rows = len(variable_names) + 1

    # set up axis
    fig, ax = plt.subplots(rows, 1, figsize=(5, rows * 4))

    for i, var in list(zip([i for i in range(1, rows)], variable_names)):
        # plot simulated profile colored by variable for each variable
        sns.lineplot(
            data=df,
            x=time_column,
            y=output_column,
            hue=df[var],
            ax=ax[i],
            palette=sns.cubehelix_palette(as_cmap=True),
        )
        # add color bar
        plt.colorbar(scratch_plots[i - 1], ax=ax[i], label=var)
        # remove discrete legend
        ax[i].legend_.remove()

    # output with one s.d. plot
    sns.lineplot(data=df, x=time_column, y=output_column, errorbar=("sd", 1), ax=ax[0])

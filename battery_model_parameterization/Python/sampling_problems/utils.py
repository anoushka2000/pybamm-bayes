from scipy.interpolate import interp1d
import numpy as np


def _fmt_variables(variables):
    lst = []
    for v in variables:
        var = v.__dict__.copy()

        var["prior"] = var["prior"].__dict__
        if "_boundaries" in var["prior"].keys():
            var["prior"] = var["prior"]["_boundaries"].__dict__.copy()
        lst.append(var)
    return lst


def _fmt_parameters(parameters):
    return {k: str(v) for k, v in parameters.items()}


def interpolate_time_over_y_values(times, y_values):
    """
    Parameters
    ----------
    times: np.ndarray
        Times at which simulation was evaluated.
    y_values: np.ndarray
        Time series simulation output.

    Returns
    ----------
    new_time: np.ndarray
        Time interpolated over y-axis values.
    """
    y_function = interp1d(x=y_values, y=times)
    min_y = y_values.min()
    max_y = y_values.max()
    new_y = np.linspace(
        start=min_y * (1 + 1e-8),
        stop=max_y * (1 - 1e-8),
        num=int(y_values.shape[0])
    )
    interpolated_time = y_function(new_y)
    return new_y, interpolated_time

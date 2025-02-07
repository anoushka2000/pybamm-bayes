import numpy as np
import inspect
from scipy.interpolate import interp1d


def _fmt_variables(variables):
    lst = []
    for v in variables:
        var = v.__dict__.copy()

        if not isinstance(var["prior"], str):
            var["prior"] = var["prior"].__dict__
            var["inverse_transform"] = inspect.getsourcelines(var["inverse_transform"])[0][0]
            if "_boundaries" in var["prior"].keys():
                var["prior"] = {
                    "lower": var["prior"]["_boundaries"].__dict__["_lower"][0],
                    "upper": var["prior"]["_boundaries"].__dict__["_upper"][0]
                }
        lst.append(var)
    return lst


def _fmt_parameters(parameters):
    return {k: str(v) for k, v in parameters.items()}


def interpolate_time_over_y_values(times, y_values, new_y=None):
    """
    Parameters
    ----------
    times: np.ndarray
        Times at which simulation was evaluated.
    y_values: np.ndarray
        Time series simulation output.
    new_y: np.ndarray
        Array of linearly spaced values to interpolate time over.
        Defaults to
    Returns
    ----------
    new_time: np.ndarray
        Time interpolated over y-axis values.
    """

    min_y = y_values.min()
    max_y = y_values.max()
    if new_y is None:
        new_y = np.linspace(
            start=min_y * (1 + 1e-8),
            stop=max_y * (1 - 1e-8),
            num=int(y_values.shape[0]),
        )
    y_function = interp1d(x=y_values, y=times)
    interpolated_time = y_function(new_y)
    return new_y, interpolated_time

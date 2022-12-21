import os

import elfi
import pints


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
                f"pints.{v['prior_type']}({list(v['prior'].values())[0]},/"
                f"{list(v['prior'].values())[1]})"
            )
            for v in metadata["variables"]
        ]
        bounds = None
    return priors, bounds

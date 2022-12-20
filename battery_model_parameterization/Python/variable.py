import elfi


class Variable:
    """
    Parameter whose value is to be identified.

    Parameters
    ----------
    name: str
        Variable name
    value: float
        For IdentifiabilityAnalysis problems:
         this is the value of the variable used to simulate data.
        For ParameterEstimation problems:
         this is the value around which sampling chains are initialized.
    prior : Union[pints.LogPrior, elfi.Prior]
        Prior from which variable is sampled.
        For `MCMC` methods use the prior must be  a pints.LogPrior object
        e.g. pints.GaussianLogPrior(-5.5, 1)
        https://pints.readthedocs.io/en/stable/log_priors.html
        For `BOLFI` methods the prior must be an elfi.Prior object initialized
        with the distribution and distribution parameter arguments
        e.g elfi.Prior("uniform", 0, 10).
        https://elfi.readthedocs.io/en/latest/api.html#elfi.Prior
    bounds : Optional[Tuple[float, float]]
        A tuple specifying the lower and upper bound of the variable
        (this is required to use the BOLFI method.)
    """

    def __init__(
            self, name, value, prior, bounds=None, prior_loc=None, prior_scale=None
    ):
        self.name = name
        self.value = value
        self.prior = prior
        if isinstance(prior, elfi.model.elfi_model.Prior):
            self.prior_type = prior.distribution.name
            if prior_loc:
                self.prior_loc = prior_loc
            elif prior.distribution.name == "norm":
                self.prior_loc = (bounds[1] + bounds[0]) / 2  # mean is center of bounds
            elif prior.distribution.name == "uniform":
                self.prior_loc = bounds[
                    0
                ]  # loc for uniform distribution is lower bound
            else:
                raise ValueError(
                    "'prior_loc' argument must be provided for elfi Priors\
                     if not Gaussian or Uniform. See scipy.stats \
                     documentation for more information on 'loc' argument to\
                     distributions."
                )

            if prior_scale:
                self.prior_scale = prior_scale
            elif prior.distribution.name == "norm":
                self.prior_scale = (
                        abs(bounds[1] - self.prior_loc) / 5
                )  # 5 standard deviations fall within bounds
            elif prior.distribution.name == "uniform":
                self.prior_scale = abs(bounds[
                    1
                ])  # loc for uniform distribution is upper bound
            else:
                raise ValueError(
                    "'prior_scale' argument must be provided for \
                      elfi Priors if not Normal or Uniform. \
                      See scipy.stats documentation for more\
                      information on 'scale' argument to distributions."
                )
        else:
            self.prior_type = str(type(self.prior)).split(".")[-1][:-2]
        self.bounds = bounds

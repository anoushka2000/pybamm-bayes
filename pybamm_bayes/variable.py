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
        (this is required to use for pints UniformPriors and the BOLFI method.)
    inverse_transform: Optional[Callable]
        Function which transforms sampled values to model parameter values.
        e.g if Diffusivity is sample in log space (log10) 
        `inverse_transform = lambda Ds: 10**Ds`.
    """

    def __init__(
        self,
        name,
        value,
        prior,
        bounds=None,
        inverse_transform = lambda x: x
    ):
        self.name = name
        self.value = value
        self.prior = prior
        self.inverse_transform = inverse_transform
        if isinstance(prior, elfi.model.elfi_model.Prior):
            self.prior_type = prior.distribution.name
            self.prior_loc = self.prior.parents[0].state["attr_dict"]["_output"]
            self.prior_scale = self.prior.parents[1].state["attr_dict"]["_output"]
        elif prior == "uniform":
            self.prior_type = prior
        else:
            self.prior_type = str(type(self.prior)).split(".")[-1][:-2]
        self.bounds = bounds

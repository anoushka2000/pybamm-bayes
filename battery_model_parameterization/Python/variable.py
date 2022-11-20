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
    bounds : Optional[Tuple[float, float]]
        A tuple specifying the lower and upper bound of the variable
        (this is required to use the BOLFI method.)
    """

    def __init__(self, name, value, prior, bounds):
        self.name = name
        self.value = value
        self.prior = prior
        self.prior_type = str(type(self.prior)).split(".")[-1][:-2]
        self.bounds = bounds

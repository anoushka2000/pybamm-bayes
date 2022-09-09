class Variable:
    """
    Parameter whose value is to be identified.

    Parameters
    ----------
    name: str
        Variable name
    value: float
        For IdentifiabilityAnalysis problems this is the value of the variable used to simulate data.
        For ParameterEstimation problems this is the value around which sampling chains are initialized.
    prior : pints.LogPrior
        Prior from which variable is sampled.
    """

    def __init__(self, name, value, prior):
        self.name = name
        self.value = value
        self.prior = prior
        self.prior_type = str(type(self.prior)).split(".")[-1][:-2]
